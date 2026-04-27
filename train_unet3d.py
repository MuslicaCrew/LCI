import argparse
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
)

from unet3d_classifier import UNet3DWithClassifier, CombinedLoss


class NodulePatchDataset(Dataset):
    """
       Precomputed dataset for LUNA16 nodule patch classification.

       Reads from the index.csv produced by precompute_patches.py.
       __getitem__ does only two np.load calls — no CT volumes, no resampling,
       no mask application. Everything heavy was done offline.

       Expected per sample:
           patch      : float32 tensor (1, 64, 64, 64)  — normalised CT patch
           seg_mask   : float32 tensor (1, 64, 64, 64)  — binary spherical mask
           cls_label  : int64 tensor (1,)               — 1 = nodule, 0 = background

       Args:
           index_csv  : path to precomputed/index.csv
           val_fold   : which subset{n} to use as the validation fold (0–9)
                        Pass None to use all subsets (--final mode)
           is_val     : if True, load only val_fold; if False, load all other folds
       """
    def __init__(self, index_csv: str, val_fold: int | None = 0, is_val: bool = False):
        index = pd.read_csv(index_csv)

        # ── Subset filtering ──────────────────────────────────────────
        if val_fold is None:
            # --final mode: train on everything, no validation split
            self.index = index.reset_index(drop=True)
        elif is_val:
            self.index = index[index["subset"] == f"subset{val_fold}"].reset_index(drop=True)
        else:
            self.index = index[index["subset"] != f"subset{val_fold}"].reset_index(drop=True)

        n_pos = int((self.index["label"] == 1).sum())
        n_neg = int((self.index["label"] == 0).sum())
        print(
            f"NodulePatchDataset ({'val' if is_val else 'train'}) : "
            f"{len(self.index):,} samples  ({n_pos:,} pos / {n_neg:,} neg)"
        )

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        row = self.index.iloc[idx]

        # ── Load precomputed .npz files — ~1ms per sample ─────────────
        patch = np.load(row["patch_path"])["patch"]  # float32 (64, 64, 64)
        seg_mask = np.load(row["seg_mask_path"])["seg_mask"]  # float32 (64, 64, 64)

        # ── Add channel dim and convert to tensor ──────────────────────
        patch_tensor = torch.from_numpy(patch).unsqueeze(0)  # (1, 64, 64, 64)
        seg_mask_tensor = torch.from_numpy(seg_mask).unsqueeze(0)  # (1, 64, 64, 64)
        cls_label = int(row["label"])

        return patch_tensor, seg_mask_tensor, cls_label

# ─────────────────────────────────────────────
# TRAINING CONFIG  — edit these before running
# ─────────────────────────────────────────────

CONFIG = {
    "patch_size"          : 64,
    "batch_size"          : 4,         # keep low for 3D — increase if VRAM allows
    "num_epochs"          : 125,
    "learning_rate"       : 1e-4,
    "features"            : 32,        # 32 for >=12 GB VRAM, 16 for 8 GB
    "seg_weight"          : 0.7,
    "cls_weight"          : 0.3,
    "num_workers"         : 8,
    "save_path"           : "best_model.pth",
    "pos_neg_train_ratio" : 3,         # negatives per positive in each training batch
                                       # 3 → 1 pos : 3 neg per batch (25% positive)
                                       # lower = more positive signal, higher = more variety
                                       # val loader is NEVER rebalanced — keeps true prevalence
}


# ─────────────────────────────────────────────
# METRICS HELPERS
# ─────────────────────────────────────────────

def dice_score(pred, target, threshold=0.5, smooth=1e-6):
    """Compute mean Dice score over a batch (after sigmoid)."""
    pred_bin = (pred > threshold).float()
    pred_f   = pred_bin.view(-1)
    tgt_f    = target.view(-1)
    intersection = (pred_f * tgt_f).sum()
    return ((2 * intersection + smooth) / (pred_f.sum() + tgt_f.sum() + smooth)).item()


def binary_accuracy(pred, target, threshold=0.5):
    """Classification accuracy."""
    pred_labels = (pred > threshold).float()
    return (pred_labels == target).float().mean().item()


# ─────────────────────────────────────────────
# ONE EPOCH
# ─────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, scaler, train=True):
    model.train() if train else model.eval()

    total_loss, total_dice, total_acc = 0.0, 0.0, 0.0
    is_cuda = device.type == "cuda"

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for patches, seg_masks, cls_labels in loader:
            patches = patches.to(device)
            seg_masks = seg_masks.to(device)
            cls_labels = cls_labels.to(device)

            if train:
                # set_to_none=True skips the memset-to-zero, freeing memory faster
                optimizer.zero_grad(set_to_none=True)

            # Mixed precision forward pass — ~1.5-2x faster, ~half VRAM on CUDA
            # Falls back to float32 transparently on CPU
            with torch.autocast(device_type=device.type, enabled=is_cuda):
                seg_pred, cls_pred = model(patches)
                loss = criterion(seg_pred, seg_masks, cls_pred.squeeze(1), cls_labels.float())

            if train:
                # scaler handles loss scaling to prevent fp16 underflow
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item()
            # dice_score receives logits — applies sigmoid internally
            total_dice += dice_score(seg_pred.detach(), seg_masks)
            total_acc += binary_accuracy(cls_pred.detach(), cls_labels)

    n = len(loader)
    return total_loss / n, total_dice / n, total_acc / n


# ─────────────────────────────────────────────
# FINAL EVALUATION
# ─────────────────────────────────────────────

def evaluate(model, loader, device):
    """
    Collects full-test-set predictions and prints:
      • Dice score (segmentation)
      • Classification report  (precision / recall / F1)
      • ROC-AUC
      • Confusion matrix plot
    """
    model.eval()

    all_cls_probs, all_cls_labels = [], []
    all_dice = []

    with torch.no_grad():
        for patches, seg_masks, cls_labels in loader:
            patches    = patches.to(device)
            seg_masks  = seg_masks.to(device)
            cls_labels = cls_labels.to(device)

            seg_pred, cls_pred = model(patches)

            all_dice.append(dice_score(seg_pred, seg_masks))
            all_cls_probs.append(cls_pred.cpu().numpy())
            all_cls_labels.append(cls_labels.cpu().numpy())

    cls_probs  = np.concatenate(all_cls_probs).flatten()   # (N,)
    cls_labels = np.concatenate(all_cls_labels).flatten()  # (N,)
    cls_preds  = (cls_probs > 0.5).astype(int)

    mean_dice = np.mean(all_dice)
    auc       = roc_auc_score(cls_labels, cls_probs)

    print("\n" + "=" * 50)
    print("FINAL TEST-SET EVALUATION")
    print("=" * 50)
    print(f"  Mean Dice (segmentation) : {mean_dice:.4f}")
    print(f"  ROC-AUC  (classification): {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(cls_labels, cls_preds, zero_division=0, target_names=["No Nodule", "Nodule"]))

    # Confusion matrix
    cm = confusion_matrix(cls_labels, cls_preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["No Nodule", "Nodule"]).plot(ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix — Test Set")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)

    print("  Confusion matrix saved → confusion_matrix.png")

    return mean_dice, auc

# ─────────────────────────────────────────────
# SAMPLER
# ─────────────────────────────────────────────

def make_weighted_sampler(dataset: NodulePatchDataset, pos_neg_ratio: int) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler that oversamples positives so each training
    epoch sees 1 positive for every pos_neg_ratio negatives.

    Uses replacement=True — required when the target number of positive draws
    exceeds the actual number of positive samples. With ~1,200 positives and
    500K negatives, each positive will be drawn ~(n_neg / pos_neg_ratio / n_pos)
    times per epoch. This is expected — use augmentation to avoid overfitting.

    The val loader never uses this sampler: it sees the true 1:458 prevalence
    so that val loss and AUC reflect real-world detection difficulty.
    """
    labels = dataset.index["label"].values  # numpy array, fast

    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())

    # Weight each sample inversely proportional to its class frequency,
    # then scale so the pos:neg draw ratio matches pos_neg_ratio
    weight_pos = 1.0
    weight_neg = 1.0 / pos_neg_ratio  # negatives drawn pos_neg_ratio× less often

    sample_weights = np.where(labels == 1, weight_pos, weight_neg).tolist()

    # num_samples controls epoch length — keep it equal to dataset size so
    # one "epoch" still means one full pass over the negatives on average
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True,
    )

    pos_frac = 1.0 / (1.0 + pos_neg_ratio)
    print(
        f"WeightedRandomSampler : {n_pos:,} pos / {n_neg:,} neg  →  "
        f"~{pos_frac*100:.0f}% positive per batch  "
        f"(each positive drawn ~{n_neg / pos_neg_ratio / max(n_pos, 1):.0f}× per epoch)"
    )
    return sampler

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_fold', type=int, default=0)
    parser.add_argument('--final', action='store_true')
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    index_csv = "precomputed/index.csv"
    test_loader = None
    train_loader = None

    if args.final:
        train_ds = NodulePatchDataset(index_csv, val_fold=None, is_val=False)
        train_sampler = make_weighted_sampler(train_ds, CONFIG["pos_neg_train_ratio"])
        train_loader = DataLoader(
            train_ds,
            batch_size=CONFIG["batch_size"],
            sampler=train_sampler,  # sampler and shuffle are mutually exclusive
            num_workers=CONFIG["num_workers"],
            pin_memory=device.type == "cuda",
        )
    else:
        train_ds = NodulePatchDataset(index_csv, val_fold=args.val_fold, is_val=False)
        test_ds = NodulePatchDataset(index_csv, val_fold=args.val_fold, is_val=True)
        train_sampler = make_weighted_sampler(train_ds, CONFIG["pos_neg_train_ratio"])

        train_loader = DataLoader(
            train_ds,
            batch_size=CONFIG["batch_size"],
            sampler=train_sampler,  # sampler handles shuffling — do not set shuffle=True
            num_workers=CONFIG["num_workers"],
            pin_memory=device.type == "cuda",
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=CONFIG["batch_size"],
            shuffle=False,  # val: natural prevalence, no rebalancing
            num_workers=CONFIG["num_workers"],
            pin_memory=device.type == "cuda",
        )


    # ── Model, loss, optimiser ──────────────────────────────────────
    print(f"Setting scheduler, optimizer and criterion")
    model     = UNet3DWithClassifier(features=CONFIG["features"]).to(device)
    criterion = CombinedLoss(CONFIG["seg_weight"], CONFIG["cls_weight"])
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=15
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}\n")

    # ── Training loop ───────────────────────────────────────────────
    history = {"train_loss": [], "val_loss": [], "train_dice": [],
               "val_dice": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")

    for epoch in range(1, CONFIG["num_epochs"] + 1):
        tr_loss, tr_dice, tr_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )

        if test_loader is not None:
            vl_loss, vl_dice, vl_acc = run_epoch(
                model, test_loader, criterion, optimizer, device, train=False
            )
        else:
            vl_loss, vl_dice, vl_acc = 0.0, 0.0, 0.0
        scheduler.step(vl_loss)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_dice"].append(tr_dice)
        history["val_dice"].append(vl_dice)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        print(
            f"Epoch [{epoch:02d}/{CONFIG['num_epochs']}]  "
            f"Loss → train: {tr_loss:.4f}  val: {vl_loss:.4f}  |  "
            f"Dice → train: {tr_dice:.4f}  val: {vl_dice:.4f}  |  "
            f"Acc  → train: {tr_acc:.3f}  val: {vl_acc:.3f}"
        )

        # Save best model
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            torch.save(model.state_dict(), CONFIG["save_path"])
            print(f"  ✓ Best model saved (val_loss: {best_val_loss:.4f})")

    # ── Learning curves ─────────────────────────────────────────────
    epochs = range(1, CONFIG["num_epochs"] + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"],   label="Val")
    axes[0].set_title("Combined Loss"); axes[0].legend(); axes[0].set_xlabel("Epoch")

    axes[1].plot(epochs, history["train_dice"], label="Train")
    axes[1].plot(epochs, history["val_dice"],   label="Val")
    axes[1].set_title("Dice Score (Segmentation)"); axes[1].legend(); axes[1].set_xlabel("Epoch")

    axes[2].plot(epochs, history["train_acc"], label="Train")
    axes[2].plot(epochs, history["val_acc"],   label="Val")
    axes[2].set_title("Accuracy (Classification)"); axes[2].legend(); axes[2].set_xlabel("Epoch")

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    print("Learning curves saved → training_curves.png")

    # ── Load best weights and run full evaluation ───────────────────
    model.load_state_dict(torch.load(CONFIG["save_path"], map_location=device))
    evaluate(model, test_loader, device)


if __name__ == "__main__":
    main()
