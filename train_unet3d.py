import argparse
import os

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
)
import torch.nn.functional as Functional
from unet3d_classifier import UNet3DWithClassifier, CombinedLoss

import warnings
warnings.filterwarnings("ignore", message="Can't initialize amdsmi")
torch.set_float32_matmul_precision('high')  # ← add here


# ─────────────────────────────────────────────
# AUGMENTATION
# ─────────────────────────────────────────────

class Augment3D:
    """
    Online augmentation for 3D CT patches. Applied at __getitem__ time
    on training samples only, never on validation.

    All spatial transforms are applied identically to both patch and seg_mask
    so they stay spatially aligned. Intensity jitter is applied to the
    patch only — the mask is binary and must not be blurred or shifted.

    Transforms:
        Random flips   : each axis flipped independently with flip_prob.
                         Biologically valid — nodules are axis-symmetric.
        Random rotation: uniform angle in [-max_angle_deg, +max_angle_deg]
                         per axis, applied as a single affine grid warp.
                         Keep small (<=20 deg) to avoid boundary artifacts.
        Intensity jitter: Gaussian noise + random brightness shift on the
                         patch only. Simulates scanner variability.

    Args:
        max_angle_deg  : max rotation per axis in degrees (default 15)
        noise_std      : std of additive Gaussian noise (default 0.01)
        brightness_max : max absolute brightness shift (default 0.05)
        flip_prob      : per-axis flip probability (default 0.5)
    """
    def __init__(
        self,
        max_angle_deg:  float = 15.0,
        noise_std:      float = 0.01,
        brightness_max: float = 0.05,
        flip_prob:      float = 0.5,
    ):
        self.max_angle_rad = max_angle_deg * (torch.pi / 180.0)
        self.noise_std      = noise_std
        self.brightness_max = brightness_max
        self.flip_prob      = flip_prob

    def __call__(
        self,
        patch: torch.Tensor,   # (1, D, H, W) float32
        mask:  torch.Tensor,   # (1, D, H, W) float32
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # ── Random flips ─────────────────────────────────────────────
        # dims 1, 2, 3 correspond to D, H, W on a (1, D, H, W) tensor
        for dim in [1, 2, 3]:
            if torch.rand(1).item() < self.flip_prob:
                patch = torch.flip(patch, dims=[dim])
                mask  = torch.flip(mask,  dims=[dim])

        # ── Random rotation ───────────────────────────────────────────
        # One angle per axis, sampled uniformly in [-max_angle_rad, +max_angle_rad]
        angles = (torch.rand(3) * 2 - 1) * self.max_angle_rad

        cos_z, sin_z = torch.cos(angles[0]), torch.sin(angles[0])
        cos_y, sin_y = torch.cos(angles[1]), torch.sin(angles[1])
        cos_x, sin_x = torch.cos(angles[2]), torch.sin(angles[2])

        Rz = torch.tensor([
            [cos_z, -sin_z, 0.0],
            [sin_z,  cos_z, 0.0],
            [0.0,    0.0,   1.0],
        ], dtype=torch.float32)
        Ry = torch.tensor([
            [ cos_y, 0.0, sin_y],
            [ 0.0,   1.0, 0.0  ],
            [-sin_y, 0.0, cos_y],
        ], dtype=torch.float32)
        Rx = torch.tensor([
            [1.0, 0.0,    0.0   ],
            [0.0, cos_x, -sin_x ],
            [0.0, sin_x,  cos_x ],
        ], dtype=torch.float32)

        # Combined rotation — zero translation column appended for affine_grid
        R     = Rz @ Ry @ Rx
        theta = torch.cat([R, torch.zeros(3, 1)], dim=1).unsqueeze(0)  # (1, 3, 4)

        # affine_grid and grid_sample require a batch dim: (1, 1, D, H, W)
        patch_b = patch.unsqueeze(0)
        mask_b  = mask.unsqueeze(0)

        grid = Functional.affine_grid(theta, patch_b.shape, align_corners=False)

        # bilinear for the patch — preserves smooth intensity values
        patch_b = Functional.grid_sample(patch_b, grid, mode="bilinear",
                                padding_mode="zeros", align_corners=False)
        # nearest for the mask — keeps values binary, no interpolation blur
        mask_b  = Functional.grid_sample(mask_b, grid, mode="nearest",
                                padding_mode="zeros", align_corners=False)

        patch = patch_b.squeeze(0)
        mask  = mask_b.squeeze(0)

        # ── Intensity jitter (patch only, mask untouched) ─────────────
        patch = patch + torch.randn_like(patch) * self.noise_std
        patch = patch + (torch.rand(1).item() * 2 - 1) * self.brightness_max
        patch = patch.clamp(0.0, 1.0)

        return patch, mask


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
    def __init__(self,
        index_csv: str,
        val_fold:  int | None = 0,
        is_val:    bool = False,
        augment:   bool = False,):
        index = pd.read_csv(index_csv)

        # ── Subset filtering ──────────────────────────────────────────
        if val_fold is None:
            # --final mode: train on everything, no validation split
            self.index = index.reset_index(drop=True)
        elif is_val:
            self.index = index[index["subset"] == f"subset{val_fold}"].reset_index(drop=True)
        else:
            self.index = index[index["subset"] != f"subset{val_fold}"].reset_index(drop=True)

        self.augmenter = Augment3D() if augment else None

        n_pos = int((self.index["label"] == 1).sum())
        n_neg = int((self.index["label"] == 0).sum())
        aug_str = "augment=ON" if augment else "augment=OFF"
        print(
            f"NodulePatchDataset ({'val' if is_val else 'train'}, {aug_str}) : "
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
        patch_tensor = torch.from_numpy(patch.astype(np.float32)).unsqueeze(0)  # (1, 64, 64, 64)
        seg_mask_tensor = torch.from_numpy(seg_mask).unsqueeze(0)  # (1, 64, 64, 64)
        cls_label = int(row["label"])

        # Spatial transforms applied to both; intensity jitter patch only.
        if self.augmenter is not None:
            patch_tensor, seg_mask_tensor = self.augmenter(patch_tensor, seg_mask_tensor)

        return patch_tensor, seg_mask_tensor, cls_label

# ─────────────────────────────────────────────
# TRAINING CONFIG  — edit these before running
# ─────────────────────────────────────────────

CONFIG = {
    "patch_size"          : 64,
    "batch_size"          : 32,         # keep low for 3D — increase if VRAM allows
    "num_epochs"          : 150,
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
    "early_stop_patience" : 20,    # epochs without improvement before stopping
    "early_stop_min_delta": 1e-4,  # minimum improvement to count as "better"
    "early_stop_min_epoch": 20,    # don't stop before this epoch regardless
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
    print(f"We are training: {train}")
    model.train() if train else model.eval()

    total_loss, total_dice, total_acc = 0.0, 0.0, 0.0
    is_cuda = device.type == "cuda"
    n_batches = len(loader)

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch_idx, (patches, seg_masks, cls_labels) in enumerate(loader, 1):   # ← enumerate starting at 1
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
                scaler.unscale_(optimizer)  # ← add this
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item()
            total_dice += dice_score(torch.sigmoid(seg_pred).detach(), seg_masks)
            total_acc += binary_accuracy(torch.sigmoid(cls_pred).detach(), cls_labels)
            print(f"  Batch [{batch_idx}/{n_batches}]  loss: {loss.item():.4f}", end="\r")  # ← print on same line


    print()  # ← newline after the \r loop finishes
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
    print("Evaluating model...")
    model.eval()
    n_batches = len(loader)
    all_cls_probs, all_cls_labels = [], []
    all_dice = []
    with torch.no_grad():
        for batch_idx, (patches, seg_masks, cls_labels) in enumerate(loader, 1):
            patches    = patches.to(device)
            seg_masks  = seg_masks.to(device)
            cls_labels = cls_labels.to(device)

            seg_pred, cls_pred = model(patches)

            all_dice.append(dice_score(torch.sigmoid(seg_pred), seg_masks))
            all_cls_probs.append(cls_pred.cpu().numpy())
            all_cls_labels.append(cls_labels.cpu().numpy())
            print(f"  Batch [{batch_idx}/{n_batches}]", end="\r")  # ← print on same line

    print()  # ← newline after loop
    print("Flatten")
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
        num_samples=n_pos * (1+pos_neg_ratio) * 6,
        replacement=True,
    )

    pos_frac = 1.0 / (1.0 + pos_neg_ratio)
    print(
        f"WeightedRandomSampler : {n_pos:,} pos / {n_neg:,} neg  →  "
        f"~{pos_frac*100:.0f}% positive per batch  "
        f"(each positive drawn ~{n_neg / pos_neg_ratio / max(n_pos, 1):.0f}× per epoch)"
    )
    return sampler

def save_patch_previews(dataset: NodulePatchDataset, n: int = 10, out_dir: str = "patch_previews") -> None:
    """
    Save n central slices of patches and their seg masks as PNG images.

    Samples the first n items from the dataset directly (no sampler) so
    you get a deterministic, fast sanity check before training starts.
    Saves side-by-side: patch slice on the left, mask slice on the right.

    For each sample i:
        patch_previews/sample_{i:02d}_pos.png  (label=1)
        patch_previews/sample_{i:02d}_neg.png  (label=0)

    Output: n PNG files, one per sample, named by index and label.
    """
    os.makedirs(out_dir, exist_ok=True)

    for i in range(min(n, len(dataset))):
        patch_t, mask_t, label = dataset[i]

        # Central slice along the z (depth) axis
        z_mid = patch_t.shape[1] // 2
        patch_slice = patch_t[0, z_mid].numpy()   # (H, W) float32 in [0, 1]
        mask_slice  = mask_t[0,  z_mid].numpy()   # (H, W) float32 binary

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        fig.suptitle(f"Sample {i:02d} — label={'NODULE' if label == 1 else 'negative'}", fontsize=12)

        axes[0].imshow(patch_slice, cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("CT patch (central slice)")
        axes[0].axis("off")

        axes[1].imshow(patch_slice, cmap="gray", vmin=0, vmax=1)   # CT as background
        axes[1].imshow(mask_slice,  cmap="Reds", alpha=0.5, vmin=0, vmax=1)  # mask overlay
        axes[1].set_title("Seg mask overlay")
        axes[1].axis("off")

        plt.tight_layout()

        label_str = "pos" if label == 1 else "neg"
        save_path = os.path.join(out_dir, f"sample_{i:02d}_{label_str}.png")
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved {min(n, len(dataset))} patch previews → {out_dir}/")


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

    index_csv = "~/precomputed/index.csv"
    test_loader = None
    train_loader = None
    N_VAL_NEG = 4_000
    patience_counter = 0

    if args.final:
        train_ds = NodulePatchDataset(index_csv, val_fold=None, is_val=False, augment=True)
        train_sampler = make_weighted_sampler(train_ds, CONFIG["pos_neg_train_ratio"])
        train_loader = DataLoader(
            train_ds,
            batch_size=CONFIG["batch_size"],
            sampler=train_sampler,  # sampler and shuffle are mutually exclusive
            num_workers=CONFIG["num_workers"],
            pin_memory=device.type == "cuda",
        )
    else:
        train_ds = NodulePatchDataset(index_csv, val_fold=args.val_fold, is_val=False, augment=True)
        test_ds = NodulePatchDataset(index_csv, val_fold=args.val_fold, is_val=True)
        train_sampler = make_weighted_sampler(train_ds, CONFIG["pos_neg_train_ratio"])

        # ── Stratified val subset ─────────────────────────────────────
        val_index = test_ds.index
        pos_idx = val_index[val_index["label"] == 1].index.tolist()  # all ~1200 positives

        train_loader = DataLoader(
            train_ds,
            batch_size=CONFIG["batch_size"],
            sampler=train_sampler,  # sampler handles shuffling — do not set shuffle=True
            num_workers=CONFIG["num_workers"],
            pin_memory=device.type == "cuda",
        )

    #print(f"Saving pictures...")
    # ── Patch preview — sanity check before training ──────────────────
    #save_patch_previews(train_ds, n=100, out_dir="patch_previews")
    #print(f"Done saving")
    #exit()

    # ── Model, loss, optimiser ──────────────────────────────────────
    print(f"Setting scheduler, otimizer and criterion")
    model = UNet3DWithClassifier(features=CONFIG["features"]).to(device)
    #model = torch.compile(model, mode='reduce-overhead')
    model = torch.compile(model, mode='default')
    criterion = CombinedLoss(CONFIG["seg_weight"], CONFIG["cls_weight"])
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    # GradScaler is a no-op when enabled=False (CPU), so safe to always create
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=15, min_lr=1e-6
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}\n")

    # ── Training loop ───────────────────────────────────────────────
    history = {"train_loss": [], "val_loss": [], "train_dice": [],
               "val_dice": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    for epoch in range(1, CONFIG["num_epochs"] + 1):
        print(f"Epoch : {epoch}")
        # ── Rebuild val loader with fresh negatives each epoch ────────────
        if not args.final:  # ← guard
            neg_idx = val_index[val_index["label"] == 0].sample(
                n=N_VAL_NEG
            ).index.tolist()
            subset_idx = pos_idx + neg_idx
            val_subset = torch.utils.data.Subset(test_ds, subset_idx)
            test_loader = DataLoader(
                val_subset,
                batch_size=CONFIG["batch_size"],
                shuffle=False,
                num_workers=CONFIG["num_workers"],
                pin_memory=device.type == "cuda",
                persistent_workers=True,
            )

        tr_loss, tr_dice, tr_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, scaler, train=True
        )

        if test_loader is not None:
            vl_loss, vl_dice, vl_acc = run_epoch(
                model, test_loader, criterion, optimizer, device, scaler, train=False
            )
            scheduler.step(vl_loss)
        else:
            vl_loss, vl_dice, vl_acc = 0.0, 0.0, 0.0

        if epoch >= CONFIG["early_stop_min_epoch"] and test_loader is not None:
            if vl_dice > best_val_dice + CONFIG["early_stop_min_delta"]:
                best_val_dice = vl_dice

            # save best checkpoint here
            else:
                patience_counter += 1

            if patience_counter >= CONFIG["early_stop_patience"]:
               print(f"\nEarly stopping triggered at epoch {epoch}")
               print(f"Best val Dice: {best_val_dice:.4f}")
               break


        print("Before history")
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
        if  test_loader is not None and vl_loss < best_val_loss:
            best_val_dice = vl_dice
            torch.save(model.state_dict(), CONFIG["save_path"])
            print(f"  ✓ Best model saved (vl_dice: {best_val_dice:.4f})")

    print("Creating plots")
    # ── Learning curves ─────────────────────────────────────────────
    epochs = range(1, len(history["train_loss"]) + 1)
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
    if test_loader is not None:
        print("Loading best state...")
        model.load_state_dict(torch.load(CONFIG["save_path"], map_location=device, weights_only=True))
        evaluate(model, test_loader, device)


if __name__ == "__main__":
    main()
