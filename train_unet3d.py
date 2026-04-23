import argparse
import glob
import os

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
)

# Import your model and losses
from unet3d_classifier import UNet3DWithClassifier, CombinedLoss


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────

class NodulePatchDataset(Dataset):
    """
    Replace the body of __init__ and __getitem__ with your real data loading logic.

    Expected per sample:
        patch      : float32 tensor (1, 64, 64, 64)  — normalised CT patch
        seg_mask   : float32 tensor (1, 64, 64, 64)  — binary voxel mask
        cls_label  : float32 tensor (1,)             — 1.0 = nodule, 0.0 = background

    Synthetic data is generated here so you can run the script immediately and
    verify the pipeline before swapping in real data.
    """
    def __init__(self, patch_size, data_dir, candidates, annotations, val_fold=0, is_val=False):
        self.patch_size  = patch_size

        self.candidates = pd.read_csv(candidates)
        self.annotations = pd.read_csv(annotations)
        self.num_samples = len(candidates)

        # Only grab scans from the relevant subsets
        all_subsets = [f"subset{i}" for i in range(10)]

        if is_val:
            active_subsets = [f"subset{val_fold}"]
        else:
            active_subsets = [s for s in all_subsets if s != f"subset{val_fold}"]

        # Only glob the active subsets
        mhd_files = []
        for subset in active_subsets:
            mhd_files += glob.glob(os.path.join(data_dir, subset, "*.mhd"))

        self.uid_to_path = {os.path.splitext(os.path.basename(f))[0]: f for f in mhd_files}

        # Filter candidates to only those in active subsets
        self.candidates = self.candidates[
            self.candidates['seriesuid'].isin(self.uid_to_path.keys())
        ]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        pass
        #img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        #image = decode_image(img_path)
        #label = self.img_labels. [idx, 1]
        #if self.transform:
            #image = self.transform(image)
        #if self.target_transform:
            #label = self.target_transform(label)
        #return image, label
        #numpy_image, numpy_origin, numpy_spacing = preprocess.load_itk_image


# ─────────────────────────────────────────────
# TRAINING CONFIG  — edit these before running
# ─────────────────────────────────────────────

CONFIG = {
    "num_samples"  : 200,       # total dataset size
    "patch_size"   : 64,
    "batch_size"   : 2,         # keep low for 3D — increase if VRAM allows
    "num_epochs"   : 10,
    "learning_rate": 1e-4,
    "features"     : 16,        # 32 for >=12 GB VRAM, 16 for 8 GB
    "seg_weight"   : 0.7,
    "cls_weight"   : 0.3,
    "num_workers"  : 0,         # set to 4+ on Linux with real data
    "save_path"    : "best_model.pth",
    "train_ratio"  : 0.7,       # 70 / 30 split
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

def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()

    total_loss, total_dice, total_acc = 0.0, 0.0, 0.0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for patches, seg_masks, cls_labels in loader:
            patches    = patches.to(device)
            seg_masks  = seg_masks.to(device)
            cls_labels = cls_labels.to(device)

            seg_pred, cls_pred = model(patches)
            loss = criterion(seg_pred, seg_masks, cls_pred, cls_labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_dice += dice_score(seg_pred.detach(), seg_masks)
            total_acc  += binary_accuracy(cls_pred.detach(), cls_labels)

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
    plt.show()
    print("  Confusion matrix saved → confusion_matrix.png")

    return mean_dice, auc


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_fold', type=int, default=0)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Args : {args.val_fold}")

    data_dir = "data"
    candidates = "candidates.csv"
    annotations = "annotations.csv"
    patch_size = 64

    #10 fold cross validation
    train_ds   = NodulePatchDataset(patch_size, data_dir, candidates, annotations,
    val_fold=args.val_fold,
    is_val=False)
    test_ds    = NodulePatchDataset(patch_size, data_dir, candidates, annotations,
    val_fold=args.val_fold,
    is_val=True)


    train_loader = DataLoader(
        train_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_ds,
        batch_size  = CONFIG["batch_size"],
        shuffle     = False,
        num_workers = CONFIG["num_workers"],
        pin_memory  = device.type == "cuda",
    )

    # ── Model, loss, optimiser ──────────────────────────────────────
    model     = UNet3DWithClassifier(features=CONFIG["features"]).to(device)
    criterion = CombinedLoss(CONFIG["seg_weight"], CONFIG["cls_weight"])
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
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
        vl_loss, vl_dice, vl_acc = run_epoch(
            model, test_loader,  criterion, optimizer, device, train=False
        )
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
    plt.show()
    print("Learning curves saved → training_curves.png")

    # ── Load best weights and run full evaluation ───────────────────
    model.load_state_dict(torch.load(CONFIG["save_path"], map_location=device))
    evaluate(model, test_loader, device)


if __name__ == "__main__":
    main()
