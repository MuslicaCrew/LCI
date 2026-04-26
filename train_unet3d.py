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

import preprocess
# Import your model and losses
from unet3d_classifier import UNet3DWithClassifier, CombinedLoss


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────

# def make_seg_mask(diameter_mm, spacing, patch_size=64):
#     mask = np.zeros((patch_size, patch_size, patch_size), dtype=np.float32)
#
#     if diameter_mm == 0:  # negative candidate
#         return mask
#
#     radius_voxels = (diameter_mm / 2) / spacing  # per axis, accounts for anisotropic spacing
#     center = patch_size // 2  # always the center of the patch
#
#     for z in range(patch_size):
#         for y in range(patch_size):
#             for x in range(patch_size):
#                 dist = np.sqrt(
#                     ((z - center) / radius_voxels[0]) ** 2 +
#                     ((y - center) / radius_voxels[1]) ** 2 +
#                     ((x - center) / radius_voxels[2]) ** 2
#                 )
#                 if dist <= 1.0:
#                     mask[z, y, x] = 1.0
#     return mask

def make_seg_mask(diameter_mm, spacing, patch_size=64):
    mask = np.zeros((patch_size, patch_size, patch_size), dtype=np.float32)

    if diameter_mm == 0:
        return mask

    radius_voxels = (diameter_mm / 2) / spacing
    center = patch_size // 2

    zz, yy, xx = np.mgrid[0:patch_size, 0:patch_size, 0:patch_size]
    dist = np.sqrt(
        ((zz - center) / radius_voxels[0]) ** 2 +
        ((yy - center) / radius_voxels[1]) ** 2 +
        ((xx - center) / radius_voxels[2]) ** 2
    )
    mask[dist <= 1.0] = 1.0
    return mask


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
    def __init__(self, patch_size, data_dir, mask_dir, candidates, annotations, val_fold=0, is_val=False):
        self.patch_size  = patch_size
        self.mask_dir = mask_dir
        self.candidates = pd.read_csv(candidates)
        self.annotations = pd.read_csv(annotations)




        # Only grab scans from the relevant subsets
        all_subsets = [f"subset{i}" for i in range(10)]

        if val_fold is None:
            active_subsets = all_subsets  # use everything
        elif is_val:
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
        self.available_masks = set(
            os.path.splitext(os.path.basename(f))[0]
            for f in glob.glob(os.path.join(self.mask_dir, "*.mhd"))
        )

        # Balance classes by undersampling negatives
        positives = self.candidates[self.candidates['class'] == 1]
        negatives = self.candidates[self.candidates['class'] == 0].sample(
            n=min(len(positives) * 10, len(self.candidates[self.candidates['class'] == 0])),
            random_state=42
        )
        self.candidates = pd.concat([positives, negatives]).sample(frac=1, random_state=42).reset_index(drop=True)
        self.num_samples = len(self.candidates)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Get the candidate ro
        print(f"At idx: {idx}")
        candidate = self.candidates.iloc[idx]
        seriesuid = candidate['seriesuid']
        coords = np.array([candidate['coordX'], candidate['coordY'], candidate['coordZ']])
        label = int(candidate['class'])

        # 1. Load .mhd file
        mhd_path = self.uid_to_path[seriesuid]

        ct_array, origin, spacing = preprocess.load_itk_image(mhd_path)

        if seriesuid in self.available_masks:
            mask_path = os.path.join(self.mask_dir, os.path.basename(mhd_path))
            mask_array, _, _ = preprocess.load_itk_image(mask_path)
            ct_array[mask_array == 0] = -1000

        ct_array = preprocess.normalize_planes(ct_array)

        # Pad before slicing to handle border candidates

        voxel_coords = preprocess.world_to_voxel_coordinates(coords, origin, spacing)

        # Shift coords to account for padding
        z = int(round(voxel_coords[2]))
        y = int(round(voxel_coords[1]))
        x = int(round(voxel_coords[0]))

        pad = self.patch_size // 2

        z_start, z_end = z - pad, z + pad
        y_start, y_end = y - pad, y + pad
        x_start, x_end = x - pad, x + pad

        vol_shape = ct_array.shape
        z1, z2 = max(0, z_start), min(vol_shape[0], z_end)
        y1, y2 = max(0, y_start), min(vol_shape[1], y_end)
        x1, x2 = max(0, x_start), min(vol_shape[2], x_end)

        # Slice the small patch first, THEN free the full volume
        patch = ct_array[z1:z2, y1:y2, x1:x2].copy()
        del ct_array

        # Pad only the small patch to reach patch_size on all sides
        pad_z = (max(0, -z_start), max(0, z_end - vol_shape[0]))
        pad_y = (max(0, -y_start), max(0, y_end - vol_shape[1]))
        pad_x = (max(0, -x_start), max(0, x_end - vol_shape[2]))

        patch = np.pad(patch, (pad_z, pad_y, pad_x), mode='constant', constant_values=-1000)
        # Safety crop: force exact patch_size in case of off-by-one from rounding
        # Center-crop to exact patch_size in case of off-by-one from rounding
        ps = self.patch_size
        cz, cy, cx = patch.shape[0] // 2, patch.shape[1] // 2, patch.shape[2] // 2
        half = ps // 2

        patch = patch[
            cz - half: cz - half + ps,
            cy - half: cy - half + ps,
            cx - half: cx - half + ps,
        ]

        assert patch.shape == (ps, ps, ps), f"Patch shape mismatch: {patch.shape}"
        patch_tensor = torch.tensor(patch, dtype=torch.float32).unsqueeze(0)

        # Get diameter from annotations if positive candidate
        if label == 1:
            ann = self.annotations[self.annotations['seriesuid'] == seriesuid]
            diameter_mm = float(ann.iloc[0]['diameter_mm']) if len(ann) > 0 else 0
        else:
            diameter_mm = 0

        seg_mask = make_seg_mask(diameter_mm, spacing, self.patch_size)
        seg_mask_tensor = torch.tensor(seg_mask, dtype=torch.float32).unsqueeze(0)


        return patch_tensor, seg_mask_tensor, label

# ─────────────────────────────────────────────
# TRAINING CONFIG  — edit these before running
# ─────────────────────────────────────────────

CONFIG = {
    "patch_size"   : 64,
    "batch_size"   : 4,         # keep low for 3D — increase if VRAM allows
    "num_epochs"   : 125,
    "learning_rate": 1e-4,
    "features"     : 32,        # 32 for >=12 GB VRAM, 16 for 8 GB
    "seg_weight"   : 0.7,
    "cls_weight"   : 0.3,
    "num_workers"  : 8,         # set to 4+ on Linux with real data
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
    print(f"Before train or eval")
    model.train() if train else model.eval()

    total_loss, total_dice, total_acc = 0.0, 0.0, 0.0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for patches, seg_masks, cls_labels in loader:
            patches    = patches.to(device)
            seg_masks  = seg_masks.to(device)
            cls_labels = cls_labels.to(device)

            if train:
                optimizer.zero_grad()  # clear old gradients first

            seg_pred, cls_pred = model(patches)
            loss = criterion(seg_pred, seg_masks, cls_pred.squeeze(1), cls_labels.float())


            if train:
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
    #fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["No Nodule", "Nodule"]).plot(ax=ax, colorbar=False)
    #ax.set_title("Confusion Matrix — Test Set")
    #plt.tight_layout()
    #plt.savefig("confusion_matrix.png", dpi=150)
    #plt.show()
    print("  Confusion matrix saved → confusion_matrix.png")

    return mean_dice, auc


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

    data_dir = "data"
    mask_dir = "seg-lungs"
    candidates = "candidates.csv"
    annotations = "annotations.csv"
    patch_size = 64
    print(f"Getting train dataset")
    #10 fold cross validation
    test_loader = None
    train_loader = None


    if args.final:
        train_ds = NodulePatchDataset(
            patch_size, data_dir, mask_dir, candidates, annotations,
            val_fold=None,  # use all subsets
            is_val=False
        )
        print(f"Getting train dataloader")
        train_loader = DataLoader(
            train_ds,
            batch_size=CONFIG["batch_size"],
            shuffle=True,
            num_workers=CONFIG["num_workers"],
            pin_memory=device.type == "cuda",
        )
    else:
        train_ds = NodulePatchDataset(patch_size, data_dir, mask_dir, candidates, annotations,
                                      val_fold=args.val_fold,
                                      is_val=False)

        print(f"Getting test dataset")
        test_ds = NodulePatchDataset(patch_size, data_dir, mask_dir, candidates, annotations,
                                     val_fold=args.val_fold,
                                     is_val=True)

        print(f"Getting train dataloader")
        train_loader = DataLoader(
            train_ds,
            batch_size=CONFIG["batch_size"],
            shuffle=True,
            num_workers=CONFIG["num_workers"],
            pin_memory=device.type == "cuda",
        )
        print(f"Getting test dataloader")
        test_loader = DataLoader(
            test_ds,
            batch_size=CONFIG["batch_size"],
            shuffle=False,
            num_workers=CONFIG["num_workers"],
            pin_memory=device.type == "cuda",
        )


    # ── Model, loss, optimiser ──────────────────────────────────────
    print(f"Setting scheduler, otimizer and criterion")
    model     = UNet3DWithClassifier(features=CONFIG["features"]).to(device)
    criterion = CombinedLoss(CONFIG["seg_weight"], CONFIG["cls_weight"])
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=15
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}\n")

    # Add this before training starts to diagnose


    print(f"Before training")
    # ── Training loop ───────────────────────────────────────────────
    history = {"train_loss": [], "val_loss": [], "train_dice": [],
               "val_dice": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    print(f"Before epoch")
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

    #plt.tight_layout()
    #plt.savefig("training_curves.png", dpi=150)
    #plt.show()
    print("Learning curves saved → training_curves.png")

    # ── Load best weights and run full evaluation ───────────────────
    model.load_state_dict(torch.load(CONFIG["save_path"], map_location=device))
    evaluate(model, test_loader, device)


if __name__ == "__main__":
    main()
