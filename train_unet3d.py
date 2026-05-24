import argparse
import os
import time

import pandas as pd
import torch
from rich import print
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
)
import torch.nn.functional as functional
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

        grid = functional.affine_grid(theta, patch_b.shape, align_corners=False)

        # bilinear for the patch — preserves smooth intensity values
        patch_b = functional.grid_sample(patch_b, grid, mode="bilinear",
                                padding_mode="zeros", align_corners=False)
        # nearest for the mask — keeps values binary, no interpolation blur
        mask_b  = functional.grid_sample(mask_b, grid, mode="nearest",
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
    # ── Focal Tversky (segmentation) ────────────────────────────────────
    # alpha < beta penalises missed nodule voxels harder than false alarms.
    # gamma > 1 focuses gradient on hard patches (paper recommends 1.33).
    # Set gamma=1.0 to reduce exactly to plain Tversky.
    "seg_focal_alpha"     : 0.3,
    "seg_focal_beta"      : 0.7,
    "seg_focal_gamma"     : 1.33,
    # ── Focal Loss (classification) ─────────────────────────────────────
    # alpha=0.25 + gamma=2.0 are the original paper defaults. Nudge alpha
    # up toward 0.5 if positives appear under-fitted (low recall on val).
    "cls_focal_alpha"     : 0.25,
    "cls_focal_gamma"     : 2.0,
    "num_workers"         : 8,
    "save_path"           : "best_model.pth",
    "pos_neg_train_ratio" : 3,         # negatives per positive in each training batch
                                       # 3 → 1 pos : 3 neg per batch (25% positive)
                                       # lower = more positive signal, higher = more variety
                                       # val loader is NEVER rebalanced — keeps true prevalence
    "early_stop_patience" : 20,    # epochs without improvement before stopping
    "early_stop_min_delta": 1e-4,  # minimum improvement to count as "better"
    "early_stop_min_epoch": 20,    # don't stop before this epoch regardless
    "warmup_epochs"       : 10,    # LR ramps from warmup_start_lr → learning_rate over this many epochs
                                   # keeps early weight updates small while model weights are still random,
                                   # which is the main cause of wild val Dice swings in early epochs
    "warmup_start_lr"     : 1e-6,  # starting LR for warmup — near-zero updates on epoch 1
    "val_neg_multiplier"  : 10,    # N_VAL_NEG = n_pos_in_fold * val_neg_multiplier
                                   # ~120 pos in fold 0 → 1,200 negatives (1:10 ratio)
                                   # more realistic than hardcoded 4,000 and scales correctly across folds
}


# ─────────────────────────────────────────────
# METRICS HELPERS
# ─────────────────────────────────────────────

def dice_score(pred, target, threshold=0.5, smooth=1e-6):
    """Compute mean Dice score over a batch (after sigmoid), averaged per sample."""
    # Flatten spatial dims only — keep batch dim (B, D*H*W)
    pred_bin = (pred > threshold).float()
    pred_f = pred_bin.view(pred_bin.shape[0], -1)
    tgt_f = target.view(target.shape[0], -1)

    intersection = (pred_f * tgt_f).sum(dim=1)
    denom = pred_f.sum(dim=1) + tgt_f.sum(dim=1)
    dice_per_sample = (2 * intersection + smooth) / (denom + smooth)

    # Only average over samples that actually contain foreground.
    # Empty-mask patches carry no segmentation signal and should not
    # be scored as "perfect" just because the model also predicted nothing.
    has_fg = tgt_f.sum(dim=1) > 0
    if has_fg.any():
        return dice_per_sample[has_fg].mean().item()
    return float("nan")  # caller skips NaN batches


def binary_accuracy(pred, target, threshold=0.5):
    """Classification accuracy."""
    pred_labels = (pred > threshold).float()
    return (pred_labels == target).float().mean().item()


def format_duration(seconds: float) -> str:
    """Format seconds as 'Hh Mm Ss' or 'Mm Ss' or 'Ss' depending on magnitude."""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


# ─────────────────────────────────────────────
# ONE EPOCH
# ─────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, scaler, train=True):
    model.train() if train else model.eval()

    total_loss, total_dice, total_acc = 0.0, 0.0, 0.0
    dice_values = []  # collect per-batch Dice; some entries may be nan
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
            dice_values.append(dice_score(torch.sigmoid(seg_pred).detach(), seg_masks))
            total_acc += binary_accuracy(torch.sigmoid(cls_pred).squeeze(1).detach(), cls_labels)
            print(f"  Batch [{batch_idx}/{n_batches}]  loss: {loss.item():.4f}", end="\r")  # ← print on same line


    print()  # ← newline after the \r loop finishes
    n = len(loader)

    # np.nanmean ignores nan entries (batches with no foreground patches).
    # If EVERY batch was nan (no positives all epoch — shouldn't happen with
    # your sampler, but guard anyway), fall back to 0.0 instead of nan.
    mean_dice = float(np.nanmean(dice_values)) if len(dice_values) > 0 else 0.0
    if np.isnan(mean_dice):
        mean_dice = 0.0

    return total_loss / n, mean_dice, total_acc / n


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
            all_cls_probs.append(torch.sigmoid(cls_pred).cpu().numpy())
            all_cls_labels.append(cls_labels.cpu().numpy())
            print(f"  Batch [{batch_idx}/{n_batches}]", end="\r")  # ← print on same line

    print()  # ← newline after loop
    print("Flatten")
    cls_probs  = np.concatenate(all_cls_probs).flatten()   # (N,)
    cls_labels = np.concatenate(all_cls_labels).flatten()  # (N,)
    cls_preds  = (cls_probs > 0.5).astype(int)

    mean_dice = np.nanmean(all_dice)
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
    pos_frac = 1.0 / (1.0 + pos_neg_ratio)  # moved up

    # num_samples controls epoch length — keep it equal to dataset size so
    # one "epoch" still means one full pass over the negatives on average
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=n_pos * (1+pos_neg_ratio) * 2,
        replacement=True,
    )

    pos_draws_per_epoch = n_pos * (1 + pos_neg_ratio) * 3 * pos_frac
    actual_draws_per_pos = pos_draws_per_epoch / n_pos  # = (1 + pos_neg_ratio) * 3 * pos_frac

    print(
        f"WeightedRandomSampler : {n_pos:,} pos / {n_neg:,} neg  →  "
        f"~{pos_frac * 100:.0f}% positive per batch  "
        f"(each positive drawn ~{actual_draws_per_pos:.1f}× per epoch)"
    )
    return sampler
# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_fold', type=int, default=0)
    parser.add_argument('--final', action='store_true')
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    index_csv = "~/precomputed/index.csv"
    test_loader = None
    train_loader = None
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
            persistent_workers=True,
        )
    else:
        train_ds = NodulePatchDataset(index_csv, val_fold=args.val_fold, is_val=False, augment=True)
        test_ds = NodulePatchDataset(index_csv, val_fold=args.val_fold, is_val=True)
        train_sampler = make_weighted_sampler(train_ds, CONFIG["pos_neg_train_ratio"])

        # ── Stratified val subset — frozen once before the training loop ──────
        # Negatives are sampled ONCE here with a fixed seed so every epoch sees
        # the identical val set. This makes val Dice a stable, comparable signal.
        # Previously negatives were resampled each epoch, meaning swings in val
        # Dice could simply reflect easier/harder negative draws, not model quality.
        val_index = test_ds.index
        pos_idx = val_index[val_index["label"] == 1].index.tolist()
        N_VAL_NEG = len(pos_idx) * CONFIG["val_neg_multiplier"]
        neg_idx = (
            val_index[val_index["label"] == 0]
            .sample(n=N_VAL_NEG, random_state=42)
            .index.tolist()
        )

        # Interleave pos and neg so positives are distributed across batches
        # instead of clustered in the first ~4. One-time, fixed-seed shuffle —
        # the val set stays frozen and identical across epochs and runs.
        combined_idx = pos_idx + neg_idx
        rng = np.random.default_rng(42)
        rng.shuffle(combined_idx)  # in-place, deterministic

        frozen_val_subset = torch.utils.data.Subset(test_ds, combined_idx)
        test_loader = DataLoader(
            frozen_val_subset,
            batch_size=CONFIG["batch_size"],
            shuffle=False,
            num_workers=CONFIG["num_workers"],
            pin_memory=device.type == "cuda",
            persistent_workers=True,   # workers persist — no rebuild cost each epoch
        )
        print(
            f"Val set frozen: {len(pos_idx):,} pos + {len(neg_idx):,} neg "
            f"= {len(frozen_val_subset):,} total (seed=42, never resampled)"
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=CONFIG["batch_size"],
            sampler=train_sampler,  # sampler handles shuffling — do not set shuffle=True
            num_workers=CONFIG["num_workers"],
            pin_memory=device.type == "cuda",
            persistent_workers=True,
        )


    # ── Model, loss, optimiser ──────────────────────────────────────
    print(f"Setting scheduler, otimizer and criterion")
    model = UNet3DWithClassifier(features=CONFIG["features"]).to(device)
    #model = torch.compile(model, mode='reduce-overhead')
    model = torch.compile(model, mode='default')
    criterion = CombinedLoss(
        seg_weight=CONFIG["seg_weight"],
        cls_weight=CONFIG["cls_weight"],
        seg_alpha=CONFIG["seg_focal_alpha"],
        seg_beta=CONFIG["seg_focal_beta"],
        seg_gamma=CONFIG["seg_focal_gamma"],
        cls_alpha=CONFIG["cls_focal_alpha"],
        cls_gamma=CONFIG["cls_focal_gamma"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    # GradScaler is a no-op when enabled=False (CPU), so safe to always create
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    # ── Two-phase LR schedule ─────────────────────────────────────────────────
    # Phase 1 — Linear warmup (epochs 1 → warmup_epochs):
    #   LR ramps from warmup_start_lr up to learning_rate over warmup_epochs epochs.
    #   This prevents large gradient updates while weights are still near-random,
    #   which was the root cause of the wild val Dice swings in early epochs.
    #   LinearLR uses a multiplicative start_factor: LR = base_lr * start_factor
    #   on epoch 1, then linearly increases to base_lr by the end of warmup.
    #
    # Phase 2 — ReduceLROnPlateau (epochs warmup_epochs+1 → end):
    #   Standard adaptive decay: halves LR when val Dice stops improving for
    #   `patience` epochs. mode="max" because higher Dice = better.
    #
    # SequentialLR switches automatically from phase 1 → phase 2 at the
    # milestone epoch. After the milestone, only plateau_scheduler is active.
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=CONFIG["warmup_start_lr"] / CONFIG["learning_rate"],  # e.g. 1e-6/1e-4 = 0.01
        end_factor=1.0,           # ramp up to full learning_rate
        total_iters=CONFIG["warmup_epochs"],
    )
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6,  # mode=max: higher vl_dice = better
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}\n")

    # ── Training loop ───────────────────────────────────────────────
    history = {"train_loss": [], "val_loss": [], "train_dice": [],
               "val_dice": [], "train_acc": [], "val_acc": []}

    # Single source of truth: vl_dice drives both checkpointing and early stopping.
    # best_val_loss is intentionally removed — combined loss is dominated by BCE
    # which barely moves, making it an unreliable checkpoint signal.
    best_val_dice = 0.0
    best_epoch = 0          # epoch number at which best_val_dice was hit (for "epochs since best")
    patience_counter = 0
    start_epoch = 1
    early_stopped = False   # tracks whether we exited the loop via early stopping (for final summary)

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        warmup_scheduler.load_state_dict(ckpt["scheduler_warmup"])
        plateau_scheduler.load_state_dict(ckpt["scheduler_plateau"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_dice = ckpt["best_val_dice"]
        best_epoch = ckpt["epoch"]   # assume the saved epoch IS the best epoch (it always is — we only save on improvement)
        print(f"  → Resumed at epoch {start_epoch}, best_val_dice: {best_val_dice:.4f}")

    # ── Wall-clock tracking ──────────────────────────────────────────────────
    # training_start: anchors total elapsed and ETA calculations.
    # epoch_times:    rolling window of recent epoch durations; ETA uses the
    #                 mean of the last few rather than the very last one so
    #                 a single slow epoch (disk hiccup, GC pause) doesn't
    #                 throw the estimate.
    training_start = time.perf_counter()
    epoch_times: list[float] = []
    ETA_WINDOW = 5  # average over the last N epochs

    for epoch in range(start_epoch, CONFIG["num_epochs"] + 1):
        # ── Reset CUDA peak memory so per-epoch reading reflects THIS epoch only ──
        # Without the reset, max_memory_allocated() grows monotonically and tells
        # us nothing useful after epoch 1. Resetting at the top makes the post-epoch
        # print a genuine "how much VRAM did this epoch use" number, which is what
        # catches memory leaks (peak should be flat — if it climbs, you have one).
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        # ── Epoch header — easy to grep for in long logs ──────────────────────
        print(f"\n[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━ "
              f"Epoch {epoch} / {CONFIG['num_epochs']} "
              f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")

        epoch_start = time.perf_counter()

        # ── Train pass (timed) ────────────────────────────────────────────────
        t0 = time.perf_counter()
        tr_loss, tr_dice, tr_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, scaler, train=True
        )
        train_duration = time.perf_counter() - t0

        # ── Val pass (timed) ──────────────────────────────────────────────────
        if test_loader is not None:
            t0 = time.perf_counter()
            vl_loss, vl_dice, vl_acc = run_epoch(
                model, test_loader, criterion, optimizer, device, scaler, train=False
            )
            val_duration = time.perf_counter() - t0
        else:
            vl_loss, vl_dice, vl_acc = 0.0, 0.0, 0.0
            val_duration = 0.0

        epoch_duration = time.perf_counter() - epoch_start
        epoch_times.append(epoch_duration)

        # ── Scheduler step ────────────────────────────────────────────────────
        # During warmup, LinearLR steps every epoch with no metric needed.
        # After warmup, ReduceLROnPlateau takes over and requires vl_dice.
        if epoch <= CONFIG["warmup_epochs"]:
            warmup_scheduler.step()
            lr_phase = f"warmup {epoch}/{CONFIG['warmup_epochs']}"
        else:
            if test_loader is not None:
                plateau_scheduler.step(vl_dice)
            lr_phase = "plateau"
        current_lr = optimizer.param_groups[0]["lr"]

        # ── Persist history ───────────────────────────────────────────────────
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_dice"].append(tr_dice)
        history["val_dice"].append(vl_dice)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        # ── ETA from rolling mean of recent epoch durations ───────────────────
        recent = epoch_times[-ETA_WINDOW:]
        mean_epoch_time = sum(recent) / len(recent)
        epochs_remaining = CONFIG["num_epochs"] - epoch
        eta_seconds = mean_epoch_time * epochs_remaining
        total_elapsed = time.perf_counter() - training_start

        # ── Per-epoch summary block ───────────────────────────────────────────
        # One coherent block per epoch — easier to scan than the old
        # interleaved scheduler/metric prints. Order: timing → metrics →
        # checkpoint status → patience/best tracking.
        print(
            f"  ⏱  {format_duration(epoch_duration)}  "
            f"(train {format_duration(train_duration)}  /  val {format_duration(val_duration)})"
            f"   |   LR: {current_lr:.2e}  [{lr_phase}]"
        )
        print(
            f"  Train  →  loss {tr_loss:.4f}   dice {tr_dice:.4f}   acc {tr_acc:.3f}"
        )
        print(
            f"  Val    →  loss {vl_loss:.4f}   dice {vl_dice:.4f}   acc {vl_acc:.3f}"
        )

        # ── GPU memory (per-epoch peak, reset at top of loop) ─────────────────
        if device.type == "cuda":
            peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            reserved_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)
            # allocated = tensors actually in use; reserved = held by the caching
            # allocator (incl. free blocks). Reserved is what shows up in nvidia-smi.
            print(f"  GPU    →  peak {peak_gb:.2f} GB allocated   /   {reserved_gb:.2f} GB reserved")

        # ── Checkpoint: save whenever val Dice improves (no epoch gate) ──────
        # Gating this on early_stop_min_epoch would risk missing a best checkpoint
        # that occurs in early epochs (e.g. epoch 15 hit 0.969 in your first run).
        improved = (
            test_loader is not None
            and vl_dice > best_val_dice + CONFIG["early_stop_min_delta"]
        )
        if improved:
            prev_best = best_val_dice
            prev_best_epoch = best_epoch
            best_val_dice = vl_dice
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler_warmup": warmup_scheduler.state_dict(),
                "scheduler_plateau": plateau_scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_val_dice": best_val_dice,
            }, CONFIG["save_path"])
            if prev_best_epoch == 0:
                # first improvement — no previous best to compare against
                print(f"  [green]✓ Best model saved   (val dice {best_val_dice:.4f})[/green]")
            else:
                delta = best_val_dice - prev_best
                print(
                    f"  [green]✓ Best model saved   "
                    f"(val dice {best_val_dice:.4f}, prev {prev_best:.4f} at epoch {prev_best_epoch}, "
                    f"Δ +{delta:.4f})[/green]"
                )
        else:
            if test_loader is not None:
                if best_epoch == 0:
                    # haven't hit any best yet — nothing to report against
                    print(f"  · No improvement   (val dice {vl_dice:.4f})")
                else:
                    print(
                        f"  · No improvement   "
                        f"(val dice {vl_dice:.4f}, best {best_val_dice:.4f} at epoch {best_epoch})"
                    )
            # Patience only counts after min epoch — let model warm up first
            if epoch >= CONFIG["early_stop_min_epoch"] and test_loader is not None:
                patience_counter += 1

        # ── Patience and timing footer ────────────────────────────────────────
        if test_loader is not None:
            epochs_since_best = (epoch - best_epoch) if best_epoch > 0 else epoch
            patience_str = (
                f"Patience: {patience_counter}/{CONFIG['early_stop_patience']}"
                if epoch >= CONFIG["early_stop_min_epoch"]
                else f"Patience: -- (counting starts at epoch {CONFIG['early_stop_min_epoch']})"
            )
            print(
                f"  {patience_str}   |   "
                f"epochs since best: {epochs_since_best}   |   "
                f"total elapsed: {format_duration(total_elapsed)}   "
                f"|   ETA: {format_duration(eta_seconds)}"
            )

        # ── Early stopping ────────────────────────────────────────────────────
        if epoch >= CONFIG["early_stop_min_epoch"] and test_loader is not None:
            if patience_counter >= CONFIG["early_stop_patience"]:
                print(f"\n[yellow]Early stopping triggered at epoch {epoch}[/yellow]")
                early_stopped = True
                break

    # ── Training-complete summary block ───────────────────────────────────────
    total_training_time = time.perf_counter() - training_start
    epochs_run = len(history["train_loss"])
    completion_tag = "early stop" if early_stopped else "natural completion"
    print(f"\n[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━ "
          f"Training complete "
          f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
    print(f"  Total time: {format_duration(total_training_time)}   "
          f"|   Epochs run: {epochs_run} / {CONFIG['num_epochs']} ({completion_tag})")
    if best_epoch > 0:
        print(f"  Best val dice: {best_val_dice:.4f} at epoch {best_epoch}")
    elif test_loader is None:
        print(f"  --final mode: no validation, no best checkpoint")
    else:
        print(f"  No improvement observed during training")

    print("\nCreating plots")
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

    # ── Load best weights and run full evaluation ─────────────────────────
    # Loads the BEST checkpoint (highest val Dice seen during training), not
    # whatever was in memory when the loop exited — which may be several epochs
    # past the best due to early-stopping patience.
    #
    # The checkpoint is now a dict (added when scheduler/scaler state was
    # introduced), so we index ckpt["model"] instead of passing the whole
    # thing to load_state_dict — the old `load_state_dict(torch.load(...))`
    # call would fail with "Unexpected key(s): epoch, optimizer, ...".
    #
    # `model` is still the torch.compile-wrapped object here, and the
    # checkpoint was saved from the same compiled model, so the
    # `_orig_mod.` prefixed keys line up cleanly. weights_only=True is safe:
    # only tensors and primitives are in the file.
    #
    # In --final mode there is no test_loader and no best checkpoint (saving
    # is gated on test_loader is not None), so this whole block is skipped.
    if test_loader is not None:
        print("\nLoading best checkpoint for final evaluation...")
        ckpt = torch.load(CONFIG["save_path"], map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        print(f"  → Loaded epoch {ckpt['epoch']}, best_val_dice: {ckpt['best_val_dice']:.4f}")
        evaluate(model, test_loader, device)


if __name__ == "__main__":
    main()