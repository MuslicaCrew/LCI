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
torch.set_float32_matmul_precision('high')


class Augment3D:

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
        patch: torch.Tensor,
        mask:  torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        for dim in [1, 2, 3]:
            if torch.rand(1).item() < self.flip_prob:
                patch = torch.flip(patch, dims=[dim])
                mask  = torch.flip(mask,  dims=[dim])

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

        R     = Rz @ Ry @ Rx
        theta = torch.cat([R, torch.zeros(3, 1)], dim=1).unsqueeze(0)

        patch_b = patch.unsqueeze(0)
        mask_b  = mask.unsqueeze(0)

        grid = functional.affine_grid(theta, patch_b.shape, align_corners=False)
        patch_b = functional.grid_sample(patch_b, grid, mode="bilinear",
                                padding_mode="zeros", align_corners=False)
        mask_b  = functional.grid_sample(mask_b, grid, mode="nearest",
                                padding_mode="zeros", align_corners=False)

        patch = patch_b.squeeze(0)
        mask  = mask_b.squeeze(0)

        patch = patch + torch.randn_like(patch) * self.noise_std
        patch = patch + (torch.rand(1).item() * 2 - 1) * self.brightness_max
        patch = patch.clamp(0.0, 1.0)

        return patch, mask


class NodulePatchDataset(Dataset):
    def __init__(self,
        index_csv: str,
        val_fold:  int | None = 0,
        is_val:    bool = False,
        augment:   bool = False,):
        index = pd.read_csv(index_csv)

        if val_fold is None:
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

        patch = np.load(row["patch_path"])["patch"]
        seg_mask = np.load(row["seg_mask_path"])["seg_mask"]

        patch_tensor = torch.from_numpy(patch.astype(np.float32)).unsqueeze(0)
        seg_mask_tensor = torch.from_numpy(seg_mask).unsqueeze(0)

        if self.augmenter is not None:
            patch_tensor, seg_mask_tensor = self.augmenter(patch_tensor, seg_mask_tensor)

        return patch_tensor, seg_mask_tensor, cls_label

CONFIG = {
    "patch_size"          : 64,
    "batch_size"          : 32,
    "num_epochs"          : 150,
    "learning_rate"       : 1e-4,
    "features"            : 32,
    "seg_weight"          : 0.5,
    "cls_weight"          : 0.5,
    "seg_focal_alpha"     : 0.3,
    "seg_focal_beta"      : 0.7,
    "seg_focal_gamma"     : 1.33,
    "seg_empty_weight" : 0.1,
    "cls_focal_alpha"     : 0.5,
    "cls_focal_gamma"     : 2.0,
    "num_workers"         : 8,
    "save_path"           : "best_model.pth",
    "pos_neg_train_ratio" : 3,
    "early_stop_patience" : 20,
    "early_stop_min_delta": 1e-4,
    "early_stop_min_epoch": 20,
    "warmup_epochs"       : 10,
    "warmup_start_lr"     : 1e-6,
    "val_neg_multiplier"  : 10,

}

def dice_score(pred, target, threshold=0.5, smooth=1e-6):
    pred_bin = (pred > threshold).float()
    pred_f = pred_bin.view(pred_bin.shape[0], -1)
    tgt_f = target.view(target.shape[0], -1)

    intersection = (pred_f * tgt_f).sum(dim=1)
    denom = pred_f.sum(dim=1) + tgt_f.sum(dim=1)
    dice_per_sample = (2 * intersection + smooth) / (denom + smooth)

    has_fg = tgt_f.sum(dim=1) > 0
    if has_fg.any():
        return dice_per_sample[has_fg].mean().item()
    return float("nan")


def binary_accuracy(pred, target, threshold=0.5):
    pred_labels = (pred > threshold).float()
    return (pred_labels == target).float().mean().item()


def format_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"



def run_epoch(model, loader, criterion, optimizer, device, scaler, train=True):
    model.train() if train else model.eval()

    total_loss, total_dice, total_acc = 0.0, 0.0, 0.0
    dice_values = []
    is_cuda = device.type == "cuda"
    n_batches = len(loader)

    all_cls_probs  = []
    all_cls_labels = []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch_idx, (patches, seg_masks, cls_labels) in enumerate(loader, 1):
            patches = patches.to(device)
            seg_masks = seg_masks.to(device)
            cls_labels = cls_labels.to(device)

            if train:
                optimizer.zero_grad(set_to_none=True)


            with torch.autocast(device_type=device.type, enabled=is_cuda):
                seg_pred, cls_pred = model(patches)
                loss = criterion(seg_pred, seg_masks, cls_pred.squeeze(1), cls_labels.float())

            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item()
            dice_values.append(dice_score(torch.sigmoid(seg_pred).detach(), seg_masks))

            cls_probs_batch = torch.sigmoid(cls_pred).squeeze(1).detach().float()
            total_acc += binary_accuracy(cls_probs_batch, cls_labels)

            all_cls_probs.append(cls_probs_batch)
            all_cls_labels.append(cls_labels.detach())

            print(f"  Batch [{batch_idx}/{n_batches}]  loss: {loss.item():.4f}", end="\r")

    print()
    n = len(loader)

    mean_dice = float(np.nanmean(dice_values)) if len(dice_values) > 0 else 0.0
    if np.isnan(mean_dice):
        mean_dice = 0.0

        cls_probs_all  = torch.cat(all_cls_probs)
    cls_labels_all = torch.cat(all_cls_labels)
    n_pos = int(cls_labels_all.sum().item())
    n_tot = cls_labels_all.numel()


    if 0 < n_pos < n_tot:
        n_neg = n_tot - n_pos

        order = torch.argsort(cls_probs_all)
        ranks = torch.empty_like(order, dtype=torch.float32)
        ranks[order] = torch.arange(1, n_tot + 1, device=cls_probs_all.device,
                                    dtype=torch.float32)
        sum_pos_ranks = ranks[cls_labels_all == 1].sum()
        cls_auc = float(((sum_pos_ranks - n_pos * (n_pos + 1) / 2)
                         / (n_pos * n_neg)).item())
    else:
        cls_auc = float("nan")

    if n_pos > 0:
        pos_probs = cls_probs_all[cls_labels_all == 1]
        cls_recall_01 = float((pos_probs > 0.1).float().mean().item())
    else:
        cls_recall_01 = float("nan")

    cls_prob_min = float(cls_probs_all.min().item())
    cls_prob_max = float(cls_probs_all.max().item())
    # ──────────────────────────────────────────────────────────────────

    return (total_loss / n, mean_dice, total_acc / n,
            cls_auc, cls_recall_01, cls_prob_min, cls_prob_max)



def evaluate(model, loader, device):
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
            print(f"  Batch [{batch_idx}/{n_batches}]", end="\r")

    print()
    print("Flatten")
    cls_probs  = np.concatenate(all_cls_probs).flatten()
    cls_labels = np.concatenate(all_cls_labels).flatten()
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

    cm = confusion_matrix(cls_labels, cls_preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["No Nodule", "Nodule"]).plot(ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix — Test Set")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)

    print("  Confusion matrix saved → confusion_matrix.png")

    return mean_dice, auc

def make_weighted_sampler(dataset: NodulePatchDataset, pos_neg_ratio: int) -> WeightedRandomSampler:

    labels = dataset.index["label"].values
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())


    weight_pos = 1.0
    weight_neg = 1.0 / pos_neg_ratio

    sample_weights = np.where(labels == 1, weight_pos, weight_neg).tolist()
    pos_frac = 1.0 / (1.0 + pos_neg_ratio)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=n_pos * (1+pos_neg_ratio) * 2,
        replacement=True,
    )

    pos_draws_per_epoch = n_pos * (1 + pos_neg_ratio) * 2 * pos_frac
    actual_draws_per_pos = pos_draws_per_epoch / n_pos

    print(
        f"WeightedRandomSampler : {n_pos:,} pos / {n_neg:,} neg  →  "
        f"~{pos_frac * 100:.0f}% positive per batch  "
        f"(each positive drawn ~{actual_draws_per_pos:.1f}× per epoch)"
    )
    return sampler

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
            sampler=train_sampler,
            num_workers=CONFIG["num_workers"],
            pin_memory=device.type == "cuda",
            persistent_workers=True,
        )
    else:
        train_ds = NodulePatchDataset(index_csv, val_fold=args.val_fold, is_val=False, augment=True)
        test_ds = NodulePatchDataset(index_csv, val_fold=args.val_fold, is_val=True)
        train_sampler = make_weighted_sampler(train_ds, CONFIG["pos_neg_train_ratio"])

        val_index = test_ds.index
        pos_idx = val_index[val_index["label"] == 1].index.tolist()
        N_VAL_NEG = len(pos_idx) * CONFIG["val_neg_multiplier"]
        neg_idx = (
            val_index[val_index["label"] == 0]
            .sample(n=N_VAL_NEG, random_state=42)
            .index.tolist()
        )

        combined_idx = pos_idx + neg_idx
        rng = np.random.default_rng(42)
        rng.shuffle(combined_idx)

        frozen_val_subset = torch.utils.data.Subset(test_ds, combined_idx)
        test_loader = DataLoader(
            frozen_val_subset,
            batch_size=CONFIG["batch_size"],
            shuffle=False,
            num_workers=CONFIG["num_workers"],
            pin_memory=device.type == "cuda",
            persistent_workers=True,
        )
        print(
            f"Val set frozen: {len(pos_idx):,} pos + {len(neg_idx):,} neg "
            f"= {len(frozen_val_subset):,} total (seed=42, never resampled)"
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=CONFIG["batch_size"],
            sampler=train_sampler,
            num_workers=CONFIG["num_workers"],
            pin_memory=device.type == "cuda",
            persistent_workers=True,
        )

    print(f"Setting scheduler, otimizer and criterion")
    model = UNet3DWithClassifier(features=CONFIG["features"]).to(device)

    model = torch.compile(model, mode='default')
    criterion = CombinedLoss(
        seg_weight=CONFIG["seg_weight"],
        cls_weight=CONFIG["cls_weight"],
        seg_alpha=CONFIG["seg_focal_alpha"],
        seg_beta=CONFIG["seg_focal_beta"],
        seg_gamma=CONFIG["seg_focal_gamma"],
        seg_empty_weight=CONFIG["seg_empty_weight"],
        cls_alpha=CONFIG["cls_focal_alpha"],
        cls_gamma=CONFIG["cls_focal_gamma"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")


    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=CONFIG["warmup_start_lr"] / CONFIG["learning_rate"],
        end_factor=1.0,
        total_iters=CONFIG["warmup_epochs"],
    )
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6,
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}\n")

    history = {"train_loss": [], "val_loss": [], "train_dice": [],
               "val_dice": [], "train_acc": [], "val_acc": []}

    best_val_dice = 0.0
    best_epoch = 0
    patience_counter = 0
    start_epoch = 1
    early_stopped = False

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
        best_epoch = ckpt["epoch"]
        print(f"  → Resumed at epoch {start_epoch}, best_val_dice: {best_val_dice:.4f}")


    training_start = time.perf_counter()
    epoch_times: list[float] = []
    ETA_WINDOW = 5

    for epoch in range(start_epoch, CONFIG["num_epochs"] + 1):

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        print(f"\n[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━ "
              f"Epoch {epoch} / {CONFIG['num_epochs']} "
              f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")

        epoch_start = time.perf_counter()


        t0 = time.perf_counter()
        tr_loss, tr_dice, tr_acc, train_auc, train_rec01, train_pmin, train_pmax = run_epoch(
            model, train_loader, criterion, optimizer, device, scaler, train=True
        )
        train_duration = time.perf_counter() - t0


        if test_loader is not None:
            t0 = time.perf_counter()
            vl_loss, vl_dice, vl_acc, val_auc, val_rec01, val_pmin, val_pmax = run_epoch(
                model, test_loader, criterion, optimizer, device, scaler, train=False
            )
            val_duration = time.perf_counter() - t0
        else:
            vl_loss, vl_dice, vl_acc = 0.0, 0.0, 0.0
            val_duration = 0.0

        epoch_duration = time.perf_counter() - epoch_start
        epoch_times.append(epoch_duration)


        if epoch <= CONFIG["warmup_epochs"]:
            warmup_scheduler.step()
            lr_phase = f"warmup {epoch}/{CONFIG['warmup_epochs']}"
        else:
            if test_loader is not None:
                plateau_scheduler.step(vl_dice)
            lr_phase = "plateau"
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_dice"].append(tr_dice)
        history["val_dice"].append(vl_dice)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        recent = epoch_times[-ETA_WINDOW:]
        mean_epoch_time = sum(recent) / len(recent)
        epochs_remaining = CONFIG["num_epochs"] - epoch
        eta_seconds = mean_epoch_time * epochs_remaining
        total_elapsed = time.perf_counter() - training_start

        print(
            f"  ⏱  {format_duration(epoch_duration)}  "
            f"(train {format_duration(train_duration)}  /  val {format_duration(val_duration)})"
            f"   |   LR: {current_lr:.2e}  [{lr_phase}]"
        )
        print(f"  Train  →  loss {tr_loss:.4f}   dice {tr_dice:.4f}   "
              f"acc {tr_acc:.3f}   AUC {train_auc:.3f}   "
              f"rec@0.1 {train_rec01:.3f}   probs [{train_pmin:.3f}, {train_pmax:.3f}]")

        print(f"  Val    →  loss {vl_loss:.4f}   dice {vl_dice:.4f}   "
              f"acc {vl_acc:.3f}   AUC {val_auc:.3f}   "
              f"rec@0.1 {val_rec01:.3f}   probs [{val_pmin:.3f}, {val_pmax:.3f}]")

        if device.type == "cuda":
            peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            reserved_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)
            print(f"  GPU    →  peak {peak_gb:.2f} GB allocated   /   {reserved_gb:.2f} GB reserved")

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
                    print(f"  · No improvement   (val dice {vl_dice:.4f})")
                else:
                    print(
                        f"  · No improvement   "
                        f"(val dice {vl_dice:.4f}, best {best_val_dice:.4f} at epoch {best_epoch})"
                    )
            if epoch >= CONFIG["early_stop_min_epoch"] and test_loader is not None:
                patience_counter += 1

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

        if epoch >= CONFIG["early_stop_min_epoch"] and test_loader is not None:
            if patience_counter >= CONFIG["early_stop_patience"]:
                print(f"\n[yellow]Early stopping triggered at epoch {epoch}[/yellow]")
                early_stopped = True
                break

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

    if test_loader is not None:
        print("\nLoading best checkpoint for final evaluation...")
        ckpt = torch.load(CONFIG["save_path"], map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        print(f"  → Loaded epoch {ckpt['epoch']}, best_val_dice: {ckpt['best_val_dice']:.4f}")
        evaluate(model, test_loader, device)


if __name__ == "__main__":
    main()