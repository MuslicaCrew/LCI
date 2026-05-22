import torch
import torch.nn as nn
import torch.nn.functional as functional


# ─────────────────────────────────────────────
# BUILDING BLOCKS (3D)
# ─────────────────────────────────────────────

class ConvBlock3D(nn.Module):
    """
    One encoder/decoder block:
    Conv3d → BN → ReLU → Conv3d → BN → ReLU
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class DownBlock3D(nn.Module):
    """
    Encoder step: ConvBlock3D followed by MaxPool3d
    Halves spatial dimensions in all 3 axes (Z, Y, X)
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock3D(in_channels, out_channels)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)  # halves Z, Y, X

    def forward(self, x):
        skip = self.conv(x)   # save for skip connection
        x = self.pool(skip)
        return x, skip        # return compressed x and the skip

class UpBlock3D(nn.Module):
    """
    Decoder step: ConvTranspose3d → concat skip → ConvBlock3D
    Doubles spatial dimensions in all 3 axes (Z, Y, X)
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock3D(out_channels * 2, out_channels)  # *2 because of skip concat

    def forward(self, x, skip):
        x = self.upsample(x)                    # double spatial size in all 3D
        x = torch.cat([x, skip], dim=1)         # concatenate skip connection
        return self.conv(x)

# ─────────────────────────────────────────────
# FULL 3D U-NET WITH BRANCHING CLASSIFIER
# ─────────────────────────────────────────────

class UNet3DWithClassifier(nn.Module):
    """
    3D U-Net designed for lung nodule patch analysis.

    Input:  (batch, 1, 64, 64, 64)  ← single-channel 3D patch from CT scan
    Output:
        seg_map          : (batch, 1, 64, 64, 64)  ← per-voxel nodule probability
        cancer_probability: (batch, 1)              ← is there a nodule? (0.0 - 1.0)

    Args:
        in_channels  : 1 for grayscale CT patches
        out_channels : 1 for binary segmentation (nodule vs background)
        features     : filters in first block — doubles each encoder level
                       Use 32 for powerful GPUs (>=12GB VRAM)
                       Use 16 for smaller GPUs (8GB VRAM)

    Memory estimates (batch_size=2, patch 64x64x64):
        features=32 → ~10-12GB VRAM
        features=16 → ~3-4GB VRAM

    Architecture:
        Encoder:     4 DownBlocks (halving spatial size each time)
        Bottleneck:  ConvBlock3D at smallest spatial resolution
        Classifier:  Branches off bottleneck → GlobalAvgPool → Linear → Sigmoid
        Decoder:     4 UpBlocks with skip connections (restoring spatial size)
        Seg head:    1x1x1 Conv → Sigmoid (per-voxel probability)
    """
    def __init__(self, in_channels=1, out_channels=1, features=16):
        super().__init__()

        # ── Encoder (contracting path) ──────────────────────────────
        # Each DownBlock halves spatial dims: 64→32→16→8→4
        self.enc1 = DownBlock3D(in_channels, features)          # 64³ → 32³, 16 filters
        self.enc2 = DownBlock3D(features, features * 2)         # 32³ → 16³, 32 filters
        self.enc3 = DownBlock3D(features * 2, features * 4)     # 16³ → 8³,  64 filters
        self.enc4 = DownBlock3D(features * 4, features * 8)     # 8³  → 4³,  128 filters

        # ── Bottleneck ───────────────────────────────────────────────
        # Deepest representation: 4³ spatial, 256 filters
        self.bottleneck = ConvBlock3D(features * 8, features * 16)  # 256 filters

        # ── Classification head (branches off bottleneck) ────────────
        # GlobalAvgPool collapses 4³ spatial → single vector
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),        # (batch, 256, 4, 4, 4) → (batch, 256, 1, 1, 1)
            nn.Flatten(),                   # → (batch, 256)
            nn.Linear(features * 16, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, 1),
            #nn.Sigmoid()                    # 0.0 = no nodule, 1.0 = nodule
        )

        # ── Decoder (expanding path) ─────────────────────────────────
        # Each UpBlock doubles spatial dims: 4→8→16→32→64
        self.dec4 = UpBlock3D(features * 16, features * 8)     # 4³  → 8³
        self.dec3 = UpBlock3D(features * 8, features * 4)      # 8³  → 16³
        self.dec2 = UpBlock3D(features * 4, features * 2)      # 16³ → 32³
        self.dec1 = UpBlock3D(features * 2, features)          # 32³ → 64³

        # ── Segmentation output ───────────────────────────────────────
        # 1x1x1 conv to get per-voxel prediction
        self.seg_head = nn.Conv3d(features, out_channels, kernel_size=1)
        # NOTE: no sigmoid here — model returns raw logits for seg_head.
        # DiceLoss applies sigmoid internally. dice_score metric applies it too.
        # This avoids double-sigmoid which distorts gradients.

    def forward(self, x):
        # ── Encoder — compress and save skip connections
        x, skip1 = self.enc1(x)   # skip1: (batch, 16,  64, 64, 64)
        x, skip2 = self.enc2(x)   # skip2: (batch, 32,  32, 32, 32)
        x, skip3 = self.enc3(x)   # skip3: (batch, 64,  16, 16, 16)
        x, skip4 = self.enc4(x)   # skip4: (batch, 128,  8,  8,  8)

        # ── Bottleneck
        x = self.bottleneck(x)    # x:     (batch, 256,  4,  4,  4)

        # ── Classification branch (off bottleneck, before decoding)
        cancer_probability = self.classifier(x)   # shape: (batch, 1)

        # ── Decoder — expand and merge skip connections
        x = self.dec4(x, skip4)   # (batch, 128,  8,  8,  8)
        x = self.dec3(x, skip3)   # (batch,  64, 16, 16, 16)
        x = self.dec2(x, skip2)   # (batch,  32, 32, 32, 32)
        x = self.dec1(x, skip1)   # (batch,  16, 64, 64, 64)

        # ── Segmentation output
        seg_map = self.seg_head(x)  # (batch, 1, 64, 64, 64) — raw logits

        return seg_map, cancer_probability


# ─────────────────────────────────────────────
# LOSS FUNCTIONS
# ─────────────────────────────────────────────

class DiceLoss(nn.Module):
    """
    Dice loss for 3D segmentation.
    Handles class imbalance well — critical for nodules
    which are tiny compared to the full patch volume.

    Expects RAW LOGITS from the model (no sigmoid applied beforehand).
    Applies sigmoid internally so the loss sees clean probabilities
    without any double-sigmoid distortion on gradients.
    """
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        pred = pred.view(pred.shape[0], -1)
        target = target.view(target.shape[0], -1)

        intersection = (pred * target).sum(dim=1)
        denominator = pred.sum(dim=1) + target.sum(dim=1)
        dice_per_sample = (2 * intersection + self.smooth) / (denominator + self.smooth)
        loss_per_sample = 1 - dice_per_sample

        # only supervise the seg head where a mask actually exists
        has_mask = target.sum(dim=1) > 0  # (B,) bool
        if has_mask.any():
            return loss_per_sample[has_mask].mean()
        else:
            return torch.tensor(0.0, device=pred.device, requires_grad=False)


class TverskyLoss(nn.Module):
    """
      Tversky loss for 3D segmentation — a generalisation of Dice.

      alpha weights false positives, beta weights false negatives.
      With alpha < beta, missing a nodule voxel costs more than a
      false alarm, which is the right tradeoff for nodule detection.
      alpha=beta=0.5 is exactly Dice.

      Expects RAW LOGITS (sigmoid applied internally, once).
      """
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha  # FP weight
        self.beta = beta    # FN weight — keep higher to penalise missed nodules
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        pred   = pred.view(pred.shape[0], -1)
        target = target.view(target.shape[0], -1)

        tp = (pred * target).sum(dim=1)
        fp = (pred * (1 - target)).sum(dim=1)
        fn = ((1 - pred) * target).sum(dim=1)

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        loss_per_sample = 1 - tversky

        has_mask = target.sum(dim=1) > 0
        if has_mask.any():
            return loss_per_sample[has_mask].mean()
        else:
            return torch.tensor(0.0, device=pred.device, requires_grad=False)


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky loss for 3D segmentation (Abraham & Khan, 2019).

    Builds on Tversky by raising the per-sample (1 - Tversky) to a power gamma:
        FTL = (1 - Tversky) ** gamma

    Intuition:
      - gamma > 1  → focus on HARD samples (low Tversky). Easy samples (Tversky
                     near 1) contribute almost nothing to the gradient, so the
                     optimiser concentrates on the few patches the network is
                     still mispredicting. Recommended for LUNA16: the tiny
                     nodule voxels are exactly the hard cases that vanilla
                     Tversky undertrains on.
      - gamma == 1 → reduces exactly to Tversky loss.
      - gamma < 1  → focuses on EASY samples (rarely useful).

    alpha weights false positives, beta weights false negatives.
    With alpha < beta, missing a nodule voxel costs more than a false alarm,
    which is the right tradeoff for nodule detection. The canonical
    recommendation from the paper is alpha=0.3, beta=0.7, gamma=0.75 OR 1.33;
    we default to gamma=1.33 (the "harder-focus" variant) since LUNA16 is
    dominated by easy empty/near-empty patches.

    Expects RAW LOGITS (sigmoid applied internally, once). Computes the focal
    exponentiation in float32 inside autocast to avoid fp16 underflow of
    (1 - Tversky) when it's close to zero.
    """
    def __init__(
        self,
        alpha: float = 0.3,
        beta:  float = 0.7,
        gamma: float = 1.33,
        smooth: float = 1e-6,
    ):
        super().__init__()
        self.alpha  = alpha   # FP weight
        self.beta   = beta    # FN weight
        self.gamma  = gamma   # focal exponent
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        pred   = pred.view(pred.shape[0], -1)
        target = target.view(target.shape[0], -1)

        tp = (pred * target).sum(dim=1)
        fp = (pred * (1 - target)).sum(dim=1)
        fn = ((1 - pred) * target).sum(dim=1)

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

        # Force fp32 for the pow — small (1-tversky) values can underflow in fp16.
        # Clamp away from 0 so log/grad of pow remain finite even at "perfect" samples.
        one_minus_tv = (1.0 - tversky).float().clamp(min=self.smooth)
        loss_per_sample = one_minus_tv.pow(self.gamma)

        has_mask = target.sum(dim=1) > 0
        if has_mask.any():
            return loss_per_sample[has_mask].mean()
        else:
            return torch.tensor(0.0, device=pred.device, requires_grad=False)


class FocalLoss(nn.Module):
    """
    Binary focal loss for classification (Lin et al., 2017, "Focal Loss for
    Dense Object Detection").

    Formula (on logits z, target y in {0,1}):
        p   = sigmoid(z)
        p_t = p   if y=1   else  1-p
        alpha_t = alpha if y=1 else 1-alpha
        FL = -alpha_t * (1 - p_t)**gamma * log(p_t)

    Intuition:
      - The (1 - p_t)**gamma factor down-weights easy examples where p_t is
        close to 1. Hard examples (p_t small) keep their full BCE gradient.
        With gamma=2 (paper default), an easy example with p_t=0.9 contributes
        ~100x less than a hard example with p_t=0.5.
      - alpha balances the two classes. With alpha=0.25, positives get
        weight 0.25 and negatives 0.75 — counter-intuitive, but the paper
        showed this is what works best in tandem with gamma, because gamma
        already handles most of the imbalance via easy-negative suppression.
        For LUNA16 (positives still rare even after WeightedRandomSampler),
        alpha=0.25 is a safe default; nudge up to ~0.5 if positives appear
        under-fitted.

    Numerically stable: computed entirely from logits via the BCE-with-logits
    identity, never explicitly calling sigmoid. AMP-safe.

    Expects:
        logits: (N,) raw scores from the classifier head
        target: (N,) float in {0.0, 1.0}
    """
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        assert reduction in ("mean", "sum", "none")
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Per-element BCE without reduction — this is -log(p_t) per sample.
        ce = functional.binary_cross_entropy_with_logits(
            logits, target, reduction="none"
        )

        # p_t = exp(-ce). When target=1, ce = -log(p) → exp(-ce) = p.
        # When target=0, ce = -log(1-p) → exp(-ce) = 1-p.  Either way it's p_t.
        # Cast to fp32 because exp() and pow() can underflow under autocast(fp16).
        p_t = torch.exp(-ce.float())

        # alpha_t: alpha for positives, (1-alpha) for negatives.
        alpha_t = torch.where(target > 0.5, self.alpha, 1.0 - self.alpha)

        focal_term = (1.0 - p_t).pow(self.gamma)
        loss = alpha_t * focal_term * ce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class CombinedLoss(nn.Module):
    """
    Multi-task loss combining:
      - Focal Tversky loss for 3D segmentation (where is the nodule?)
      - Focal loss         for classification  (is there a nodule?)

    Focal variants replace the previous Tversky + BCE pair to give more
    gradient weight to the hard, rare cases that dominate LUNA16:
      - hard-to-segment nodule voxels (Focal Tversky, gamma > 1)
      - hard-to-classify near-miss patches (Focal Loss, gamma=2)

    seg_weight + cls_weight should sum to 1.0
    Higher seg_weight = prioritise localisation accuracy
    Higher cls_weight = prioritise detection accuracy
    """
    def __init__(
        self,
        seg_weight: float = 0.8,
        cls_weight: float = 0.2,
        # ── segmentation (Focal Tversky) hyperparameters ──
        seg_alpha: float = 0.3,
        seg_beta:  float = 0.7,
        seg_gamma: float = 1.33,
        # ── classification (Focal Loss) hyperparameters ──
        cls_alpha: float = 0.25,
        cls_gamma: float = 2.0,
    ):
        super().__init__()
        self.seg = FocalTverskyLoss(alpha=seg_alpha, beta=seg_beta, gamma=seg_gamma)
        self.cls = FocalLoss(alpha=cls_alpha, gamma=cls_gamma, reduction="mean")
        self.seg_weight = seg_weight
        self.cls_weight = cls_weight

    def forward(self, seg_pred, seg_target, cls_pred, cls_target):
        seg_loss = self.seg(seg_pred, seg_target)
        cls_loss = self.cls(cls_pred, cls_target)
        return self.seg_weight * seg_loss + self.cls_weight * cls_loss


# ─────────────────────────────────────────────
# SANITY CHECK
# ─────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # features=16 is safer for most GPUs with 3D patches
    model = UNet3DWithClassifier(in_channels=1, out_channels=1, features=16).to(device)

    # Simulate a batch of 2 CT patches: (batch, channels, Z, Y, X)
    dummy_input = torch.randn(2, 1, 64, 64, 64).to(device)

    seg_map, cancer_prob = model(dummy_input)

    print(f"Input shape            : {dummy_input.shape}")     # (2, 1, 64, 64, 64)
    print(f"Segmentation map shape : {seg_map.shape}")         # (2, 1, 64, 64, 64)
    print(f"Cancer probability     : {cancer_prob.shape}")     # (2, 1)
    print(f"Cancer % for sample 1  : {cancer_prob[0].item() * 100:.1f}%")
    print(f"Cancer % for sample 2  : {cancer_prob[1].item() * 100:.1f}%")

    # Test combined loss
    dummy_seg_target = torch.randint(0, 2, (2, 1, 64, 64, 64)).float().to(device)
    dummy_cls_target = torch.tensor([[1.0], [0.0]]).to(device)  # sample1=nodule, sample2=nothing

    criterion = CombinedLoss(seg_weight=0.7, cls_weight=0.3)
    loss = criterion(seg_map, dummy_seg_target, cancer_prob, dummy_cls_target)
    print(f"Combined loss          : {loss.item():.4f}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable params : {total_params:,}")