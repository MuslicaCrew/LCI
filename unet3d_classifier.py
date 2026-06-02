import torch
import torch.nn as nn
import torch.nn.functional as functional

class ConvBlock3D(nn.Module):
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

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock3D(in_channels, out_channels)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.conv(x)
        x = self.pool(skip)
        return x, skip

class UpBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock3D(out_channels * 2, out_channels)

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class UNet3DWithClassifier(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=16):
        super().__init__()

        self.enc1 = DownBlock3D(in_channels, features)          # 64³ → 32³, 16 filters
        self.enc2 = DownBlock3D(features, features * 2)         # 32³ → 16³, 32 filters
        self.enc3 = DownBlock3D(features * 2, features * 4)     # 16³ → 8³,  64 filters
        self.enc4 = DownBlock3D(features * 4, features * 8)     # 8³  → 4³,  128 filters

        self.bottleneck = ConvBlock3D(features * 8, features * 16)  # 256 filters

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(features * 16, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, 1),
        )

        self.dec4 = UpBlock3D(features * 16, features * 8)     # 4³  → 8³
        self.dec3 = UpBlock3D(features * 8, features * 4)      # 8³  → 16³
        self.dec2 = UpBlock3D(features * 4, features * 2)      # 16³ → 32³
        self.dec1 = UpBlock3D(features * 2, features)          # 32³ → 64³

        self.seg_head = nn.Conv3d(features, out_channels, kernel_size=1)

    def forward(self, x):
        x, skip1 = self.enc1(x)
        x, skip2 = self.enc2(x)
        x, skip3 = self.enc3(x)
        x, skip4 = self.enc4(x)

        x = self.bottleneck(x)

        cancer_probability = self.classifier(x)

        x = self.dec4(x, skip4)
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)

        seg_map = self.seg_head(x)

        return seg_map, cancer_probability

class DiceLoss(nn.Module):
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

        has_mask = target.sum(dim=1) > 0  # (B,) bool
        if has_mask.any():
            return loss_per_sample[has_mask].mean()
        else:
            return torch.tensor(0.0, device=pred.device, requires_grad=False)


class TverskyLoss(nn.Module):

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
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
    def __init__(
        self,
        alpha: float = 0.3,
        beta:  float = 0.7,
        gamma: float = 1.33,
        smooth: float = 1e-6,
        empty_weight: float = 0.1,  # ← new
    ):
        super().__init__()
        self.alpha  = alpha
        self.beta   = beta
        self.gamma  = gamma
        self.smooth = smooth
        self.empty_weight = empty_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = pred.view(pred.shape[0], -1)
        target_f = target.view(target.shape[0], -1)
        pred_sig = torch.sigmoid(logits)

        has_mask = target_f.sum(dim=1) > 0
        no_mask = ~has_mask

        total_loss = torch.tensor(0.0, device=pred.device)
        if has_mask.any():
            p = pred_sig[has_mask]
            t = target_f[has_mask]
            tp = (p * t).sum(dim=1)
            fp = (p * (1 - t)).sum(dim=1)
            fn = ((1 - p) * t).sum(dim=1)
            tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
            one_minus_tv = (1.0 - tversky).float().clamp(min=self.smooth)
            total_loss = total_loss + one_minus_tv.pow(self.gamma).mean()

        if no_mask.any():
            zeros = torch.zeros_like(logits[no_mask])
            empty_loss = functional.binary_cross_entropy_with_logits(
                logits[no_mask], zeros, reduction="mean"
            )
            total_loss = total_loss + self.empty_weight * empty_loss

        return total_loss


class FocalLoss(nn.Module):
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
        ce = functional.binary_cross_entropy_with_logits(
            logits, target, reduction="none"
        )
        p_t = torch.exp(-ce.float())
        alpha_t = torch.where(target > 0.5, self.alpha, 1.0 - self.alpha)
        focal_term = (1.0 - p_t).pow(self.gamma)
        loss = alpha_t * focal_term * ce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class CombinedLoss(nn.Module):
    def __init__(
        self,
        seg_weight: float = 0.8,
        cls_weight: float = 0.2,
        seg_alpha: float = 0.3,
        seg_beta:  float = 0.7,
        seg_gamma: float = 1.33,
        seg_empty_weight: float = 0.1,
        cls_alpha: float = 0.25,
        cls_gamma: float = 2.0,
    ):
        super().__init__()
        self.seg = FocalTverskyLoss(alpha=seg_alpha, beta=seg_beta, gamma=seg_gamma, empty_weight=seg_empty_weight)
        self.cls = FocalLoss(alpha=cls_alpha, gamma=cls_gamma, reduction="mean")
        self.seg_weight = seg_weight
        self.cls_weight = cls_weight

    def forward(self, seg_pred, seg_target, cls_pred, cls_target):
        seg_loss = self.seg(seg_pred, seg_target)
        cls_loss = self.cls(cls_pred, cls_target)
        return self.seg_weight * seg_loss + self.cls_weight * cls_loss

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = UNet3DWithClassifier(in_channels=1, out_channels=1, features=16).to(device)

    dummy_input = torch.randn(2, 1, 64, 64, 64).to(device)

    seg_map, cancer_prob = model(dummy_input)

    print(f"Input shape            : {dummy_input.shape}")
    print(f"Segmentation map shape : {seg_map.shape}")
    print(f"Cancer probability     : {cancer_prob.shape}")
    print(f"Cancer % for sample 1  : {cancer_prob[0].item() * 100:.1f}%")
    print(f"Cancer % for sample 2  : {cancer_prob[1].item() * 100:.1f}%")

    dummy_seg_target = torch.randint(0, 2, (2, 1, 64, 64, 64)).float().to(device)
    dummy_cls_target = torch.tensor([[1.0], [0.0]]).to(device)

    criterion = CombinedLoss(seg_weight=0.7, cls_weight=0.3)
    loss = criterion(seg_map, dummy_seg_target, cancer_prob, dummy_cls_target)
    print(f"Combined loss          : {loss.item():.4f}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable params : {total_params:,}")