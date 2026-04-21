import torch
import torch.nn as nn


# ─────────────────────────────────────────────
# BUILDING BLOCKS
# ─────────────────────────────────────────────

class ConvBlock(nn.Module):
    """One encoder/decoder block: Conv → BN → ReLU → Conv → BN → ReLU"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  # Conv
            nn.BatchNorm2d(out_channels),                                     # BN
            nn.ReLU(inplace=True),                                            # ReLU
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), # Conv
            nn.BatchNorm2d(out_channels),                                     # BN
            nn.ReLU(inplace=True),                                            # ReLU
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    """Encoder step: ConvBlock followed by MaxPool"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # halves spatial size

    def forward(self, x):
        skip = self.conv(x)   # save for skip connection
        x = self.pool(skip)
        return x, skip        # return both compressed x and the skip


class UpBlock(nn.Module):
    """Decoder step: Upsample → concat skip → ConvBlock"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels * 2, out_channels)  # *2 because of skip concat

    def forward(self, x, skip):
        x = self.upsample(x)          # double spatial size
        x = torch.cat([x, skip], dim=1)  # concatenate skip connection
        return self.conv(x)


# ─────────────────────────────────────────────
# FULL U-NET WITH BRANCHING CLASSIFIER
# ─────────────────────────────────────────────

class UNetWithClassifier(nn.Module):
    """
    2D U-Net with:
      - Segmentation head: outputs a per-pixel probability map (where is the cancer?)
      - Classification head: outputs a single probability score (is there cancer?)

    Args:
        in_channels:  number of input channels (1 for grayscale CT slices)
        out_channels: number of segmentation classes (1 for binary: tumor vs. not)
        features:     number of filters in the first block (doubles each level)
    
    To make this 3D for full CT volumes:
        - Replace Conv2d  → Conv3d
        - Replace MaxPool2d → MaxPool3d
        - Replace ConvTranspose2d → ConvTranspose3d
        - Replace BatchNorm2d → BatchNorm3d
        - Replace AdaptiveAvgPool2d → AdaptiveAvgPool3d
    """
    def __init__(self, in_channels=1, out_channels=1, features=32):
        super().__init__()

        # ── Encoder (contracting path) ──────────────────
        self.enc1 = DownBlock(in_channels, features)       # 32 filters
        self.enc2 = DownBlock(features, features * 2)      # 64 filters
        self.enc3 = DownBlock(features * 2, features * 4)  # 128 filters
        self.enc4 = DownBlock(features * 4, features * 8)  # 256 filters

        # ── Bottleneck ───────────────────────────────────
        self.bottleneck = ConvBlock(features * 8, features * 16)  # 512 filters

        # ── Classification head (branches off bottleneck) ─
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # Global Average Pooling → (batch, 512, 1, 1)
            nn.Flatten(),              # → (batch, 512)
            nn.Linear(features * 16, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 1),
            nn.Sigmoid()               # outputs value between 0 and 1 (% certainty)
        )

        # ── Decoder (expanding path) ─────────────────────
        self.dec4 = UpBlock(features * 16, features * 8)   # 256 filters
        self.dec3 = UpBlock(features * 8, features * 4)    # 128 filters
        self.dec2 = UpBlock(features * 4, features * 2)    # 64 filters
        self.dec1 = UpBlock(features * 2, features)        # 32 filters

        # ── Segmentation output ───────────────────────────
        self.seg_head = nn.Conv2d(features, out_channels, kernel_size=1)
        self.seg_activation = nn.Sigmoid()  # per-pixel probability

    def forward(self, x):
        # Encoder — save skips
        x, skip1 = self.enc1(x)
        x, skip2 = self.enc2(x)
        x, skip3 = self.enc3(x)
        x, skip4 = self.enc4(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Classification branch (off bottleneck)
        cancer_probability = self.classifier(x)  # shape: (batch, 1)

        # Decoder — use skips
        x = self.dec4(x, skip4)
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)

        # Segmentation output
        seg_map = self.seg_activation(self.seg_head(x))  # shape: (batch, 1, H, W)

        return seg_map, cancer_probability


# ─────────────────────────────────────────────
# LOSS FUNCTION
# ─────────────────────────────────────────────

class DiceLoss(nn.Module):
    """Dice loss for segmentation — handles class imbalance well."""
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = pred.view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        return 1 - (2 * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)


class CombinedLoss(nn.Module):
    """
    Combined loss for multi-task training.
    seg_weight and cls_weight let you control how much each task matters.
    """
    def __init__(self, seg_weight=0.7, cls_weight=0.3):
        super().__init__()
        self.dice = DiceLoss()
        self.bce = nn.BCELoss()
        self.seg_weight = seg_weight
        self.cls_weight = cls_weight

    def forward(self, seg_pred, seg_target, cls_pred, cls_target):
        seg_loss = self.dice(seg_pred, seg_target)
        cls_loss = self.bce(cls_pred, cls_target)
        return self.seg_weight * seg_loss + self.cls_weight * cls_loss


# ─────────────────────────────────────────────
# QUICK SANITY CHECK
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Starting program")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = UNetWithClassifier(in_channels=1, out_channels=1, features=32).to(device)
    print("Model created")

    # Simulate a batch of 2 grayscale CT slices at 256×256
    dummy_input = torch.randn(2, 1, 256, 256).to(device)

    seg_map, cancer_prob = model(dummy_input)

    print(f"Segmentation map shape : {seg_map.shape}")       # (2, 1, 256, 256)
    print(f"Cancer probability     : {cancer_prob.shape}")   # (2, 1)
    print(f"Cancer % for sample 1  : {cancer_prob[0].item() * 100:.1f}%")
    print(f"Cancer % for sample 2  : {cancer_prob[1].item() * 100:.1f}%")

    # Test the combined loss
    dummy_seg_target = torch.randint(0, 2, (2, 1, 256, 256)).float().to(device)
    dummy_cls_target = torch.tensor([[1.0], [0.0]]).to(device)

    criterion = CombinedLoss(seg_weight=0.7, cls_weight=0.3)
    loss = criterion(seg_map, dummy_seg_target, cancer_prob, dummy_cls_target)
    print(f"Combined loss          : {loss.item():.4f}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable params : {total_params:,}")
