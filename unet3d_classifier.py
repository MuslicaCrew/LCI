import torch
import torch.nn as nn

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

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)   # logits → probabilities, applied once here
        pred = pred.view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        return 1 - (2 * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)

class CombinedLoss(nn.Module):
    """
    Multi-task loss combining:
      - Dice loss   for 3D segmentation (where is the nodule?)
      - BCE loss    for classification  (is there a nodule?)

    seg_weight + cls_weight should sum to 1.0
    Higher seg_weight = prioritise localisation accuracy
    Higher cls_weight = prioritise detection accuracy
    """
    def __init__(self, seg_weight=0.7, cls_weight=0.3):
        super().__init__()
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.seg_weight = seg_weight
        self.cls_weight = cls_weight

    def forward(self, seg_pred, seg_target, cls_pred, cls_target):
        seg_loss = self.dice(seg_pred, seg_target)
        cls_loss = self.bce(cls_pred, cls_target)
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