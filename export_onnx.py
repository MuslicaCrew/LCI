"""
export_onnx.py — Export trained UNet3DWithClassifier to ONNX format.

Usage:
    python export_onnx.py                          # uses defaults
    python export_onnx.py --weights best_model.pth --out unet3d.onnx
    python export_onnx.py --weights best_model.pth --out unet3d.onnx --features 16
    python export_onnx.py --weights best_model.pth --no-verify

Outputs:
    unet3d.onnx  — exported model ready for ONNX Runtime inference

Requirements:
    pip install onnx onnxruntime --break-system-packages
    (for GPU inference: pip install onnxruntime-gpu --break-system-packages)
"""

import argparse
import sys
import numpy as np
import torch

from unet3d_classifier import UNet3DWithClassifier
import warnings
warnings.filterwarnings("ignore", message="Can't initialize amdsmi")

def load_trained_weights(
    model: torch.nn.Module,
    weights_path: str,
    device: torch.device,
) -> None:
    """
    Load weights from a training checkpoint into an uncompiled model.

    best_model.pth is the full training dict (epoch/optimizer/scheduler/
    scaler/best_val_dice + weights under "model"), not a bare state_dict.
    Training also wrapped the model in torch.compile, which prefixes every
    key with "_orig_mod." — strip it since this model is uncompiled.
    """
    ckpt = torch.load(weights_path, map_location=device, weights_only=True)

    # Unwrap: full checkpoint dict vs. bare state_dict
    if isinstance(ckpt, dict) and "model" in ckpt:
        epoch = ckpt.get("epoch", "?")
        best = ckpt.get("best_val_dice", float("nan"))
        print(f"  Checkpoint: epoch {epoch}, best_val_dice={best:.4f}")
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    # Strip torch.compile's "_orig_mod." prefix (leading occurrence only)
    state_dict = {
        k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()
    }

    model.load_state_dict(state_dict)



# ─────────────────────────────────────────────
# EXPORT
# ─────────────────────────────────────────────

def export(
    weights_path: str,
    output_path:  str,
    features:     int,
    patch_size:   int,
    #opset:        int,
    device:       torch.device,
) -> None:
    """
    Load trained weights and trace the model to ONNX.

    Notes:
        - Model must NOT be torch.compiled before export — tracing
          requires the raw nn.Module, not the compiled wrapper.
        - dummy_input values are irrelevant; only shape and dtype matter.
        - dynamic_axes allows any batch size at inference time.
    """
    print(f"\n{'='*50}")
    print("ONNX EXPORT")
    print(f"{'='*50}")
    print(f"  Weights   : {weights_path}")
    print(f"  Output    : {output_path}")
    print(f"  Features  : {features}")
    print(f"  Patch size: {patch_size}³")
    #print(f"  Opset     : {opset}")
    print(f"  Device    : {device}")

    # ── Load model — raw nn.Module, no torch.compile ──────────────────
    print("\nLoading model weights...")
    model = UNet3DWithClassifier(
        in_channels=1,
        out_channels=1,
        features=features,
    ).to(device)

    load_trained_weights(model, weights_path, device)
    model.eval()
    print("  Weights loaded ✓")

    # ── Dummy input — shape must match training exactly ───────────────
    # Values don't matter, only shape=(B, C, D, H, W) and dtype=float32
    dummy_input = torch.randn(1, 1, patch_size, patch_size, patch_size, device=device)

    # ── Export ────────────────────────────────────────────────────────
    print(f"\nTracing and exporting → {output_path} ...")
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,          # embed weights into the .onnx file
            #opset_version=opset,         # opset 17 has solid 3D conv support
            do_constant_folding=True,    # fold constant subgraphs → smaller file
            input_names=["ct_patch"],
            output_names=["seg_map", "cls_prob"],
            dynamic_axes={
                # batch dim is flexible — at inference pass any batch size
                "ct_patch" : {0: "batch_size"},
                "seg_map"  : {0: "batch_size"},
                "cls_prob" : {0: "batch_size"},
            },
        )
    print(f"  Export complete ✓")

    # ── File size report ──────────────────────────────────────────────
    import os
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"  File size : {size_mb:.1f} MB")


# ─────────────────────────────────────────────
# VERIFY
# ─────────────────────────────────────────────

def verify(
    weights_path: str,
    output_path:  str,
    features:     int,
    patch_size:   int,
    device:       torch.device,
    tolerance:    float = 1e-4,
) -> bool:
    """
    Run the same dummy input through PyTorch and ONNX Runtime,
    compare outputs. Max absolute difference should be < tolerance.

    Returns True if verification passes, False otherwise.
    """
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        print("\n[VERIFY] onnx / onnxruntime not installed — skipping verification.")
        print("  pip install onnx onnxruntime --break-system-packages")
        return False

    print(f"\n{'='*50}")
    print("ONNX VERIFICATION")
    print(f"{'='*50}")

    # ── Structural check ──────────────────────────────────────────────
    print("Checking ONNX model structure...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("  Structure valid ✓")

    # ── Numerical check ───────────────────────────────────────────────
    print("Comparing PyTorch vs ONNX Runtime outputs...")

    # Fixed seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    dummy_np = np.random.randn(1, 1, patch_size, patch_size, patch_size).astype(np.float32)

    # PyTorch reference output
    model = UNet3DWithClassifier(
        in_channels=1, out_channels=1, features=features
    ).to(device)
    load_trained_weights(model, weights_path, device)
    model.eval()
    with torch.no_grad():
        pt_seg, pt_cls = model(torch.from_numpy(dummy_np).to(device))
        pt_seg = pt_seg.cpu().numpy()
        pt_cls = pt_cls.cpu().numpy()

    # ONNX Runtime output
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] \
        if device.type == "cuda" else ["CPUExecutionProvider"]
    session = ort.InferenceSession(output_path, providers=providers)
    onnx_seg, onnx_cls = session.run(None, {"ct_patch": dummy_np})

    # Compare
    seg_diff = np.abs(pt_seg - onnx_seg).max()
    cls_diff = np.abs(pt_cls - onnx_cls).max()

    print(f"  seg_map  max diff : {seg_diff:.2e}  {'✓' if seg_diff < tolerance else '✗ FAIL'}")
    print(f"  cls_prob max diff : {cls_diff:.2e}  {'✓' if cls_diff < tolerance else '✗ FAIL'}")

    passed = seg_diff < tolerance and cls_diff < tolerance

    if passed:
        print("\n  Verification PASSED ✓")
        print(f"  Both outputs match within tolerance ({tolerance})")
    else:
        print("\n  Verification FAILED ✗")
        print("  Try a lower opset version or check for unsupported ops.")

    return passed


# ─────────────────────────────────────────────
# INFERENCE EXAMPLE
# ─────────────────────────────────────────────

def print_inference_example(output_path: str, patch_size: int) -> None:
    """Print a ready-to-use inference snippet for the exported model."""
    print(f"\n{'='*50}")
    print("HOW TO RUN INFERENCE WITH THE EXPORTED MODEL")
    print(f"{'='*50}")
    print(f"""
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession(
    "{output_path}",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

# Your preprocessed CT patch — float32, normalised to [0, 1]
patch = np.random.randn(1, 1, {patch_size}, {patch_size}, {patch_size}).astype(np.float32)
#                        ^  ^   ^────────────────────────^
#                  batch=1  ch         D × H × W

seg_map, cls_prob = session.run(None, {{"ct_patch": patch}})

# seg_map  : (1, 1, {patch_size}, {patch_size}, {patch_size}) — raw logits, apply sigmoid for probabilities
# cls_prob : (1, 1)              — raw logits, apply sigmoid for nodule probability

import torch
seg_prob  = torch.sigmoid(torch.from_numpy(seg_map)).numpy()
nodule_prob = torch.sigmoid(torch.from_numpy(cls_prob)).item()

print(f"Nodule probability: {{nodule_prob*100:.1f}}%")
print(f"Seg map range     : {{seg_prob.min():.3f}} – {{seg_prob.max():.3f}}")
""")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export UNet3DWithClassifier to ONNX"
    )
    parser.add_argument("--weights",    type=str,  default="best_model.pth",  help="Path to trained .pth weights")
    parser.add_argument("--out",        type=str,  default="unet3d.onnx",     help="Output .onnx file path")
    parser.add_argument("--features",   type=int,  default=32,                help="Model features (must match training)")
    parser.add_argument("--patch-size", type=int,  default=64,                help="Patch size (must match training)")
    #parser.add_argument("--opset",      type=int,  default=17,                help="ONNX opset version")
    parser.add_argument("--no-verify",  action="store_true",                  help="Skip numerical verification step")
    args = parser.parse_args()

    # CPU-only: ROCm GPU tracing captures aten.miopen_batch_norm,
    # which the ONNX exporter cannot translate. The exported model
    # is device-agnostic regardless of export device.
    device = torch.device("cpu")
    print(f"Device : {device}")

    # ── Export ────────────────────────────────────────────────────────
    export(
        weights_path=args.weights,
        output_path=args.out,
        features=args.features,
        patch_size=args.patch_size,
        device=device,
    )

    # ── Verify ────────────────────────────────────────────────────────
    if not args.no_verify:
        passed = verify(
            weights_path=args.weights,
            output_path=args.out,
            features=args.features,
            patch_size=args.patch_size,
            device=device,
        )
        if not passed:
            print("\nExport completed but verification failed.")
            print("The .onnx file exists but outputs may not match PyTorch.")
            sys.exit(1)

    # ── Usage example ─────────────────────────────────────────────────
    print_inference_example(args.out, args.patch_size)
    print(f"\nDone. Model exported → {args.out}\n")


if __name__ == "__main__":
    main()