"""
precompute_patches.py — LUNA16 Phase 1 Preprocessing
======================================================
Offline script. Run this ONCE before training.

Balancing strategy
------------------
All positives (~1,200) are always kept.
Negatives are randomly downsampled to `neg_ratio × n_positives` (default 100×).
At 100:1 this produces ~121,200 candidates total — well within a 200 GB budget
even without compression.

Compression
-----------
Files are saved with np.savez_compressed (.npz).
CT patches compress ~3–5×; all-zero seg masks for negatives compress ~20×.
Compared to raw .npy the total footprint is typically 5–8× smaller.

Disk budget guard
-----------------
Before writing anything the script estimates the uncompressed upper bound and
aborts if it would exceed --max_gb (default 150 GB, comfortably under 200 GB).
The real on-disk size after compression will be significantly lower.

Directory layout expected:
    data/
        subset0/  *.mhd  *.raw
        subset1/  ...
        ...
    seg-lungs/    *.mhd  *.raw   (optional, same UIDs as data/)
    candidates.csv
    annotations.csv

Output:
    patches/          — one .npz per candidate  (key: 'patch')
    seg_masks/        — one .npz per candidate  (key: 'seg_mask')
    index.csv         — master index consumed by NodulePatchDataset

Loading in the Dataset:
    patch    = np.load(row.patch_path)['patch']        # float32 (64,64,64)
    seg_mask = np.load(row.seg_mask_path)['seg_mask']  # float32 (64,64,64)

Usage:
    python precompute_patches.py                          # all defaults
    python precompute_patches.py --patch_size 64 \\
        --data_dir data --mask_dir seg-lungs \\
        --candidates candidates.csv --annotations annotations.csv \\
        --out_dir precomputed --workers 8 \\
        --neg_ratio 100 --max_gb 150
"""

import argparse
import glob
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import numpy as np
import pandas as pd
import SimpleITK as sitk


# ─────────────────────────────────────────────────────────────────────────────
# PATCH EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_patch(
    volume: np.ndarray,       # (D, H, W) float32, already normalised
    voxel_zyx: np.ndarray,    # (3,) int — center in (z, y, x) order
    patch_size: int,
    pad_value: float = 0.0,   # -1000 HU normalises to 0.0
) -> np.ndarray:
    """
    Extract a fixed cubic patch centred on voxel_zyx.
    Pads with pad_value wherever the patch extends beyond volume bounds.

    Handles edge cases:
      - Candidate near the volume boundary (partial overlap)
      - Candidate centre outside the volume entirely (returns all-pad patch)

    Returns:
        patch: float32 ndarray of shape (patch_size, patch_size, patch_size)
    """
    half = patch_size // 2
    z, y, x = int(voxel_zyx[0]), int(voxel_zyx[1]), int(voxel_zyx[2])
    D, H, W = volume.shape

    patch = np.full((patch_size, patch_size, patch_size), pad_value, dtype=np.float32)

    # Desired slice boundaries in volume coordinates
    z0_vol, z1_vol = z - half, z + half
    y0_vol, y1_vol = y - half, y + half
    x0_vol, x1_vol = x - half, x + half

    # Clamped read boundaries — the intersection with the actual volume
    z0_r, z1_r = max(0, z0_vol), min(D, z1_vol)
    y0_r, y1_r = max(0, y0_vol), min(H, y1_vol)
    x0_r, x1_r = max(0, x0_vol), min(W, x1_vol)

    # If the centre landed fully outside the volume on any axis, return the pad patch
    if z0_r >= z1_r or y0_r >= y1_r or x0_r >= x1_r:
        return patch

    # Destination slice start inside the patch (offset from the pad boundary)
    z0_p = z0_r - z0_vol
    y0_p = y0_r - y0_vol
    x0_p = x0_r - x0_vol

    # Read sizes — how many voxels we actually got on each axis
    dz = z1_r - z0_r
    dy = y1_r - y0_r
    dx = x1_r - x0_r

    # Explicit destination end — never overflow the patch array
    patch[z0_p:z0_p + dz, y0_p:y0_p + dy, x0_p:x0_p + dx] = \
        volume[z0_r:z1_r, y0_r:y1_r, x0_r:x1_r]

    return patch


# ─────────────────────────────────────────────────────────────────────────────
# SEGMENTATION MASK GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def make_seg_mask(
    diameter_mm: float,
    spacing: np.ndarray,    # (z, y, x) mm/voxel — already reversed by load_itk_image
    patch_size: int,
) -> np.ndarray:
    """
    Build a binary spherical mask of shape (patch_size, patch_size, patch_size).
    The nodule is always centred in the patch (centre = patch_size // 2).

    For negative candidates (diameter_mm == 0) returns an all-zero mask.
    All-zero masks compress extremely well with savez_compressed (~20×).

    Uses vectorised mgrid — no Python loops.
    """
    mask = np.zeros((patch_size, patch_size, patch_size), dtype=np.float32)

    if diameter_mm <= 0:
        return mask

    radius_voxels = (diameter_mm / 2.0) / spacing  # per-axis radius in voxels
    center = patch_size // 2

    zz, yy, xx = np.mgrid[0:patch_size, 0:patch_size, 0:patch_size]
    dist_sq = (
        ((zz - center) / radius_voxels[0]) ** 2 +
        ((yy - center) / radius_voxels[1]) ** 2 +
        ((xx - center) / radius_voxels[2]) ** 2
    )
    mask[dist_sq <= 1.0] = 1.0
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# PER-SCAN WORKER  (runs in a subprocess when parallelised)
# ─────────────────────────────────────────────────────────────────────────────

def process_scan(
    seriesuid: str,
    mhd_path: str,
    candidates_for_scan: pd.DataFrame,   # already balanced rows for this UID
    annotations_for_scan: pd.DataFrame,
    mask_dir: Optional[str],
    patch_dir: str,
    seg_dir: str,
    patch_size: int,
    subset: str,
) -> list[dict]:
    """
    Process all candidates that belong to a single CT scan.
    The CT volume is loaded once and reused for all candidates in the scan.

    Returns a list of index-row dicts — one per successfully saved candidate.
    Exceptions are caught per-scan so one bad file does not abort the whole run.
    """
    rows: list[dict] = []

    try:
        # ── 1. Load CT volume (once per scan) ─────────────────────────
        ct_array, origin, spacing = load_itk_image(mhd_path)
        # ct_array : float32 (D, H, W)  HU values, axes (z, y, x)
        # origin   : (z, y, x) world coords of voxel [0,0,0]
        # spacing  : (z, y, x) mm per voxel

        # ── 2. Apply lung segmentation mask (optional) ─────────────────
        mask_path = os.path.join(mask_dir, os.path.basename(mhd_path)) if mask_dir else None
        if mask_path and os.path.exists(mask_path):
            mask_array, _, _ = load_itk_image(mask_path)
            ct_array[mask_array == 0] = -1000.0  # non-lung → air HU

        # ── 3. Normalise to [0, 1] ────────────────────────────────────
        ct_array = normalize_planes(ct_array)
        # After normalisation: -1000 HU → 0.0, so pad_value=0.0 is correct.

        # ── 4. Build annotation lookup for this scan ───────────────────
        ann_coords    = None
        ann_diameters = None
        if len(annotations_for_scan) > 0:
            ann_coords    = annotations_for_scan[['coordX', 'coordY', 'coordZ']].values.astype(float)
            ann_diameters = annotations_for_scan['diameter_mm'].values.astype(float)

        # ── 5. Process each (balanced) candidate ──────────────────────
        for local_idx, (_, cand) in enumerate(candidates_for_scan.iterrows()):
            label = int(cand['class'])

            # World → voxel. load_itk_image reverses axes to (z,y,x),
            # so voxel_zyx is already in the correct numpy indexing order.
            world_xyz = np.array([cand['coordX'], cand['coordY'], cand['coordZ']])
            voxel_zyx = world_to_voxel_coordinates(world_xyz, origin, spacing)

            # ── 5a. Extract CT patch ──────────────────────────────────
            patch = extract_patch(ct_array, voxel_zyx, patch_size, pad_value=0.0)

            # ── 5b. Determine nodule diameter ─────────────────────────
            if label == 1 and ann_coords is not None:
                # Match to the spatially closest annotation in world space
                dists = np.linalg.norm(ann_coords - world_xyz, axis=1)
                closest_idx = int(np.argmin(dists))
                diameter_mm = float(ann_diameters[closest_idx])
            else:
                diameter_mm = 0.0

            # ── 5c. Build segmentation mask ───────────────────────────
            seg_mask = make_seg_mask(diameter_mm, spacing, patch_size)

            # ── 5d. Save compressed to disk ───────────────────────────
            # np.savez_compressed saves a .npz archive.
            # Load with: np.load(path)['patch'] or ['seg_mask']
            stem          = f"{seriesuid}_{local_idx}"
            patch_path    = os.path.join(patch_dir, f"{stem}.npz")
            seg_mask_path = os.path.join(seg_dir,   f"{stem}.npz")

            np.savez_compressed(patch_path,    patch=patch)
            np.savez_compressed(seg_mask_path, seg_mask=seg_mask)

            rows.append({
                "seriesuid":     seriesuid,
                "subset":        subset,
                "label":         label,
                "diameter_mm":   diameter_mm,
                "coord_world_x": cand['coordX'],
                "coord_world_y": cand['coordY'],
                "coord_world_z": cand['coordZ'],
                "patch_path":    patch_path,
                "seg_mask_path": seg_mask_path,
            })

    except Exception:
        print(f"\n[ERROR] Failed on scan {seriesuid}:\n{traceback.format_exc()}")

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# BALANCING
# ─────────────────────────────────────────────────────────────────────────────

def balance_candidates(
    candidates: pd.DataFrame,
    neg_ratio: int,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Keep ALL positives and downsample negatives to neg_ratio × n_positives.

    The negative pool is sampled with a fixed seed for reproducibility.
    If there are fewer negatives than the target, all negatives are kept
    and a warning is printed.

    Returns the balanced DataFrame, shuffled.
    """
    positives = candidates[candidates['class'] == 1]
    negatives = candidates[candidates['class'] == 0]

    n_pos        = len(positives)
    n_neg_target = n_pos * neg_ratio

    if len(negatives) <= n_neg_target:
        print(
            f"[WARN] Requested {n_neg_target:,} negatives ({neg_ratio}× {n_pos:,} positives) "
            f"but only {len(negatives):,} available — keeping all negatives."
        )
        sampled_negatives = negatives
    else:
        sampled_negatives = negatives.sample(n=n_neg_target, random_state=random_seed)

    balanced = pd.concat([positives, sampled_negatives])
    balanced = balanced.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    print(
        f"Balancing  : {n_pos:,} positives  +  {len(sampled_negatives):,} negatives"
        f"  =  {len(balanced):,} total  (1:{neg_ratio} ratio)"
    )
    return balanced


# ─────────────────────────────────────────────────────────────────────────────
# DISK BUDGET CHECK
# ─────────────────────────────────────────────────────────────────────────────

def check_disk_budget(n_candidates: int, patch_size: int, max_gb: float) -> None:
    """
    Pre-run safety check using the exact uncompressed size as the upper bound.

    We cannot know the compression ratio before writing any files, so we compare
    the worst-case uncompressed size against --max_gb. In practice compressed
    files will be much smaller (57x observed on masked lung CT data), so if the
    uncompressed estimate passes the budget check, the compressed output
    definitely will too.

    Raises SystemExit if the budget would be exceeded.
    """
    bytes_per_array = patch_size ** 3 * 4
    gb_uncompressed = (n_candidates * bytes_per_array * 2) / 1e9   # patch + seg_mask

    print(f"Disk estimate (uncompressed upper bound) : {gb_uncompressed:.1f} GB")
    print(f"Disk budget (--max_gb)                   : {max_gb:.1f} GB")
    print( "Note: compressed output will be much smaller")

    if gb_uncompressed > max_gb:
        raise SystemExit(
            f"\n[ABORT] Uncompressed upper bound ({gb_uncompressed:.1f} GB) exceeds "
            f"--max_gb ({max_gb:.1f} GB).\n"
            f"Lower --neg_ratio (currently {n_candidates:,} candidates) "
            f"or raise --max_gb if you have the space.\n"
            f"Remember: actual compressed size will be far smaller than this estimate."
        )

    print("[OK] Disk budget check passed.\n")


def measure_compressed_size(index_df: "pd.DataFrame", sample_size: int = 200) -> float:
    """
    After files are written, sample real .npz files to estimate total compressed
    disk usage. Returns estimated GB.

    Samples from both patch and seg_mask columns so the average accounts for
    the different compression ratios of each type (seg masks compress far more).
    """
    all_paths = index_df["patch_path"].tolist() + index_df["seg_mask_path"].tolist()
    rng = np.random.default_rng(0)
    n_sample = min(sample_size, len(all_paths))
    sampled = rng.choice(all_paths, size=n_sample, replace=False)
    sampled_bytes = sum(os.path.getsize(p) for p in sampled if os.path.exists(p))
    avg_bytes = sampled_bytes / max(n_sample, 1)
    return (avg_bytes * len(all_paths)) / 1e9

# ─────────────────────────────────────────────────────────────────────────────
# LOADING IMAGES
# ─────────────────────────────────────────────────────────────────────────────


def load_itk_image(filename):
    itk_image = sitk.ReadImage(filename)
    numpy_image = sitk.GetArrayFromImage(itk_image).astype(np.float32)

    numpy_origin = np.array(list(reversed(itk_image.GetOrigin())))
    numpy_spacing = np.array(list(reversed(itk_image.GetSpacing())))

    return numpy_image, numpy_origin, numpy_spacing

def world_to_voxel_coordinates(worldCoordinates, origin, spacing):
    stretched_voxel_coordinates = np.absolute(worldCoordinates - origin)
    voxel_coordinates = stretched_voxel_coordinates / spacing
    return np.round(voxel_coordinates).astype(int)

def normalize_planes(array):
    maxHU = 400
    minHU = -1000

    array = (array - minHU) / (maxHU - minHU)
    array[array < 0] = 0
    array[array > 1] = 1
    return array

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="LUNA16 Phase 1 patch precomputation")
    parser.add_argument("--data_dir",    default="data",            help="Root dir containing subset0..subset9")
    parser.add_argument("--mask_dir",    default="seg-lungs",       help="Dir with lung segmentation .mhd files (optional)")
    parser.add_argument("--candidates",  default="candidates.csv",  help="Path to candidates.csv")
    parser.add_argument("--annotations", default="annotations.csv", help="Path to annotations.csv")
    parser.add_argument("--out_dir",     default="precomputed",     help="Output root directory")
    parser.add_argument("--patch_size",  type=int,   default=64,    help="Cubic patch side length in voxels")
    parser.add_argument("--workers",     type=int,   default=4,     help="Parallel worker processes (1 = serial/debug mode)")
    parser.add_argument("--neg_ratio",   type=int,   default=100,   help="Negatives per positive (e.g. 100 → 1:100 balance)")
    parser.add_argument("--max_gb",      type=float, default=150.0, help="Abort if uncompressed estimate exceeds this many GB")
    parser.add_argument("--seed",        type=int,   default=42,    help="Random seed for negative downsampling")
    args = parser.parse_args()

    # ── Resolve output directories ────────────────────────────────────────
    patch_dir  = os.path.join(args.out_dir, "patches")
    seg_dir    = os.path.join(args.out_dir, "seg_masks")
    index_path = os.path.join(args.out_dir, "index.csv")
    os.makedirs(patch_dir, exist_ok=True)
    os.makedirs(seg_dir,   exist_ok=True)

    # ── Validate mask_dir ─────────────────────────────────────────────────
    mask_dir: Optional[str] = args.mask_dir if os.path.isdir(args.mask_dir) else None
    if mask_dir is None:
        print(f"[WARN] mask_dir '{args.mask_dir}' not found — skipping lung masking.")

    # ── Load CSVs ─────────────────────────────────────────────────────────
    candidates  = pd.read_csv(args.candidates)
    annotations = pd.read_csv(args.annotations)
    print(f"Candidates (raw) : {len(candidates):,}  ({int(candidates['class'].sum()):,} positive)")
    print(f"Annotations      : {len(annotations):,}")

    # ── Build uid → (mhd path, subset name) map ──────────────────────────
    uid_to_path:   dict[str, str] = {}
    uid_to_subset: dict[str, str] = {}
    n_subsets_found = 0
    for subset_idx in range(10):
        subset_name = f"subset{subset_idx}"
        subset_path = os.path.join(args.data_dir, subset_name)
        if not os.path.isdir(subset_path):
            continue
        n_subsets_found += 1
        for mhd_path in glob.glob(os.path.join(subset_path, "*.mhd")):
            uid = os.path.splitext(os.path.basename(mhd_path))[0]
            uid_to_path[uid]   = mhd_path
            uid_to_subset[uid] = subset_name

    print(f"Scans found      : {len(uid_to_path):,}  across {n_subsets_found}/10 subsets")

    # ── Filter candidates to scans we have on disk ────────────────────────
    candidates = candidates[candidates['seriesuid'].isin(uid_to_path)].copy()
    print(f"Candidates after path filter : {len(candidates):,}")

    # ── Balance: all positives + neg_ratio × negatives ───────────────────
    candidates = balance_candidates(candidates, args.neg_ratio, args.seed)

    # ── Disk budget check (uncompressed upper bound) ──────────────────────
    check_disk_budget(len(candidates), args.patch_size, args.max_gb)

    # ── Split into done / to-do by scan ──────────────────────────────────
    grouped: dict[str, pd.DataFrame] = {uid: grp for uid, grp in candidates.groupby("seriesuid")}

    # Build a set of stems that exist in BOTH patch_dir and seg_dir with a
    # single os.listdir() call per directory instead of one os.path.exists()
    # call per candidate (240K+ filesystem calls on WSL/NTFS is very slow).
    patch_stems   = {os.path.splitext(f)[0] for f in os.listdir(patch_dir)}
    seg_stems     = {os.path.splitext(f)[0] for f in os.listdir(seg_dir)}
    done_stems    = patch_stems & seg_stems   # both files must exist

    uids_done:       list[str] = []
    uids_to_process: list[str] = []
    for uid, grp in grouped.items():
        stems = [f"{uid}_{i}" for i in range(len(grp))]
        if all(s in done_stems for s in stems):
            uids_done.append(uid)
        else:
            uids_to_process.append(uid)

    print(f"Scans already complete : {len(uids_done):,}")
    print(f"Scans to process       : {len(uids_to_process):,}")

    # ── Collect already-done rows from existing index.csv ─────────────────
    all_rows: list[dict] = []
    if os.path.exists(index_path) and uids_done:
        existing_index = pd.read_csv(index_path)
        done_rows = existing_index[existing_index['seriesuid'].isin(uids_done)].to_dict('records')
        all_rows.extend(done_rows)
        print(f"Loaded {len(done_rows):,} existing index rows from {index_path}")

    # ── Dispatch work ─────────────────────────────────────────────────────
    total_scans = len(uids_to_process)

    def iter_scan_args():
        for uid in uids_to_process:
            yield (
                uid,
                uid_to_path[uid],
                grouped[uid],
                annotations[annotations['seriesuid'] == uid],
                mask_dir,
                patch_dir,
                seg_dir,
                args.patch_size,
                uid_to_subset[uid],
            )

    if args.workers == 1:
        # ── Serial mode — full tracebacks, easier to debug ───────────
        for i, scan_args in enumerate(iter_scan_args(), 1):
            uid  = scan_args[0]
            rows = process_scan(*scan_args)
            all_rows.extend(rows)
            pos  = sum(1 for r in rows if r['label'] == 1)
            print(f"[{i:4d}/{total_scans}] {uid}  → {len(rows):4d} patches  ({pos} pos)", flush=True)
    else:
        # ── Parallel mode — one process per scan ──────────────────────
        completed = 0
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(process_scan, *scan_args): scan_args[0]
                for scan_args in iter_scan_args()
            }
            for future in as_completed(futures):
                uid = futures[future]
                completed += 1
                try:
                    rows = future.result()
                    all_rows.extend(rows)
                    pos = sum(1 for r in rows if r['label'] == 1)
                    print(
                        f"[{completed:4d}/{total_scans}] {uid}"
                        f"  → {len(rows):4d} patches  ({pos} pos)",
                        flush=True,
                    )
                except Exception:
                    print(f"[{completed:4d}/{total_scans}] {uid} — FAILED\n{traceback.format_exc()}")

    # ── Write index.csv ───────────────────────────────────────────────────
    index_df = pd.DataFrame(all_rows, columns=[
        "seriesuid", "subset", "label", "diameter_mm",
        "coord_world_x", "coord_world_y", "coord_world_z",
        "patch_path", "seg_mask_path",
    ])
    index_df.to_csv(index_path, index=False)

    # ── Summary ───────────────────────────────────────────────────────────
    total = len(index_df)
    n_pos = int((index_df['label'] == 1).sum())
    n_neg = int((index_df['label'] == 0).sum())
    sample_size = 2000
    estimated_gb = measure_compressed_size(index_df, sample_size=sample_size)

    print("\n" + "=" * 60)
    print("PRECOMPUTATION COMPLETE")
    print("=" * 60)
    print(f"  Total patches saved : {total:,}")
    print(f"  Positives           : {n_pos:,}  ({100*n_pos/total:.1f}%)")
    print(f"  Negatives           : {n_neg:,}  ({100*n_neg/total:.1f}%)")
    print(f"  Patch dir           : {patch_dir}")
    print(f"  Seg mask dir        : {seg_dir}")
    print(f"  Index               : {index_path}")
    print(f"  Est. disk usage     : ~{estimated_gb:.2f} GB  (compressed, sampled from {sample_size} files)")
    print("=" * 60)
    print("\nTo load in your Dataset:")
    print("  patch    = np.load(row.patch_path)['patch']")
    print("  seg_mask = np.load(row.seg_mask_path)['seg_mask']")
    print("=" * 60)


if __name__ == "__main__":
    main()
