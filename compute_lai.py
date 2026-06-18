"""
compute_lai.py — LAI estimation from DHP segmentation masks

Computes Leaf Area Index (LAI) from binary sky/vegetation masks using the
Miller (1967) integration formula over LiCOR LAI-2200C ring geometry.

Usage
-----
  python compute_lai.py --masks ./masks/ --output lai_results.csv
  python compute_lai.py --masks ./masks/ --output lai_results.csv \\
      --centre 2048 2048 --radius 2048 --down_factor 4

Output CSV columns
------------------
  filename      — mask filename (stem)
  LAI           — estimated LAI (m²/m²)
  gap_fraction  — mean gap fraction over 0–60° zenith

Formula
-------
  LAI = 2 · Σ_k [ −ln(T_k) · cos(θ_k) · w_k ]

  where T_k = mean gap fraction in ring k, θ_k = ring centre zenith (degrees),
  w_k = integration weight following Miller (1967).

  LiCOR LAI-2200C ring geometry (5 rings):
    Ring 1: 0–13°  (centre  7°, weight 0.041)
    Ring 2: 13–30° (centre 23°, weight 0.131)
    Ring 3: 30–46° (centre 38°, weight 0.201)
    Ring 4: 46–61° (centre 53°, weight 0.290)
    Ring 5: 61–74° (centre 68°, weight 0.337)

  Gap fraction per ring is the mean over 36 azimuth cells of 10° each,
  after spatial averaging at 1/4 resolution (down_factor=4).

Note: Fisheye projection is assumed equiangular (r = f·θ). For other
lens types, angular mapping should be adjusted via the calibration
polynomial of the specific lens/camera combination.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from PIL import Image

try:
    from skimage import measure as _skm
    _BLOCK_REDUCE = _skm.block_reduce
except ImportError:
    _skm = None
    _BLOCK_REDUCE = None


IMG_EXTS = (".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG",
            ".tif", ".tiff", ".TIF", ".TIFF")

# ── LiCOR LAI-2200C ring definition ──────────────────────────────────────────
LICOR_RINGS    = [(0, 13, 7), (13, 30, 23), (30, 46, 38), (46, 61, 53), (61, 74, 68)]
_RING_CENTRES  = np.array([r[2] for r in LICOR_RINGS], dtype=float)
_WEIGHTS       = np.array([0.041, 0.131, 0.201, 0.290, 0.337])
_COS_THETA     = np.cos(np.radians(_RING_CENTRES))
_AZ_BIN        = 10           # azimuth cell width in degrees
_N_AZ_CELLS    = 360 // _AZ_BIN


def _block_mean(arr: np.ndarray, factor: int) -> np.ndarray:
    """Downsample 2-D array by averaging factor×factor blocks."""
    if _BLOCK_REDUCE is not None:
        return _BLOCK_REDUCE(arr.astype(float), (factor, factor), np.mean)
    # Fallback: reshape + mean (works when dims are exact multiples)
    h, w = arr.shape
    return arr.astype(float)[:h // factor * factor, :w // factor * factor] \
              .reshape(h // factor, factor, w // factor, factor) \
              .mean(axis=(1, 3))


def _build_geometry(img_size: int, centre: tuple[float, float],
                    r_max: float, down_factor: int):
    """Pre-compute zenith and azimuth arrays at downsampled resolution."""
    ds      = img_size // down_factor
    ds_cen  = (centre[0] / down_factor, centre[1] / down_factor)
    ds_rmax = r_max / down_factor
    yy, xx  = np.mgrid[0:ds, 0:ds].astype(float)
    dist    = np.sqrt((xx - ds_cen[1]) ** 2 + (yy - ds_cen[0]) ** 2)
    zenith  = (dist / ds_rmax) * 90.0
    dy = ds_cen[0] - yy
    dx = xx - ds_cen[1]
    azimuth     = np.degrees(np.arctan2(dx, dy)) % 360.0
    fisheye_ok  = zenith <= 90.0
    return zenith, azimuth, fisheye_ok


def _precompute_ring_az(zenith, azimuth, fisheye_ok):
    """Return list[list[bool array]]: ring_az_idx[ring][az_cell]."""
    idx = []
    for lo, hi, _ in LICOR_RINGS:
        az_cells = []
        for j in range(_N_AZ_CELLS):
            az_lo, az_hi = j * _AZ_BIN, (j + 1) * _AZ_BIN
            mask = ((zenith >= lo) & (zenith < hi) &
                    (azimuth >= az_lo) & (azimuth < az_hi) &
                    fisheye_ok)
            az_cells.append(mask)
        idx.append(az_cells)
    return idx


def compute_lai(mask_uint8: np.ndarray,
                zenith, azimuth, fisheye_ok,
                ring_az_idx, down_factor: int) -> tuple[float, float]:
    """
    Compute LAI and mean gap fraction (0–60°) from a binary mask.

    Args:
        mask_uint8:  H×W uint8 array (255 = sky, 0 = vegetation).
        zenith, azimuth, fisheye_ok, ring_az_idx: from _build_geometry / _precompute_ring_az.
        down_factor: spatial averaging factor.

    Returns:
        (LAI, gap_fraction_0_60)
    """
    binary  = (mask_uint8 > 127).astype(float)
    mask_ds = _block_mean(binary, down_factor)

    ring_gf = np.zeros(len(LICOR_RINGS))
    for k, az_cells in enumerate(ring_az_idx):
        cell_gfs = [float(mask_ds[idx].mean())
                    for idx in az_cells if idx.sum() > 0]
        ring_gf[k] = np.nanmean(cell_gfs) if cell_gfs else np.nan

    gf_0_60 = float(mask_ds[(zenith < 60) & fisheye_ok].mean())

    with np.errstate(divide="ignore", invalid="ignore"):
        ln_gf = np.where(ring_gf > 0, np.log(ring_gf), np.nan)

    lai = 2.0 * float(np.nansum(-ln_gf * _COS_THETA * _WEIGHTS))
    return lai, gf_0_60


def collect_masks(masks_path: str) -> list[str]:
    if os.path.isfile(masks_path):
        return [masks_path]
    paths = []
    for f in sorted(os.listdir(masks_path)):
        if any(f.endswith(e) for e in IMG_EXTS):
            paths.append(os.path.join(masks_path, f))
    return paths


def main():
    parser = argparse.ArgumentParser(
        description="Estimate LAI from binary sky/vegetation masks.")
    parser.add_argument("--masks",       required=True,
                        help="Mask file or folder of binary PNG masks "
                             "(255=sky, 0=vegetation)")
    parser.add_argument("--output",      required=True,
                        help="Output CSV file path")
    parser.add_argument("--img_size",    type=int, default=4096,
                        help="Expected square image size in pixels (default: 4096)")
    parser.add_argument("--centre",      type=float, nargs=2,
                        metavar=("ROW", "COL"), default=None,
                        help="Optical centre in pixels (row col). "
                             "Default: image centre (img_size/2, img_size/2)")
    parser.add_argument("--radius",      type=float, default=None,
                        help="Fisheye circle radius in pixels. "
                             "Default: img_size/2")
    parser.add_argument("--down_factor", type=int, default=4,
                        help="Spatial averaging factor before LAI integration "
                             "(default: 4, matching 1024×1024 effective resolution)")
    args = parser.parse_args()

    img_size    = args.img_size
    centre      = tuple(args.centre) if args.centre else (img_size / 2, img_size / 2)
    r_max       = args.radius if args.radius else img_size / 2
    down_factor = args.down_factor

    print(f"Image size  : {img_size}×{img_size}")
    print(f"Centre      : row={centre[0]:.1f}, col={centre[1]:.1f}")
    print(f"Radius      : {r_max:.1f} px")
    print(f"Down factor : {down_factor}×  →  effective {img_size//down_factor}×{img_size//down_factor}")

    zenith, azimuth, fisheye_ok = _build_geometry(img_size, centre, r_max, down_factor)
    ring_az_idx = _precompute_ring_az(zenith, azimuth, fisheye_ok)

    mask_paths = collect_masks(args.masks)
    if not mask_paths:
        print(f"ERROR: no mask images found in {args.masks}")
        sys.exit(1)
    print(f"\nProcessing {len(mask_paths)} mask(s)…")

    rows = []
    for i, path in enumerate(mask_paths, 1):
        stem = os.path.splitext(os.path.basename(path))[0]
        try:
            mask = np.array(Image.open(path).convert("L"))
            if mask.shape[0] != img_size or mask.shape[1] != img_size:
                # Resize to expected size (nearest-neighbour to preserve binary values)
                mask = np.array(
                    Image.fromarray(mask).resize((img_size, img_size), Image.NEAREST))
            lai, gf = compute_lai(mask, zenith, azimuth, fisheye_ok,
                                  ring_az_idx, down_factor)
        except Exception as e:
            print(f"  [{i:4d}/{len(mask_paths)}] {stem}  ERROR: {e}")
            rows.append(dict(filename=stem, LAI=float("nan"), gap_fraction=float("nan")))
            continue

        rows.append(dict(
            filename     = stem,
            LAI          = round(float(lai), 4) if np.isfinite(lai) else float("nan"),
            gap_fraction = round(gf, 4),
        ))
        if i % 50 == 0 or i == len(mask_paths):
            print(f"  [{i:4d}/{len(mask_paths)}]  last: {stem}  LAI={lai:.3f}")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df.to_csv(args.output, index=False)
    n_valid = df["LAI"].notna().sum()
    print(f"\nSaved {n_valid}/{len(df)} valid results → {args.output}")
    print(f"LAI range: {df['LAI'].min():.3f} – {df['LAI'].max():.3f}  "
          f"(mean {df['LAI'].mean():.3f})")


if __name__ == "__main__":
    main()
