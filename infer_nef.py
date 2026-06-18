# ─────────────────────────────────────────────────────────────────────────────
#  infer_nef.py
#
#  Run Att-DHPSegNet inference on NEF (Nikon RAW) files.
#
#  Two modes:
#    Default  : rectangular image at original resolution via tiled inference
#               (no crop, no resize, no fisheye mask)
#    --fisheye: legacy fisheye mode — centre-crops to square, resizes to 4096,
#               applies circular mask  (matches Canon training data)
#
#  Steps (default mode):
#    1. Read NEF → uint8 RGB at original resolution
#    2. Pad to nearest multiple of tile_size
#    3. Run model on overlapping tiles, stitch with soft blending
#    4. Unpad → binary mask PNG  (255 = sky, 0 = vegetation)
#    5. Optional colour overlay for visual QC
#
#  Install dependency first:
#    pip install rawpy
#
#  Usage:
#    # Rectangular (whole-frame) mode — your 7360×4912 images
#    python infer_nef.py --input /path/to/nef_folder --output ./masks_nef
#    python infer_nef.py --input /path/to/nef_folder --output ./masks_nef --overlay --save_rgb
#
#    # Fisheye mode (original Canon pipeline)
#    python infer_nef.py --input /path/to/nef_folder --output ./masks_nef --fisheye
#
#    # Single file
#    python infer_nef.py --input single_image.NEF --output ./masks_nef
#
#  Tile size:
#    Default tile_size=1024 with overlap=128.
#    7360×4912 → ~40 tiles of 1024×1024 (comfortable on MPS/CPU).
#    Increase --tile_size 2048 if you want fewer, larger tiles (needs more RAM).
# ─────────────────────────────────────────────────────────────────────────────

import os, sys, argparse, time
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

try:
    import rawpy
except ImportError:
    print("ERROR: rawpy is not installed.  Run:  pip install rawpy")
    sys.exit(1)

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = "mps" if torch.backends.mps.is_available() else \
         "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

_ROOT    = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(_ROOT, "checkpoints")

# ── Model architecture  (must match training exactly) ─────────────────────────

class ConvBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
    def forward(self, x): return self.block(x)


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g  = nn.Sequential(nn.Conv2d(F_g,    F_int, 1, bias=False),
                                   nn.BatchNorm2d(F_int))
        self.W_x  = nn.Sequential(nn.Conv2d(F_l,    F_int, 1, bias=False),
                                   nn.BatchNorm2d(F_int))
        self.psi  = nn.Sequential(nn.Conv2d(F_int, 1,     1, bias=False),
                                   nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)
    def forward(self, g, x):
        g1 = F.interpolate(self.W_g(g), size=x.shape[2:],
                           mode='bilinear', align_corners=True)
        return x * self.psi(self.relu(g1 + self.W_x(x)))


class AttDHPSegNet(nn.Module):
    """Att-DHPSegNet  —  matches light_att_unet_best.pth
       Channels: [16, 32, 64, 128, 256]  (lightweight half-width variant)
    """
    def __init__(self):
        super().__init__()
        c = [16, 32, 64, 128, 256]          # half-width vs full Att-UNet
        self.pool = nn.MaxPool2d(2)
        self.enc1 = ConvBnRelu(3,    c[0])
        self.enc2 = ConvBnRelu(c[0], c[1])
        self.enc3 = ConvBnRelu(c[1], c[2])
        self.enc4 = ConvBnRelu(c[2], c[3])
        self.bot  = ConvBnRelu(c[3], c[4])  # named 'bot' to match checkpoint
        self.att4 = AttentionGate(c[4], c[3], c[2])
        self.att3 = AttentionGate(c[3], c[2], c[1])
        self.att2 = AttentionGate(c[2], c[1], c[0])
        self.att1 = AttentionGate(c[1], c[0], c[0] // 2)
        self.up4  = nn.ConvTranspose2d(c[4], c[3], 2, stride=2)
        self.dec4 = ConvBnRelu(c[4], c[3])
        self.up3  = nn.ConvTranspose2d(c[3], c[2], 2, stride=2)
        self.dec3 = ConvBnRelu(c[3], c[2])
        self.up2  = nn.ConvTranspose2d(c[2], c[1], 2, stride=2)
        self.dec2 = ConvBnRelu(c[2], c[1])
        self.up1  = nn.ConvTranspose2d(c[1], c[0], 2, stride=2)
        self.dec1 = ConvBnRelu(c[1], c[0])
        self.out  = nn.Conv2d(c[0], 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bot(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b),  self.att4(b,  e4)], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), self.att3(d4, e3)], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), self.att2(d3, e2)], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), self.att1(d2, e1)], 1))
        return torch.sigmoid(self.out(d1))


# ── NEF reading ───────────────────────────────────────────────────────────────

def read_nef_raw(path: str) -> np.ndarray:
    """
    Demosaic a NEF file → uint8 RGB at native camera resolution.

    rawpy settings mimic in-camera JPEG rendering:
      use_camera_wb=True   : camera white balance
      no_auto_bright=False : allow auto brightness
      output_color=sRGB    : sRGB colour space
      gamma=(2.222, 4.5)   : sRGB gamma curve
      output_bps=8         : 8-bit output
    """
    with rawpy.imread(path) as raw:
        rgb = raw.postprocess(
            use_camera_wb   = True,
            no_auto_bright  = False,
            output_color    = rawpy.ColorSpace.sRGB,
            gamma           = (2.222, 4.5),
            output_bps      = 8,
        )
    return rgb   # uint8  H × W × 3


def read_nef_fisheye(path: str, crop_size: int = 4096) -> np.ndarray:
    """
    Read NEF and return a centre-cropped, resized crop_size × crop_size image.
    Use for fisheye images that were used during training (Canon EOS 5D Mark II).
    """
    rgb  = read_nef_raw(path)
    h, w = rgb.shape[:2]
    side = min(h, w)
    top  = (h - side) // 2
    left = (w - side) // 2
    rgb  = rgb[top:top+side, left:left+side]
    if rgb.shape[0] != crop_size:
        rgb = np.array(Image.fromarray(rgb).resize((crop_size, crop_size), Image.LANCZOS))
    return rgb


def make_circular_mask(size: int) -> np.ndarray:
    """True inside the fisheye circle."""
    cx, cy = size // 2, size // 2
    r      = size // 2
    y, x   = np.ogrid[:size, :size]
    return (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2


# ── Tiled inference ───────────────────────────────────────────────────────────

def _run_tile(model, tile_rgb: np.ndarray) -> np.ndarray:
    """Forward pass on a single uint8 RGB tile → float32 prob map [0,1]."""
    arr    = tile_rgb.astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = model(tensor)
    return pred.squeeze().cpu().numpy()   # float32 H × W


def infer_tiled(model, rgb: np.ndarray,
                tile_size: int = 1024,
                overlap: int   = 128) -> np.ndarray:
    """
    Run model on a large rectangular image using overlapping tiles.

    - Pads the image so it is divisible by tile_size.
    - Runs the model on each tile with `overlap` pixels of context on each side.
    - Blends tile predictions using a raised-cosine weight window to suppress
      seam artefacts.
    - Returns binary mask uint8 (255 = sky, 0 = other).
    """
    H, W = rgb.shape[:2]

    # Pad to nearest multiple of tile_size
    pad_h = math.ceil(H / tile_size) * tile_size - H
    pad_w = math.ceil(W / tile_size) * tile_size - W
    if pad_h > 0 or pad_w > 0:
        rgb = np.pad(rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    pH, pW = rgb.shape[:2]

    # Accumulator arrays (float64 for stable blending)
    prob_sum   = np.zeros((pH, pW), dtype=np.float64)
    weight_sum = np.zeros((pH, pW), dtype=np.float64)

    # Raised-cosine weight window for one tile (smooth blend at edges)
    def cosine_window(size):
        ramp = np.hanning(size)
        return np.outer(ramp, ramp).astype(np.float64)

    win = cosine_window(tile_size)

    step = tile_size   # step between tile starts (no gap, overlap is extra context)
    ny   = pH // step
    nx   = pW // step

    total_tiles = ny * nx
    for ty in range(ny):
        for tx in range(nx):
            # Core region of this tile
            y0, x0 = ty * step, tx * step
            y1, x1 = y0 + tile_size, x0 + tile_size

            # Expanded region including overlap context
            ey0 = max(0, y0 - overlap)
            ex0 = max(0, x0 - overlap)
            ey1 = min(pH, y1 + overlap)
            ex1 = min(pW, x1 + overlap)

            tile   = rgb[ey0:ey1, ex0:ex1]
            prob   = _run_tile(model, tile)       # float32 (ey1-ey0) × (ex1-ex0)

            # Extract the core region from the prediction
            cy0 = y0 - ey0
            cx0 = x0 - ex0
            core = prob[cy0 : cy0 + tile_size, cx0 : cx0 + tile_size]

            prob_sum  [y0:y1, x0:x1] += core * win
            weight_sum[y0:y1, x0:x1] += win

            done = ty * nx + tx + 1
            if done % 10 == 0 or done == total_tiles:
                print(f"    tile {done}/{total_tiles}", end="\r", flush=True)

    print()   # newline after \r progress

    # Normalise and threshold
    prob_map = prob_sum / np.maximum(weight_sum, 1e-8)
    mask     = (prob_map[:H, :W] > 0.5).astype(np.uint8) * 255
    return mask


def infer_fisheye(model, rgb: np.ndarray) -> np.ndarray:
    """Single-pass inference for 4096×4096 fisheye images + circular mask."""
    arr    = rgb.astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = model(tensor)
    mask   = (pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255
    circle = make_circular_mask(4096)
    mask[~circle] = 0
    return mask


# ── Overlay helper ─────────────────────────────────────────────────────────────

def save_overlay(rgb: np.ndarray, mask: np.ndarray, path: str):
    """Colour overlay: sky = cyan tint, vegetation = original."""
    overlay = rgb.copy()
    sky = mask == 255
    overlay[sky, 0] = (overlay[sky, 0].astype(int) * 0.4).astype(np.uint8)
    overlay[sky, 1] = np.clip(overlay[sky, 1].astype(int) + 60, 0, 255).astype(np.uint8)
    overlay[sky, 2] = np.clip(overlay[sky, 2].astype(int) + 60, 0, 255).astype(np.uint8)
    Image.fromarray(overlay).save(path)


# ── File collection ────────────────────────────────────────────────────────────

def collect_nef(input_path: str):
    NEF_EXTS = ('.nef', '.NEF', '.nrw', '.NRW')
    if os.path.isfile(input_path):
        return [input_path]
    files = []
    for f in sorted(os.listdir(input_path)):
        if f.endswith(NEF_EXTS):
            files.append(os.path.join(input_path, f))
    return files


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NEF inference with Att-DHPSegNet")
    parser.add_argument("--input",     required=True,
                        help="Path to a NEF file or folder containing NEF files")
    parser.add_argument("--output",    default="./masks_nef",
                        help="Output folder for binary masks (default: ./masks_nef)")
    parser.add_argument("--overlay",   action="store_true",
                        help="Save colour overlay images for visual QC")
    parser.add_argument("--save_rgb",  action="store_true",
                        help="Save intermediate RGB image (after demosaic, before inference)")
    parser.add_argument("--fisheye",   action="store_true",
                        help="Fisheye mode: centre-crop → 4096×4096, apply circular mask "
                             "(use for Canon training-matched images)")
    parser.add_argument("--tile_size", type=int, default=1024,
                        help="Tile size for tiled inference (default: 1024). "
                             "Must be divisible by 16. Increase if seams are visible.")
    parser.add_argument("--overlap",   type=int, default=128,
                        help="Overlap context pixels around each tile (default: 128)")
    parser.add_argument("--ckpt",      default=os.path.join(CKPT_DIR, "light_att_unet_best.pth"),
                        help="Path to checkpoint (default: checkpoints/light_att_unet_best.pth)")
    args = parser.parse_args()

    if args.tile_size % 16 != 0:
        print(f"ERROR: --tile_size must be divisible by 16 (got {args.tile_size})")
        sys.exit(1)

    # ── Load model ────────────────────────────────────────────────────────────
    if not os.path.exists(args.ckpt):
        print(f"ERROR: checkpoint not found at {args.ckpt}")
        sys.exit(1)

    print(f"Loading Att-DHPSegNet from {args.ckpt}")
    model = AttDHPSegNet().to(DEVICE)
    model.load_state_dict(torch.load(args.ckpt, map_location=DEVICE))
    model.eval()
    n = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  {n:.2f}M parameters  |  checkpoint loaded OK")

    mode_str = "fisheye (4096×4096 crop + circular mask)" if args.fisheye \
               else f"tiled rectangular  (tile={args.tile_size}, overlap={args.overlap})"
    print(f"  Mode: {mode_str}")

    # ── Collect files ─────────────────────────────────────────────────────────
    nef_files = collect_nef(args.input)
    if not nef_files:
        print(f"ERROR: no NEF files found in {args.input}")
        sys.exit(1)
    print(f"  {len(nef_files)} NEF file(s) found")

    # ── Output dirs ───────────────────────────────────────────────────────────
    os.makedirs(args.output, exist_ok=True)
    if args.overlay:
        overlay_dir = args.output + "_overlay"
        os.makedirs(overlay_dir, exist_ok=True)
    if args.save_rgb:
        rgb_dir = args.output + "_rgb"
        os.makedirs(rgb_dir, exist_ok=True)

    # ── Process ───────────────────────────────────────────────────────────────
    total_t = 0
    for i, nef_path in enumerate(nef_files, 1):
        stem     = os.path.splitext(os.path.basename(nef_path))[0]
        out_mask = os.path.join(args.output, stem + ".png")

        t0 = time.time()

        # 1. Read NEF → RGB
        try:
            if args.fisheye:
                rgb = read_nef_fisheye(nef_path)
            else:
                rgb = read_nef_raw(nef_path)
        except Exception as e:
            print(f"  [{i:3d}/{len(nef_files)}] {stem}  ERROR reading NEF: {e}")
            continue

        print(f"  [{i:3d}/{len(nef_files)}] {stem}  ({rgb.shape[1]}×{rgb.shape[0]})")

        # 1b. Save intermediate RGB (optional)
        if args.save_rgb:
            Image.fromarray(rgb).save(
                os.path.join(rgb_dir, stem + "_rgb.jpg"), quality=92)
            print(f"    RGB saved → {rgb_dir}/{stem}_rgb.jpg")

        # 2. Inference
        if args.fisheye:
            mask = infer_fisheye(model, rgb)
        else:
            mask = infer_tiled(model, rgb, args.tile_size, args.overlap)

        # 3. Save mask
        Image.fromarray(mask, mode="L").save(out_mask)

        # 4. Optional overlay
        if args.overlay:
            save_overlay(rgb, mask,
                         os.path.join(overlay_dir, stem + "_overlay.jpg"))

        elapsed  = time.time() - t0
        total_t += elapsed
        sky_pct  = (mask == 255).mean() * 100
        eta      = (total_t / i) * (len(nef_files) - i)
        print(f"    sky={sky_pct:5.1f}%  {elapsed:.1f}s  ETA {eta/60:.1f}min")

    print(f"\nDone.  Masks saved to:  {args.output}")
    if args.save_rgb:
        print(f"RGB images saved to:    {rgb_dir}")
    if args.overlay:
        print(f"Overlays saved to:      {overlay_dir}")
    print(f"Total time: {total_t/60:.1f} min  ({total_t/len(nef_files):.1f}s per image)")


if __name__ == "__main__":
    main()
