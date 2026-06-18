"""
segment.py — DHP sky/vegetation segmentation

Segments fisheye DHP images at full resolution using a trained deep-learning
model. Produces binary PNG masks (255 = sky, 0 = vegetation).

Usage
-----
  python segment.py --input ./images/ --output ./masks/
  python segment.py --input ./images/ --output ./masks/ --model att-dhpsegnet
  python segment.py --input img.jpg   --output ./masks/ --checkpoint custom.pth
  python segment.py --input ./images/ --output ./masks/ --overlay

Models
------
  att-dhpsegnet  (default, recommended)  ~1.98M params, ~7.4 s/image @ 4096
  dhpsegnet                              ~1.94M params, ~3.8 s/image @ 4096
  att-unet                               ~7.89M params, ~52 s/image @ 4096
  habitatnet                             ~14.42M params, OOM on 16 GB at 4096

See checkpoints/README.md for download links.
"""

import os
import sys
import argparse
import time
import numpy as np
from PIL import Image

import torch

# Allow running from repo root or from a different working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.dhpsegnet import load_model, _DEFAULT_CKPT

IMG_EXTS = (".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG",
            ".tif", ".tiff", ".TIF", ".TIFF")

_ROOT    = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(_ROOT, "checkpoints")


def _best_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def collect_images(input_path: str) -> list[str]:
    if os.path.isfile(input_path):
        return [input_path]
    paths = []
    for f in sorted(os.listdir(input_path)):
        if any(f.endswith(e) for e in IMG_EXTS):
            paths.append(os.path.join(input_path, f))
    return paths


def segment_image(model: torch.nn.Module, img_path: str,
                  device: str) -> np.ndarray:
    """Run model on a single image. Returns uint8 mask (255=sky, 0=veg)."""
    img = Image.open(img_path).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(tensor)
    return (pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255


def save_overlay(rgb_path: str, mask: np.ndarray, out_path: str):
    """Save colour overlay: sky = cyan tint over original pixels."""
    rgb = np.array(Image.open(rgb_path).convert("RGB"))
    out = rgb.copy()
    sky = mask == 255
    out[sky, 0] = (out[sky, 0].astype(int) * 40 // 100).astype(np.uint8)
    out[sky, 1] = np.clip(out[sky, 1].astype(int) + 60, 0, 255).astype(np.uint8)
    out[sky, 2] = np.clip(out[sky, 2].astype(int) + 60, 0, 255).astype(np.uint8)
    Image.fromarray(out).save(out_path)


def main():
    parser = argparse.ArgumentParser(
        description="Segment DHP fisheye images into sky/vegetation masks.")
    parser.add_argument("--input",  required=True,
                        help="Image file or folder of images")
    parser.add_argument("--output", required=True,
                        help="Output folder for binary mask PNGs")
    parser.add_argument("--model",  default="att-dhpsegnet",
                        choices=["dhpsegnet", "att-dhpsegnet", "att-unet", "habitatnet"],
                        help="Model architecture (default: att-dhpsegnet)")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to .pth checkpoint. "
                             "Defaults to checkpoints/<model_default>.pth")
    parser.add_argument("--device", default=None,
                        help="PyTorch device (cpu/cuda/mps). Auto-detected by default.")
    parser.add_argument("--overlay", action="store_true",
                        help="Save colour overlay images alongside masks")
    parser.add_argument("--skip_existing", action="store_true", default=True,
                        help="Skip images that already have an output mask (default: on)")
    parser.add_argument("--no_skip", action="store_true",
                        help="Re-run inference even if output mask already exists")
    args = parser.parse_args()

    device = args.device or _best_device()
    print(f"Device : {device}")

    ckpt_path = args.checkpoint or os.path.join(CKPT_DIR, _DEFAULT_CKPT[args.model])
    if not os.path.exists(ckpt_path):
        print(f"ERROR: checkpoint not found at {ckpt_path}")
        print("  Download weights from: https://github.com/garimayadav97/DHPSegNet"
              "#model-weights")
        sys.exit(1)

    print(f"Loading {args.model} from {ckpt_path}")
    model = load_model(args.model, ckpt_path, device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  {n_params:.2f}M parameters — checkpoint loaded OK")

    images = collect_images(args.input)
    if not images:
        print(f"ERROR: no images found in {args.input}")
        sys.exit(1)
    print(f"  {len(images)} image(s) found")

    os.makedirs(args.output, exist_ok=True)
    if args.overlay:
        overlay_dir = args.output.rstrip("/") + "_overlay"
        os.makedirs(overlay_dir, exist_ok=True)

    skip = args.skip_existing and not args.no_skip
    total_t = 0
    skipped = 0

    for i, img_path in enumerate(images, 1):
        stem     = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(args.output, stem + ".png")

        if skip and os.path.exists(out_path):
            skipped += 1
            print(f"  [{i:3d}/{len(images)}] {os.path.basename(img_path)} — skip (exists)")
            continue

        t0 = time.time()
        try:
            mask = segment_image(model, img_path, device)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  [{i:3d}/{len(images)}] {os.path.basename(img_path)} — "
                      f"OOM: model requires more memory than available on {device}. "
                      f"Try --device cpu or a smaller model.")
            else:
                print(f"  [{i:3d}/{len(images)}] {os.path.basename(img_path)} — "
                      f"ERROR: {e}")
            continue

        Image.fromarray(mask, mode="L").save(out_path)

        if args.overlay:
            save_overlay(img_path, mask,
                         os.path.join(overlay_dir, stem + "_overlay.jpg"))

        elapsed  = time.time() - t0
        total_t += elapsed
        n_done   = i - skipped
        eta      = (total_t / n_done) * (len(images) - i) if n_done > 0 else 0
        sky_pct  = (mask == 255).mean() * 100
        print(f"  [{i:3d}/{len(images)}] {os.path.basename(img_path):<28s}  "
              f"sky={sky_pct:5.1f}%  {elapsed:.1f}s  ETA {eta/60:.1f}min")

    n_done = len(images) - skipped
    print(f"\nDone.  {n_done} segmented, {skipped} skipped.")
    print(f"Masks saved to: {args.output}")
    if n_done > 0:
        print(f"Mean time: {total_t/n_done:.1f}s per image")
    if args.overlay:
        print(f"Overlays saved to: {overlay_dir}")


if __name__ == "__main__":
    main()
