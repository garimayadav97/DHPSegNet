# DHPSegNet

**Automated deep learning workflow for digital hemispherical photography (DHP) segmentation and LAI estimation.**

This repository accompanies the manuscript:

> Yadav, G. *et al.* (2026). Lightweight deep learning models for full-resolution digital hemispherical photograph segmentation and LAI estimation in fine-needleleaf forest canopies. *[Journal]*.

---

## Overview

DHPSegNet provides an end-to-end pipeline for processing fisheye DHP images:

1. **Segment** — sky vs. vegetation binary mask at full 4096×4096 resolution
2. **Compute LAI** — Miller (1967) integration over LiCOR LAI-2200C ring geometry
3. **Annotate** — interactive GUI tool for creating and correcting manual masks

---

## Installation

```bash
git clone https://github.com/garimayadav97/DHPSegNet.git
cd DHPSegNet
pip install -r requirements.txt
```

Download model weights from the [Releases page](https://github.com/garimayadav97/DHPSegNet/releases) and place `.pth` files in `checkpoints/`. See [checkpoints/README.md](checkpoints/README.md) for details.

---

## Quick start

### 1 — Segment images

```bash
python segment.py --input ./images/ --output ./masks/
```

This runs **Att-DHPSegNet** (recommended, 1.98M params, ~7.4 s/image on Apple M4) and saves binary PNG masks (255 = sky, 0 = vegetation).

```bash
# Save colour overlay for visual QC
python segment.py --input ./images/ --output ./masks/ --overlay

# Use lighter/faster model (3.8 s/image, slightly lower accuracy)
python segment.py --input ./images/ --output ./masks/ --model dhpsegnet

# Specify a custom checkpoint
python segment.py --input ./images/ --output ./masks/ --checkpoint path/to/model.pth
```

### 2 — Compute LAI

```bash
python compute_lai.py --masks ./masks/ --output lai_results.csv
```

Default geometry matches a Canon EOS 5D Mark II + EF 8–15mm fisheye at 8mm on a 4096×4096 image. Adjust for other setups:

```bash
python compute_lai.py --masks ./masks/ --output lai_results.csv \
    --img_size 4096 --centre 2048 2048 --radius 2048 --down_factor 4
```

Output CSV columns: `filename`, `LAI` (m²/m²), `gap_fraction` (0–60° zenith).

### 3 — Manual annotation tool

```bash
python annotation_tool/manual_segment.py --input ./images/ --output ./masks/
```

The GUI supports:

| Action | How |
|--------|-----|
| Sky paint mode | `S` key |
| Vegetation paint mode | `V` key |
| Box selection | `B` key + drag |
| Flood fill | Click |
| Brush paint | Drag |
| Undo | `Ctrl+Z` |
| Save + next | `Enter` |
| Zoom / pan | Scroll / `Space+drag` |

Auto-threshold initialisation (Otsu / ISODATA / Li / Manual) is available on load. Drafts are auto-saved every 2 s.

---

## Models

| Model | Params | Inference @ 4096 | Checkpoint |
|-------|--------|------------------|------------|
| **Att-DHPSegNet** (recommended) | 1.98M | 7.4 s (M4 MPS) | `att_dhpsegnet_best.pth` |
| DHPSegNet | 1.94M | 3.8 s (M4 MPS) | `dhpsegnet_best.pth` |

All models use the same fully convolutional inference — no resizing required. See [checkpoints/README.md](checkpoints/README.md) for download links.

---

## Repository structure

```
DHPSegNet/
├── segment.py              ← CLI: segment images → binary masks
├── compute_lai.py          ← CLI: compute LAI from masks
├── models/
│   └── dhpsegnet.py        ← all model architectures (single source of truth)
├── annotation_tool/
│   └── manual_segment.py   ← interactive GUI annotation tool
├── checkpoints/
│   └── README.md           ← weight download links + hardware requirements
├── example/
│   └── quickstart.ipynb    ← end-to-end example notebook
└── requirements.txt
```

---

## LAI formula

LAI is estimated using the Miller (1967) logarithmic integration:

```
LAI = 2 · Σ_k [ −ln(T_k) · cos(θ_k) · w_k ]
```

where *T_k* is the mean gap fraction in ring *k*, *θ_k* is the ring centre zenith angle, and *w_k* is the integration weight.

Ring geometry matches the LiCOR LAI-2200C sensor (5 rings, 0–74°). Gap fraction per ring is computed as the mean over 36 azimuth cells of 10° each at 1/4 spatial resolution.

> **Lens projection note:** The current implementation assumes an equiangular fisheye projection (*r = f·θ*). For equisolid-angle lenses (e.g. Canon EF 8–15mm at 8mm), a small systematic bias in high-zenith rings may result. Lens-specific calibration polynomials can be substituted in `compute_lai.py`.

---

## Citation

```bibtex
@article{yadav2026dhpsegnet,
  title   = {Lightweight deep learning models for full-resolution digital hemispherical
             photograph segmentation and LAI estimation in fine-needleleaf forest canopies},
  author  = {Yadav, Garima and others},
  journal = {[Journal]},
  year    = {2026}
}
```

---

## Licence

Code: MIT  
Model weights: CC BY 4.0 (academic use; cite the paper above)
