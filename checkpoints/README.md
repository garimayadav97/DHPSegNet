# Model Weights

Pre-trained checkpoint files are not stored in this repository due to file size.
Download them from the links below and place them in this folder (`checkpoints/`).

| File | Model | Size | Link |
|------|-------|------|------|
| `light_att_unet_best.pth` | Att-DHPSegNet (**recommended**) | ~8 MB | [Zenodo](https://zenodo.org) |
| `att_unet_best.pth` | Attention U-Net | ~32 MB | [Zenodo](https://zenodo.org) |
| `habitatnet_ewp_best.pth` | HabitatNet | ~57 MB | [Zenodo](https://zenodo.org) |

> **Note:** Zenodo links will be added upon paper acceptance. Contact the authors for early access.

## Quick download (once links are live)

```bash
# Recommended model only
wget -P checkpoints/ <zenodo_url>/light_att_unet_best.pth

# All models
wget -P checkpoints/ <zenodo_url>/att_unet_best.pth
wget -P checkpoints/ <zenodo_url>/habitatnet_ewp_best.pth
```

## Hardware requirements

| Model | Params | RAM @ 4096×4096 | Inference time (Apple M4) |
|-------|--------|-----------------|--------------------------|
| DHPSegNet | 1.94M | ~4–5 GB | 3.8 s |
| **Att-DHPSegNet** | **1.98M** | **~5–7 GB** | **7.4 s** |
| Attention U-Net | 7.89M | ~10–14 GB | 52 s |
| HabitatNet | 14.42M | >14 GB | OOM on 16 GB |
