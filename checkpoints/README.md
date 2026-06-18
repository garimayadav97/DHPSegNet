# Model Weights

Pre-trained checkpoint files are not stored in this repository due to file size.
Download them from the GitHub Release and place them in this folder (`checkpoints/`).

| File | Model | Size |
|------|-------|------|
| `att_dhpsegnet_best.pth` | Att-DHPSegNet (**recommended**) | ~8 MB |
| `att_unet_best.pth` | Attention U-Net | ~32 MB |
| `habitatnet_ewp_best.pth` | HabitatNet | ~57 MB |

## Download

```bash
# Recommended model only
wget -P checkpoints/ https://github.com/garimayadav97/DHPSegNet/releases/download/v1.0/att_dhpsegnet_best.pth

# All models
wget -P checkpoints/ https://github.com/garimayadav97/DHPSegNet/releases/download/v1.0/att_unet_best.pth
wget -P checkpoints/ https://github.com/garimayadav97/DHPSegNet/releases/download/v1.0/habitatnet_ewp_best.pth
```

Or download from the [Releases page](https://github.com/garimayadav97/DHPSegNet/releases).

## Hardware requirements

| Model | Params | RAM @ 4096×4096 | Inference time (Apple M4) |
|-------|--------|-----------------|--------------------------|
| DHPSegNet | 1.94M | ~4–5 GB | 3.8 s |
| **Att-DHPSegNet** | **1.98M** | **~5–7 GB** | **7.4 s** |
| Attention U-Net | 7.89M | ~10–14 GB | 52 s |
| HabitatNet | 14.42M | >14 GB | OOM on 16 GB |
