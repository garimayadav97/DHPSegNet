# Model Weights

Pre-trained checkpoint files are not stored in this repository due to file size.
Download them from the [Releases page](https://github.com/garimayadav97/DHPSegNet/releases).

| File | Model | Size |
|------|-------|------|
| `att_dhpsegnet_best.pth` | Att-DHPSegNet (**recommended**) | ~8 MB |
| `dhpsegnet_best.pth` | DHPSegNet | ~8 MB |

## Download

```bash
# Recommended model
wget -P checkpoints/ https://github.com/garimayadav97/DHPSegNet/releases/download/v1.0/att_dhpsegnet_best.pth

# Lighter/faster model
wget -P checkpoints/ https://github.com/garimayadav97/DHPSegNet/releases/download/v1.0/dhpsegnet_best.pth
```

## Hardware requirements

| Model | Params | RAM @ 4096×4096 | Inference time (Apple M4) |
|-------|--------|-----------------|--------------------------|
| DHPSegNet | 1.94M | ~4–5 GB | 3.8 s |
| **Att-DHPSegNet** | **1.98M** | **~5–7 GB** | **7.4 s** |
