"""
Model architectures for DHP sky/vegetation segmentation.

  DHPSegNet     — lightweight U-Net, channels [16,32,64,128,256], ~1.94M params
  AttDHPSegNet  — DHPSegNet + attention gates (Oktay et al. 2018), ~1.98M params
  AttentionUNet — standard Att U-Net, channels [32,64,128,256,512], ~7.89M params
  HabitatNet    — deep encoder-decoder, ~14.42M params (requires >14 GB RAM at 4096)

Recommended: AttDHPSegNet — best accuracy/speed trade-off at full 4096×4096 resolution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Shared building blocks ────────────────────────────────────────────────────

class ConvBnRelu(nn.Module):
    """Two-layer Conv→BN→ReLU block used throughout all architectures."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AttentionGate(nn.Module):
    """Soft attention gate (Oktay et al. 2018)."""
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g  = nn.Sequential(nn.Conv2d(F_g,   F_int, 1, bias=False),
                                   nn.BatchNorm2d(F_int))
        self.W_x  = nn.Sequential(nn.Conv2d(F_l,   F_int, 1, bias=False),
                                   nn.BatchNorm2d(F_int))
        self.psi  = nn.Sequential(nn.Conv2d(F_int, 1,     1, bias=False),
                                   nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = F.interpolate(self.W_g(g), size=x.shape[2:],
                           mode="bilinear", align_corners=True)
        return x * self.psi(self.relu(g1 + self.W_x(x)))


# ── DHPSegNet ─────────────────────────────────────────────────────────────────

class DHPSegNet(nn.Module):
    """
    Lightweight U-Net for DHP sky/vegetation segmentation.

    Channels: [16, 32, 64, 128, 256]  (half-width vs standard U-Net)
    Parameters: ~1.94M
    Inference at 4096×4096: ~3.8 s on Apple M4 (MPS)

    Checkpoint: checkpoints/dhpsegnet_best.pth
    """
    def __init__(self):
        super().__init__()
        c = [16, 32, 64, 128, 256]
        self.pool = nn.MaxPool2d(2)
        self.enc1 = ConvBnRelu(3,    c[0])
        self.enc2 = ConvBnRelu(c[0], c[1])
        self.enc3 = ConvBnRelu(c[1], c[2])
        self.enc4 = ConvBnRelu(c[2], c[3])
        self.bot  = ConvBnRelu(c[3], c[4])
        self.up4  = nn.ConvTranspose2d(c[4], c[3], 2, stride=2)
        self.dec4 = ConvBnRelu(c[4], c[3])
        self.up3  = nn.ConvTranspose2d(c[3], c[2], 2, stride=2)
        self.dec3 = ConvBnRelu(c[3], c[2])
        self.up2  = nn.ConvTranspose2d(c[2], c[1], 2, stride=2)
        self.dec2 = ConvBnRelu(c[2], c[1])
        self.up1  = nn.ConvTranspose2d(c[1], c[0], 2, stride=2)
        self.dec1 = ConvBnRelu(c[1], c[0])
        self.out  = nn.Conv2d(c[0], 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bot(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b),  e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return torch.sigmoid(self.out(d1))


# ── Att-DHPSegNet ─────────────────────────────────────────────────────────────

class AttDHPSegNet(nn.Module):
    """
    DHPSegNet with soft attention gates on all skip connections.

    Channels: [16, 32, 64, 128, 256]  (same backbone as DHPSegNet)
    Parameters: ~1.98M
    Inference at 4096×4096: ~7.4 s on Apple M4 (MPS)

    Checkpoint: checkpoints/light_att_unet_best.pth
    """
    def __init__(self):
        super().__init__()
        c = [16, 32, 64, 128, 256]
        self.pool = nn.MaxPool2d(2)
        self.enc1 = ConvBnRelu(3,    c[0])
        self.enc2 = ConvBnRelu(c[0], c[1])
        self.enc3 = ConvBnRelu(c[1], c[2])
        self.enc4 = ConvBnRelu(c[2], c[3])
        self.bot  = ConvBnRelu(c[3], c[4])
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


# ── Attention U-Net (standard, larger) ───────────────────────────────────────

class AttentionUNet(nn.Module):
    """
    Standard Attention U-Net (Oktay et al. 2018).

    Channels: [32, 64, 128, 256, 512]
    Parameters: ~7.89M
    Inference at 4096×4096: ~52 s on Apple M4 (MPS, ~10–14 GB peak)

    Checkpoint: checkpoints/att_unet_best.pth
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super().__init__()
        c = [32, 64, 128, 256, 512]
        self.pool = nn.MaxPool2d(2)
        self.enc1 = ConvBnRelu(in_channels, c[0])
        self.enc2 = ConvBnRelu(c[0], c[1])
        self.enc3 = ConvBnRelu(c[1], c[2])
        self.enc4 = ConvBnRelu(c[2], c[3])
        self.bottleneck = ConvBnRelu(c[3], c[4])
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
        self.out  = nn.Conv2d(c[0], out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b),  self.att4(b,  e4)], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), self.att3(d4, e3)], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), self.att2(d3, e2)], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), self.att1(d2, e1)], 1))
        return torch.sigmoid(self.out(d1))


# ── HabitatNet ────────────────────────────────────────────────────────────────

def _conv_relu(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=True),
                         nn.ReLU(inplace=True))

def _conv_relu_bn(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=True),
                         nn.ReLU(inplace=True),
                         nn.BatchNorm2d(out_ch, eps=1e-3))


class HabitatNet(nn.Module):
    """
    Deep encoder-decoder for binary habitat segmentation.

    Parameters: ~14.42M
    NOTE: Requires >14 GB unified memory at 4096×4096. OOM on 16 GB Apple M4.
          Use tiled inference or reduce resolution for inference on limited hardware.

    Checkpoint: checkpoints/habitatnet_ewp_best.pth
    """
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.enc1_c1 = _conv_relu_bn(3,   32);  self.enc1_c2 = _conv_relu_bn(32,  32)
        self.enc1_c3 = _conv_relu_bn(32,  32)
        self.enc2_c1 = _conv_relu(32,    64);   self.enc2_c2 = _conv_relu_bn(64,  64)
        self.enc2_c3 = _conv_relu_bn(64,  64);  self.enc2_c4 = _conv_relu_bn(64,  64)
        self.enc3_c1 = _conv_relu(64,   128);   self.enc3_c2 = _conv_relu_bn(128, 128)
        self.enc3_c3 = _conv_relu_bn(128, 128); self.enc3_c4 = _conv_relu_bn(128, 128)
        self.enc4_c1 = _conv_relu(128,  256);   self.enc4_c2 = _conv_relu_bn(256, 256)
        self.bot_c1  = _conv_relu(256,  512);   self.bot_c2  = _conv_relu_bn(512, 512)
        self.bot_c3  = _conv_relu_bn(512, 512); self.bot_c4  = _conv_relu_bn(512, 512)
        self.up4  = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4_c1 = _conv_relu(512,  256);   self.dec4_c2 = _conv_relu_bn(256, 256)
        self.dec4_c3 = _conv_relu_bn(256, 256); self.dec4_c4 = _conv_relu_bn(256, 256)
        self.up3  = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3_c1 = _conv_relu(256,  128);   self.dec3_c2 = _conv_relu_bn(128, 128)
        self.dec3_c3 = _conv_relu_bn(128, 128); self.dec3_c4 = _conv_relu_bn(128, 128)
        self.up2  = nn.ConvTranspose2d(128, 64,  2, stride=2)
        self.dec2_c1 = _conv_relu(128,  64);    self.dec2_c2 = _conv_relu_bn(64,  64)
        self.dec2_c3 = _conv_relu_bn(64,  64);  self.dec2_c4 = _conv_relu_bn(64,  64)
        self.up1  = nn.ConvTranspose2d(64,  32,  2, stride=2)
        self.dec1_c1 = _conv_relu(64,   32);    self.dec1_c2 = _conv_relu_bn(32,  32)
        self.dec1_c3 = _conv_relu_bn(32,  32)
        self.out  = nn.Conv2d(32, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1_c3(self.enc1_c2(self.enc1_c1(x)))
        e2 = self.enc2_c4(self.enc2_c3(self.enc2_c2(self.enc2_c1(self.pool(e1)))))
        e3 = self.enc3_c4(self.enc3_c3(self.enc3_c2(self.enc3_c1(self.pool(e2)))))
        e4 = self.enc4_c2(self.enc4_c1(self.pool(e3)))
        b  = self.bot_c4(self.bot_c3(self.bot_c2(self.bot_c1(self.pool(e4)))))
        d4 = self.dec4_c4(self.dec4_c3(self.dec4_c2(self.dec4_c1(
                torch.cat([self.up4(b),  e4], 1)))))
        d3 = self.dec3_c4(self.dec3_c3(self.dec3_c2(self.dec3_c1(
                torch.cat([self.up3(d4), e3], 1)))))
        d2 = self.dec2_c4(self.dec2_c3(self.dec2_c2(self.dec2_c1(
                torch.cat([self.up2(d3), e2], 1)))))
        d1 = self.dec1_c3(self.dec1_c2(self.dec1_c1(
                torch.cat([self.up1(d2), e1], 1))))
        return torch.sigmoid(self.out(d1))


# ── Convenience loader ────────────────────────────────────────────────────────

_MODEL_REGISTRY = {
    "dhpsegnet":    DHPSegNet,
    "att-dhpsegnet": AttDHPSegNet,
    "att-unet":     AttentionUNet,
    "habitatnet":   HabitatNet,
}

_DEFAULT_CKPT = {
    "dhpsegnet":     "dhpsegnet_best.pth",
    "att-dhpsegnet": "light_att_unet_best.pth",
    "att-unet":      "att_unet_best.pth",
    "habitatnet":    "habitatnet_ewp_best.pth",
}


def load_model(model_name: str, checkpoint_path: str,
               device: str = "cpu") -> nn.Module:
    """
    Instantiate and load a trained model.

    Args:
        model_name:      One of 'dhpsegnet', 'att-dhpsegnet', 'att-unet', 'habitatnet'.
        checkpoint_path: Path to the .pth checkpoint file.
        device:          PyTorch device string ('cpu', 'cuda', 'mps').

    Returns:
        Model in eval mode on the specified device.
    """
    if model_name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Choose from: {list(_MODEL_REGISTRY.keys())}")
    model = _MODEL_REGISTRY[model_name]()
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
