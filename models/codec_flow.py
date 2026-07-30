"""
Codec-latent flow enhancer.

Same OT-CFM flow as mel_flow, but operates in EnCodec's continuous (pre-quantization)
encoder latent space instead of mel spectrograms.

  source  Z0 = encodec_encoder( aria_golf coarse synthesis )  — (B, 128, T_lat)
  target  Z1 = encodec_encoder( real recording )
  cond    = Z0  (locks content/formants like mel-flow)
  path    Z_t = (1-t) Z0 + t Z1
  infer   Z0 --ODE--> Z_enhanced --(encodec decoder)--> waveform

Advantages over mel_flow:
  - Encoder and decoder are co-trained adversarially → decoder is robust to its own latent
  - Latent captures phase structure implicitly (convolutional encoder)
  - No dependency on Vocos, whose mel feature extractor struggles with DDSP-style mels

Architecture is identical to MelFlowEnhancer with n_mels → n_channels.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.flow import GatedConvBlock, timestep_embedding

ENCODEC_DIM = 128    # EnCodec 24kHz continuous encoder output dimension
ENCODEC_FPS = 75     # frames per second (320x downsampling from 24kHz)


class CodecVelocityNet(nn.Module):
    """v_theta(Z_t, t | Z_cond) over codec latents (B, n_channels, T).
    Structurally identical to MelVelocityNet; n_channels replaces n_mels."""
    def __init__(self, n_channels: int = ENCODEC_DIM, inner: int = 256, n_layers: int = 8,
                 kernel: int = 5, t_dim: int = 64,
                 dilations: tuple = (1, 2, 4, 8, 1, 2, 4, 8)):
        super().__init__()
        self.t_dim = t_dim
        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, inner), nn.SiLU(), nn.Linear(inner, inner))
        self.in_conv  = nn.Conv1d(n_channels, inner, kernel, padding=kernel // 2)
        self.cond_conv = nn.Conv1d(n_channels, inner, kernel, padding=kernel // 2)
        dils = list(dilations)
        if len(dils) < n_layers:
            dils = (dils * ((n_layers // len(dils)) + 1))[:n_layers]
        self.blocks = nn.ModuleList(
            [GatedConvBlock(inner, inner, kernel, dils[i]) for i in range(n_layers)])
        self.out_conv = nn.Conv1d(inner, n_channels, kernel, padding=kernel // 2)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, Z_t, t, Z_cond):
        h = self.in_conv(Z_t)
        c = self.cond_conv(Z_cond) + self.t_mlp(timestep_embedding(t, self.t_dim))[..., None]
        for blk in self.blocks:
            h = blk(h, c)
        return self.out_conv(h)


class CodecFlowEnhancer(nn.Module):
    """Flow matching from coarse aria_golf codec latent (Z0) to real codec latent (Z1)."""
    def __init__(self, n_channels: int = ENCODEC_DIM, inner: int = 256, n_layers: int = 8,
                 kernel: int = 5):
        super().__init__()
        self.n_channels = n_channels
        self.velocity = CodecVelocityNet(n_channels, inner, n_layers, kernel)

    def flow_loss(self, Z0, Z1, cond):
        T = min(Z0.shape[-1], Z1.shape[-1], cond.shape[-1])
        Z0, Z1, cond = Z0[..., :T], Z1[..., :T], cond[..., :T]
        t = torch.rand(Z0.shape[0], device=Z0.device, dtype=Z0.dtype)
        Z_t = (1 - t)[:, None, None] * Z0 + t[:, None, None] * Z1
        v_pred = self.velocity(Z_t, t, cond)
        return F.mse_loss(v_pred, Z1 - Z0)

    def sample(self, Z0, cond, steps: int = 8):
        Z = Z0
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((Z.shape[0],), i * dt, device=Z.device, dtype=Z.dtype)
            Z = Z + dt * self.velocity(Z, t, cond)
        return Z


if __name__ == "__main__":
    torch.manual_seed(0)
    B, D, T = 2, ENCODEC_DIM, 150   # 2s of audio at 75 fps
    enh = CodecFlowEnhancer(n_channels=D, inner=128, n_layers=6)
    Z0 = torch.randn(B, D, T)
    Z1 = Z0 + 0.3 * torch.randn(B, D, T)
    loss = enh.flow_loss(Z0, Z1, cond=Z0)
    loss.backward()
    g = sum(p.grad.abs().sum() for p in enh.parameters() if p.grad is not None)
    with torch.no_grad():
        Z_enh = enh.sample(Z0, cond=Z0, steps=8)
    print(f"loss={loss.item():.4f}  grad={g.item():.2f}  sample={tuple(Z_enh.shape)}")
    print(f"params={sum(p.numel() for p in enh.parameters())/1e6:.2f}M")
    print("OK")
