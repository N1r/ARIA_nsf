"""
Mel-domain flow enhancer — the 2026-SOTA-style path (cf. FlowSE / DiT-Flow).

Instead of flowing in the time domain (waveform/excitation, which over-smooths because of
harmonic-phase sensitivity), the flow operates on MEL-SPECTROGRAMS:

  source  M0 = mel of aria_golf's deterministic (controllable but "coarse") synthesis
  target  M1 = mel of the real recording
  cond    = M0  (the coarse mel — locks in formants/content, like FlowSE conditions on noisy)
  path    M_t = (1-t) M0 + t M1 ;  velocity target = M1 - M0
  infer   M0 --ODE--> M_enhanced --(neural vocoder)--> waveform

Because the source is the aria_golf mel (not Gaussian noise) the probability path is short
(few ODE steps). Mel magnitude is phase-insensitive, so this dodges the time-domain trap.
Controllability stays in aria_golf (analytic filter); the flow only enhances quality and is
conditioned on the coarse mel so it does not move the formants.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.flow import GatedConvBlock, timestep_embedding


class MelVelocityNet(nn.Module):
    """v_theta(M_t, t | M_cond) over a mel-spectrogram (B, n_mels, T). 1D conv over time,
    mel bins as channels; FiLM conditioning on the coarse mel + timestep."""
    def __init__(self, n_mels: int = 100, channels: int = 256, n_layers: int = 8,
                 kernel: int = 5, t_dim: int = 64,
                 dilations: tuple = (1, 2, 4, 8, 1, 2, 4, 8)):
        super().__init__()
        self.t_dim = t_dim
        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, channels), nn.SiLU(), nn.Linear(channels, channels))
        self.in_conv = nn.Conv1d(n_mels, channels, kernel, padding=kernel // 2)
        self.cond_conv = nn.Conv1d(n_mels, channels, kernel, padding=kernel // 2)
        dils = list(dilations)
        if len(dils) < n_layers:
            dils = (dils * ((n_layers // len(dils)) + 1))[:n_layers]
        self.blocks = nn.ModuleList(
            [GatedConvBlock(channels, channels, kernel, dils[i]) for i in range(n_layers)])
        self.out_conv = nn.Conv1d(channels, n_mels, kernel, padding=kernel // 2)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, M_t, t, M_cond):
        h = self.in_conv(M_t)
        c = self.cond_conv(M_cond) + self.t_mlp(timestep_embedding(t, self.t_dim))[..., None]
        for blk in self.blocks:
            h = blk(h, c)
        return self.out_conv(h)


class MelFlowEnhancer(nn.Module):
    """Flow matching from the coarse aria_golf mel (M0) to the real mel (M1), conditioned
    on M0. flow_loss = OT-CFM velocity regression; sample = Euler ODE."""
    def __init__(self, n_mels: int = 100, channels: int = 256, n_layers: int = 8,
                 kernel: int = 5):
        super().__init__()
        self.n_mels = n_mels
        self.velocity = MelVelocityNet(n_mels, channels, n_layers, kernel)

    def flow_loss(self, M0, M1, cond):
        T = min(M0.shape[-1], M1.shape[-1], cond.shape[-1])
        M0, M1, cond = M0[..., :T], M1[..., :T], cond[..., :T]
        t = torch.rand(M0.shape[0], device=M0.device, dtype=M0.dtype)
        M_t = (1 - t)[:, None, None] * M0 + t[:, None, None] * M1
        v_pred = self.velocity(M_t, t, cond)
        return F.mse_loss(v_pred, M1 - M0)

    def sample(self, M0, cond, steps: int = 8):
        M = M0
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((M.shape[0],), i * dt, device=M.device, dtype=M.dtype)
            M = M + dt * self.velocity(M, t, cond)
        return M


if __name__ == "__main__":
    torch.manual_seed(0)
    B, n_mels, T = 4, 100, 200
    enh = MelFlowEnhancer(n_mels=n_mels, channels=128, n_layers=6)
    M0 = torch.randn(B, n_mels, T)
    M1 = M0 + 0.3 * torch.randn(B, n_mels, T)     # clean = coarse + residual
    loss = enh.flow_loss(M0, M1, cond=M0)
    loss.backward()
    g = sum(p.grad.abs().sum() for p in enh.parameters() if p.grad is not None)
    with torch.no_grad():
        M_enh = enh.sample(M0, cond=M0, steps=8)
    improved = F.l1_loss(M_enh, M1) < F.l1_loss(M0, M1)
    print(f"loss={loss.item():.4f} grad={g.item():.2f} sample={tuple(M_enh.shape)} "
          f"finite={torch.isfinite(M_enh).all().item()} "
          f"params={sum(p.numel() for p in enh.parameters())/1e6:.2f}M")
    print("OK")
