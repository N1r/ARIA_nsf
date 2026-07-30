"""
Excitation Flow Matching with a glottal informed prior.

Instead of standard flow matching from N(0, I) noise, the SOURCE distribution is the
deterministic glottal-table excitation x0 (driven by F0 / R_d), and the TARGET x1 is the
true excitation obtained by inverse-filtering the real waveform through the all-pole
vocal-tract filter A(z):   x1 = fir_filt(y_real, [1, a1, a2, ...]).

This is a stochastic-interpolant / bridge formulation. The probability path
    x_t = (1 - t) * x0 + t * x1
has a SHORT displacement  (x1 - x0) = the residual the glottal model misses (aspiration,
jitter, non-periodic energy, spectral-tilt error). A velocity network v_theta(x_t, t|cond)
regresses that displacement; inference integrates dx/dt = v_theta from x0 (glottal) to the
refined excitation, which then passes through the SAME analytic vocal-tract filter, so
formant controllability is fully preserved.

Key consequence: because periodicity + phase already live in x0, the network only learns
the *stochastic residual* — this is what makes a TIME-DOMAIN flow tractable here (vs. the
usual mel-domain flow that needs a separate vocoder and loses source-filter control).

cond = frame-rate encoder features (e.g. F0, voicing, formant ctrl), upsampled to the
sample-rate length of x_t before being passed in as (B, cond_dim, T).
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0):
    """Sinusoidal embedding of a (B,) timestep in [0, 1] -> (B, dim)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device, dtype=t.dtype) / half
    )
    args = t[:, None] * freqs[None] * 1000.0  # scale [0,1] -> wider phase range
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class FiLM(nn.Module):
    """Feature-wise linear modulation: conditions feature map on (cond + t) signal."""
    def __init__(self, cond_ch: int, feat_ch: int):
        super().__init__()
        self.proj = nn.Conv1d(cond_ch, feat_ch * 2, 1)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        scale, shift = self.proj(c).chunk(2, dim=1)
        return x * (1 + scale) + shift


class GatedConvBlock(nn.Module):
    """Dilated gated 1D conv with FiLM conditioning and residual (WaveNet-style)."""
    def __init__(self, channels: int, cond_ch: int, kernel: int = 5, dilation: int = 1):
        super().__init__()
        self.film = FiLM(cond_ch, channels)
        pad = (kernel - 1) // 2 * dilation
        self.conv = nn.Conv1d(channels, channels * 2, kernel, padding=pad, dilation=dilation)
        self.out = nn.Conv1d(channels, channels, 1)

    def forward(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x = self.film(h, c)
        a, b = self.conv(x).chunk(2, dim=1)
        x = torch.tanh(a) * torch.sigmoid(b)
        return h + self.out(x)


class VelocityNet(nn.Module):
    """v_theta(x_t, t | cond): time-domain velocity field over the excitation.

    Lightweight on purpose — it only models the glottal->true residual, not the whole
    signal. Inputs: x_t (B, T), t (B,), cond_up (B, cond_dim, T). Output: v (B, T).
    """
    def __init__(self, cond_dim: int, channels: int = 64, n_layers: int = 8,
                 kernel: int = 5, t_dim: int = 64,
                 dilations: tuple = (1, 2, 4, 8, 1, 2, 4, 8)):
        super().__init__()
        self.t_dim = t_dim
        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, channels), nn.SiLU(), nn.Linear(channels, channels))
        self.in_conv = nn.Conv1d(1, channels, kernel, padding=kernel // 2)
        self.cond_proj = nn.Conv1d(cond_dim, channels, 1)
        dils = list(dilations)
        if len(dils) < n_layers:
            dils = (dils * ((n_layers // len(dils)) + 1))[:n_layers]
        self.blocks = nn.ModuleList([
            GatedConvBlock(channels, channels, kernel, dils[i]) for i in range(n_layers)])
        self.out_conv = nn.Conv1d(channels, 1, kernel, padding=kernel // 2)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond_up: torch.Tensor):
        T = x_t.shape[1]
        h = self.in_conv(x_t.unsqueeze(1))                     # (B, C, T)
        c = self.cond_proj(cond_up[..., :T])                   # (B, C, T)
        temb = self.t_mlp(timestep_embedding(t, self.t_dim))   # (B, C)
        c = c + temb[..., None]                                # broadcast over time
        for blk in self.blocks:
            h = blk(h, c)
        return self.out_conv(h).squeeze(1)                     # (B, T)


class ExcitationFlow(nn.Module):
    """Glottal-prior flow-matching head over the excitation signal.

    flow_loss(x0, x1, cond_up): rectified-flow / OT-CFM regression of (x1 - x0).
    sample(x0, cond_up, steps): Euler ODE from the glottal prior to refined excitation.
    """
    def __init__(self, cond_dim: int, channels: int = 64, n_layers: int = 8,
                 kernel: int = 5, sigma: float = 0.0):
        super().__init__()
        self.velocity = VelocityNet(cond_dim, channels, n_layers, kernel)
        self.sigma = sigma  # optional interpolation noise (0 = straight bridge)

    def flow_loss(self, x0: torch.Tensor, x1: torch.Tensor, cond_up: torch.Tensor):
        B = x0.shape[0]
        T = min(x0.shape[1], x1.shape[1], cond_up.shape[-1])
        x0, x1, cond_up = x0[:, :T], x1[:, :T], cond_up[..., :T]
        t = torch.rand(B, device=x0.device, dtype=x0.dtype)
        x_t = (1 - t)[:, None] * x0 + t[:, None] * x1
        if self.sigma > 0:
            x_t = x_t + self.sigma * torch.randn_like(x_t)
        v_target = x1 - x0
        v_pred = self.velocity(x_t, t, cond_up)
        return F.mse_loss(v_pred, v_target)

    def sample(self, x0: torch.Tensor, cond_up: torch.Tensor, steps: int = 4):
        """Euler ODE from the glottal prior to the refined excitation. Differentiable so
        an end-to-end spectral loss can back-prop through the integration; callers that
        only need inference (validation / eval) should wrap this in torch.no_grad()."""
        x = x0
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((x.shape[0],), i * dt, device=x.device, dtype=x.dtype)
            x = x + dt * self.velocity(x, t, cond_up)
        return x


if __name__ == "__main__":
    # smoke unit test: loss backward + sampling shape/finiteness
    torch.manual_seed(0)
    B, T, C = 4, 8000, 6
    flow = ExcitationFlow(cond_dim=C, channels=32, n_layers=6)
    x0 = torch.randn(B, T)
    x1 = x0 + 0.1 * torch.randn(B, T)          # target = glottal + small residual
    cond = torch.randn(B, C, T)
    loss = flow.flow_loss(x0, x1, cond)
    loss.backward()
    gnorm = sum(p.grad.abs().sum() for p in flow.parameters() if p.grad is not None)
    with torch.no_grad():
        y = flow.sample(x0, cond, steps=4)
    n_params = sum(p.numel() for p in flow.parameters())
    print(f"loss={loss.item():.4f}  grad_sum={gnorm.item():.3f}  "
          f"sample_shape={tuple(y.shape)}  finite={torch.isfinite(y).all().item()}  "
          f"params={n_params/1e6:.2f}M")
    assert torch.isfinite(loss) and torch.isfinite(y).all()
    print("OK")
