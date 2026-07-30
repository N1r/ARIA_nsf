"""
Discriminators for adversarial vocoder training.

Provides:
  - MultiScaleSpectrogramDiscriminator  (MSSD, magnitude STFT)
  - MultiPeriodDiscriminator            (MPD, HiFi-GAN style)
  - MultiScaleSTFTDiscriminator         (MS-STFTD, complex STFT real+imag)
  - MultiScaleCQTDiscriminator          (MS-SB-CQTD, sub-band CQT filterbank)
  - LSGAN loss utilities
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def spectrogram_pyramid(
    audio: torch.Tensor,
    scales: tuple[int, ...] | list[int] = (2048, 1024, 512, 256, 128, 64),
    fft_factor: int = 1,
    overlap: float = 0.75,
) -> list[torch.Tensor]:
    specs = []
    for scale in scales:
        win_length = int(scale)
        n_fft = int(fft_factor) * win_length
        hop_length = max(1, int((1.0 - float(overlap)) * win_length))
        window = torch.hann_window(win_length, device=audio.device, dtype=audio.dtype)
        spec = torch.stft(audio, n_fft=n_fft, win_length=win_length,
                          hop_length=hop_length, window=window, return_complex=True)
        specs.append(spec.abs().unsqueeze(1))
    return specs


class _ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2,
                 padding=1, use_leaky_relu=True):
        super().__init__()
        layers = [nn.utils.weight_norm(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))]
        if use_leaky_relu:
            layers.append(nn.LeakyReLU(0.2))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class SpectrogramDiscriminator(nn.Module):
    def __init__(self, in_channels=1, features=(32, 64, 128, 256)):
        super().__init__()
        feats = list(features)
        layers = [_ConvBlock(in_channels, feats[0], stride=2)]
        in_ch = feats[0]
        for f in feats[1:]:
            layers.append(_ConvBlock(in_ch, int(f), stride=2))
            in_ch = int(f)
        layers.append(_ConvBlock(in_ch, 1, stride=2, use_leaky_relu=False))
        self.layers = nn.ModuleList(layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)

    def forward(self, x, return_features=False):
        feat = x
        features = []
        for layer in self.layers:
            feat = layer(feat)
            if return_features:
                features.append(feat)
        if return_features:
            return feat, features
        return feat


class MultiScaleSpectrogramDiscriminator(nn.Module):
    def __init__(self, nscales=6, in_channels=1, features=(32, 64, 128, 256)):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [SpectrogramDiscriminator(in_channels, features) for _ in range(int(nscales))])

    def forward(self, specs, detach=False, return_features=False):
        if len(specs) != len(self.discriminators):
            raise ValueError(f"Expected {len(self.discriminators)} specs, got {len(specs)}")
        outs, features = [], []
        for spec, disc in zip(specs, self.discriminators):
            item = disc(spec.detach() if detach else spec, return_features=return_features)
            if return_features:
                score, fm = item
                outs.append(score); features.append(fm)
            else:
                outs.append(item)
        return (outs, features) if return_features else outs


# ── Multi-Period Discriminator (HiFi-GAN style, time-domain periodicity) ──
# Complements the spectrogram discriminator: reshapes the waveform by period p so
# 1D temporal periodicity (glottal harmonic structure) becomes a 2D pattern the
# conv stack can judge. This is exactly what a source-filter model needs and what
# the pure-magnitude spectrogram MSD was missing.

class PeriodDiscriminator(nn.Module):
    def __init__(self, period, channels=(32, 128, 256, 512), kernel=5, stride=3):
        super().__init__()
        self.period = int(period)
        norm = nn.utils.weight_norm
        ch = [1] + list(channels)
        self.convs = nn.ModuleList([
            norm(nn.Conv2d(ch[i], ch[i + 1], (kernel, 1), (stride, 1),
                           padding=((kernel - 1) // 2, 0)))
            for i in range(len(channels))
        ])
        self.conv_post = norm(nn.Conv2d(channels[-1], 1, (3, 1), 1, padding=(1, 0)))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)

    def forward(self, x, return_features=False):
        # x: (B, T) waveform → (B, 1, T//period, period)
        b, t = x.shape
        if t % self.period != 0:
            pad = self.period - (t % self.period)
            x = F.pad(x, (0, pad), mode="reflect")
            t = t + pad
        x = x.view(b, 1, t // self.period, self.period)
        feats = []
        for c in self.convs:
            x = F.leaky_relu(c(x), 0.1)
            if return_features:
                feats.append(x)
        x = self.conv_post(x)
        if return_features:
            feats.append(x)
            return x, feats
        return x


class MultiPeriodDiscriminator(nn.Module):
    """Bank of PeriodDiscriminators over prime periods. Same (scores[, features])
    interface as MultiScaleSpectrogramDiscriminator but consumes a raw waveform."""
    def __init__(self, periods=(2, 3, 5, 7, 11), channels=(32, 128, 256, 512)):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [PeriodDiscriminator(p, channels) for p in periods])

    def forward(self, x, detach=False, return_features=False):
        x = x.detach() if detach else x
        outs, features = [], []
        for d in self.discriminators:
            item = d(x, return_features=return_features)
            if return_features:
                score, fm = item
                outs.append(score); features.append(fm)
            else:
                outs.append(item)
        return (outs, features) if return_features else outs


# ── Multi-Scale STFT Discriminator (MS-STFTD) ─────────────────────────────────
# Complex STFT at multiple resolutions: splits into (real, imag) → 2-channel 2D
# conv. Catches phase artifacts that magnitude-only discriminators miss.

class _STFTSubDisc(nn.Module):
    def __init__(self, n_fft: int, hop: int, features=(32, 64, 128, 256)):
        super().__init__()
        norm = nn.utils.weight_norm
        ch = [2] + list(features)  # 2 input channels: real + imag
        self.convs = nn.ModuleList([
            norm(nn.Conv2d(ch[i], ch[i+1], kernel_size=(3, 9), stride=(1, 2),
                           padding=(1, 4)))
            for i in range(len(features))
        ])
        self.conv_post = norm(nn.Conv2d(features[-1], 1, kernel_size=(3, 3),
                                        stride=1, padding=(1, 1)))
        self.n_fft = n_fft
        self.hop   = hop
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)

    def forward(self, x, return_features=False):
        # x: (B, T) → STFT complex → cat(real,imag) (B,2,F,T') → conv stack
        win = torch.hann_window(self.n_fft, device=x.device, dtype=x.dtype)
        stft = torch.stft(x, self.n_fft, self.hop, self.n_fft,
                          window=win, return_complex=True)          # (B, F, T')
        spec = torch.stack([stft.real, stft.imag], dim=1)           # (B,2,F,T')
        feats, h = [], spec
        for c in self.convs:
            h = F.leaky_relu(c(h), 0.1)
            if return_features:
                feats.append(h)
        h = self.conv_post(h)
        if return_features:
            feats.append(h)
            return h, feats
        return h


class MultiScaleSTFTDiscriminator(nn.Module):
    """MS-STFTD: complex STFT discriminators at multiple resolutions.

    Default scales follow BigVGAN-v2 / Vocos convention.
    Interface mirrors MultiPeriodDiscriminator (takes raw waveform).
    """
    def __init__(
        self,
        # (n_fft, hop) pairs – keep hop ≈ n_fft//4
        configs: list[tuple[int, int]] = (
            (1024,  256),
            (2048,  512),
            (512,   128),
            (256,    64),
        ),
        features: tuple[int, ...] = (32, 64, 128, 256),
    ):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [_STFTSubDisc(n, h, features) for n, h in configs])

    def forward(self, x, detach=False, return_features=False):
        x = x.detach() if detach else x
        outs, all_feats = [], []
        for d in self.discriminators:
            item = d(x, return_features=return_features)
            if return_features:
                score, fm = item
                outs.append(score); all_feats.append(fm)
            else:
                outs.append(item)
        return (outs, all_feats) if return_features else outs


# ── Multi-Scale Sub-Band CQT Discriminator (MS-SB-CQTD) ──────────────────────
# Pure-PyTorch CQT via a pre-computed filterbank (no nnAudio dependency).
# CQT gives geometrically-spaced bins (equal bins per octave) → better resolution
# at low frequencies for voiced speech harmonics.
# Sub-band split: divide CQT into octave bands, discriminate each band separately.

class _CQTLayer(nn.Module):
    """Differentiable CQT magnitude via pre-computed Hann-windowed filterbank."""
    def __init__(self, sr: int = 24000, n_bins: int = 84, fmin: float = 32.7,
                 bins_per_octave: int = 12, hop_length: int = 256):
        super().__init__()
        Q = bins_per_octave / (2 ** (1.0 / bins_per_octave) - 1)
        kr, ki = [], []
        for k in range(n_bins):
            fk = fmin * (2.0 ** (k / bins_per_octave))
            Nk = max(4, round(Q * sr / fk))
            t   = torch.arange(Nk, dtype=torch.float64) / sr
            win = torch.hann_window(Nk, dtype=torch.float64)
            scale = 2.0 / Nk
            kr.append((win * torch.cos(2 * math.pi * fk * t) * scale).float())
            ki.append((win * torch.sin(2 * math.pi * fk * t) * scale).float())
        max_len = max(v.shape[0] for v in kr)
        def pad(lst):
            return torch.stack([F.pad(v, (0, max_len - v.shape[0])) for v in lst])
        self.register_buffer("kr", pad(kr).unsqueeze(1))   # (n_bins, 1, L)
        self.register_buffer("ki", pad(ki).unsqueeze(1))
        self.hop = hop_length
        self.pad = max_len // 2
        self.n_bins = n_bins

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T) → (B, n_bins, T')
        xp = F.pad(x.unsqueeze(1), (self.pad, self.pad))
        r  = F.conv1d(xp, self.kr, stride=self.hop)
        i  = F.conv1d(xp, self.ki, stride=self.hop)
        return torch.sqrt(r**2 + i**2 + 1e-8)


class _SBCQTSubDisc(nn.Module):
    """2D discriminator for one CQT sub-band (bins × time)."""
    def __init__(self, in_channels=1, features=(32, 64, 128)):
        super().__init__()
        norm = nn.utils.weight_norm
        ch = [in_channels] + list(features)
        self.convs = nn.ModuleList([
            norm(nn.Conv2d(ch[i], ch[i+1], kernel_size=(3, 5),
                           stride=(1, 2), padding=(1, 2)))
            for i in range(len(features))
        ])
        self.conv_post = norm(nn.Conv2d(features[-1], 1, (3, 3), 1, padding=(1, 1)))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)

    def forward(self, x, return_features=False):
        # x: (B, 1, bins_per_band, T')
        feats, h = [], x
        for c in self.convs:
            h = F.leaky_relu(c(h), 0.1)
            if return_features:
                feats.append(h)
        h = self.conv_post(h)
        if return_features:
            feats.append(h)
            return h, feats
        return h


class MultiScaleCQTDiscriminator(nn.Module):
    """MS-SB-CQTD: CQT sub-band discriminators at multiple hop lengths.

    n_bins should be a multiple of n_bands (each band gets n_bins//n_bands CQT bins).
    Default: 84 bins = 7 octaves × 12 bpo → 7 sub-bands.
    Interface mirrors MultiPeriodDiscriminator (takes raw waveform).
    """
    def __init__(
        self,
        sr: int = 24000,
        n_bins: int = 84,
        fmin: float = 32.7,
        bins_per_octave: int = 12,
        hop_lengths: tuple[int, ...] = (256, 512, 1024),
        n_bands: int = 7,
        features: tuple[int, ...] = (32, 64, 128),
    ):
        super().__init__()
        assert n_bins % n_bands == 0, "n_bins must be divisible by n_bands"
        self.n_bands = n_bands
        self.bins_per_band = n_bins // n_bands
        # One CQT layer per hop length
        self.cqt_layers = nn.ModuleList([
            _CQTLayer(sr, n_bins, fmin, bins_per_octave, hop)
            for hop in hop_lengths
        ])
        # One sub-disc per (hop × band)
        total = len(hop_lengths) * n_bands
        self.discriminators = nn.ModuleList(
            [_SBCQTSubDisc(1, features) for _ in range(total)])

    def forward(self, x, detach=False, return_features=False):
        x = x.detach() if detach else x
        outs, all_feats = [], []
        disc_idx = 0
        for cqt in self.cqt_layers:
            mag = cqt(x)                                    # (B, n_bins, T')
            bands = mag.chunk(self.n_bands, dim=1)          # n_bands × (B, bpb, T')
            for band in bands:
                d = self.discriminators[disc_idx]; disc_idx += 1
                item = d(band.unsqueeze(1), return_features=return_features)
                if return_features:
                    score, fm = item
                    outs.append(score); all_feats.append(fm)
                else:
                    outs.append(item)
        return (outs, all_feats) if return_features else outs


def _make_real_label_targets(real_scores, *, real_label_min=1.0, real_label_max=1.0,
                             real_label_mode="batch_scalar"):
    lo, hi = float(real_label_min), float(real_label_max)
    if hi < lo:
        lo, hi = hi, lo
    mode = str(real_label_mode or "batch_scalar").lower()
    if lo == hi:
        return [torch.full_like(r, lo) for r in real_scores]
    if mode == "batch_scalar":
        s = real_scores[0].new_empty(()).uniform_(lo, hi)
        return [s.expand_as(r) for r in real_scores]
    return [torch.empty_like(r).uniform_(lo, hi) for r in real_scores]


def lsgan_discriminator_loss(real_scores, fake_scores, real_label_min=1.0,
                             real_label_max=1.0, real_label_mode="batch_scalar"):
    loss = real_scores[0].new_tensor(0.0)
    targets = _make_real_label_targets(
        real_scores, real_label_min=real_label_min, real_label_max=real_label_max,
        real_label_mode=real_label_mode)
    for real, tgt in zip(real_scores, targets):
        loss = loss + F.mse_loss(real, tgt) / len(real_scores)
    for fake in fake_scores:
        loss = loss + F.mse_loss(fake, torch.zeros_like(fake)) / len(fake_scores)
    return 0.5 * loss


def lsgan_generator_loss(fake_scores):
    loss = fake_scores[0].new_tensor(0.0)
    for fake in fake_scores:
        loss = loss + F.mse_loss(fake, torch.ones_like(fake)) / len(fake_scores)
    return loss


def feature_matching_loss(real_features, fake_features):
    loss = real_features[0][0].new_tensor(0.0)
    count = 0
    for rs, fs in zip(real_features, fake_features):
        for r, f in zip(rs, fs):
            loss = loss + F.l1_loss(f, r.detach())
            count += 1
    return loss / max(count, 1)
