"""
Hybrid Analytic + Learned vocal tract filter (ARIA).

Two modes:
  1. VocalTractCascade          — time-domain cascade (torchlpc); drop-in for LTVMinimumPhaseFilterPrecise
  2. AnalyticHarmonicsOscillator — freq-domain A_k oscillator; drop-in for AdditiveSynthesizer

Both expose alpha (tilt), F1/B1, F2/B2 as explicit analytic params + learned remainder.
"""

import math
import torch
import torch.nn as nn
from torchlpc import sample_wise_lpc
from models.audiotensor import AudioTensor
from models.ctrl import wrap_ctrl_fn, Controllable
from models.filters import LTVFilterInterface, LTVMinimumPhaseFilterPrecise
from models.synth import OscillatorInterface
from models.utils import biquads2lpc, get_logits2biquads, fir_filt


# ─────────────────────────────────────────────
# Shared: convert formant (F, BW) → biquad AR coefs
# ─────────────────────────────────────────────

def _inv_sigmoid(val: float, lo: float, hi: float) -> float:
    """Inverse of sigmoid(x)*(hi-lo)+lo → x.  For writing Hz back into logits."""
    p = (float(val) - lo) / (hi - lo)
    p = min(max(p, 1e-4), 1.0 - 1e-4)
    return math.log(p / (1.0 - p))


def formant_to_cd(f: torch.Tensor, bw: torch.Tensor, sr: int):
    """
    f, bw : (...) in Hz
    Returns c, d : (...) second-order AR coefficients
      H(z) = 1 / (1 + c*z^-1 + d*z^-2)
    """
    r     = torch.exp(-math.pi * bw / sr)
    theta = 2 * math.pi * f / sr
    c     = -2 * r * torch.cos(theta)
    d     = r ** 2
    return c, d


def formant_log_mag_at_harmonics(k_freq: torch.Tensor,
                                  f: torch.Tensor,
                                  bw: torch.Tensor,
                                  sr: int) -> torch.Tensor:
    """
    Log magnitude of one formant at harmonic frequencies.
    k_freq : (B, T, K) harmonic frequencies in Hz
    f, bw  : (B, T, 1) formant frequency and bandwidth
    Returns log|H(k·F0)| : (B, T, K)
    """
    r     = torch.exp(-math.pi * bw / sr)
    theta = 2 * math.pi * f  / sr
    omega = 2 * math.pi * k_freq / sr
    diff  = omega - theta
    log_denom = 0.5 * torch.log(
        (1 - r * torch.cos(diff)).pow(2) +
        (r * torch.sin(diff)).pow(2) + 1e-8
    )
    return -log_denom


def tilt_log_mag_at_harmonics(k_freq: torch.Tensor,
                               alpha: torch.Tensor,
                               sr: int) -> torch.Tensor:
    """
    Log magnitude of 1st-order tilt filter at harmonic frequencies.
    alpha : (B, T, 1) or scalar
    """
    omega = 2 * math.pi * k_freq / sr
    log_denom = 0.5 * torch.log(
        (1 - alpha * torch.cos(omega)).pow(2) +
        (alpha * torch.sin(omega)).pow(2) + 1e-8
    )
    return -log_denom


# ─────────────────────────────────────────────
# Mode 1: Time-domain cascade (for GOLF end_filter)
# ─────────────────────────────────────────────

class VocalTractCascade(LTVMinimumPhaseFilterPrecise):
    """
    Hybrid analytic + learned all-pole vocal-tract filter (ARIA, stable version).

    Cascade structure (coefficient-domain, à la GOLF):
        [α tilt 1st-order] → [F1/B1 biquad] → [F2/B2 biquad]
                           → [learned biquad × n_learned, conj-parameterised]

    KEY: instead of N serial torchlpc calls (which explode in backward for
    high-Q poles), we build each section's biquad [1, a1, a2], cascade them in
    the COEFFICIENT domain via biquads2lpc (polynomial product), then run a
    SINGLE sample_wise_lpc — exactly like LTVMinimumPhaseFilterPrecise.
    The learned sections use the `conj` parameterisation which guarantees
    |pole| < max_abs_pole, and analytic formant poles are radius-clamped too.

    n_ctrl = 2 (log_gain, alpha) + 2 * n_formants (F, B per formant)
           + 2 * n_learned   (conj logits per learned biquad)
    AR order = 2 * (1 + n_formants + n_learned)   (1 = tilt section).
    n_formants=2 → F1/F2 (order 2*(3+n_learned), backward-compatible with old ckpts);
    n_formants=3 → adds F3.

    Integrate as end_filter in SourceFilterSynth:
      end_filter:
        class_path: models.analytic_filter.VocalTractCascade
        init_args: {sr: 16000, n_learned: 6}
    """

    def __init__(self, sr: int = 16000, n_learned: int = 6,
                 f1_range=(150, 1000), f2_range=(600, 3200),
                 f3_range=(1800, 4000),
                 b1_range=(30, 330), b2_range=(30, 430), b3_range=(40, 540),
                 n_formants: int = 2,
                 max_abs_pole: float = 0.99,
                 use_tilt: bool = True,
                 learned_placement: bool = False):
        # Skip LTVMinimumPhaseFilterPrecise.__init__ (it builds its own ctrl);
        # initialise the Controllable/nn.Module base directly.
        Controllable.__init__(self)
        self.sr           = sr
        self.n_learned    = n_learned
        self.f1_range     = f1_range
        self.f2_range     = f2_range
        self.f3_range     = f3_range
        self.max_abs_pole = max_abs_pole
        self.use_tilt     = use_tilt
        # n_formants analytic formant resonances. 2 = F1/F2 (backward-compatible with
        # existing checkpoints); 3 adds F3 for rhotic / erhua / fine vowel contrasts.
        self.n_formants   = n_formants
        self.f_ranges     = [f1_range, f2_range, f3_range][:n_formants]
        # bw_range = (lo, hi) in Hz; stored as (offset, scale) for sigmoid: bw = sigmoid*scale+offset
        _bw_defaults = [(30, 300), (30, 400), (40, 500)]
        _bw_cfg      = [
            (b1_range[0], b1_range[1] - b1_range[0]),
            (b2_range[0], b2_range[1] - b2_range[0]),
            (b3_range[0], b3_range[1] - b3_range[0]),
        ]
        # override defaults only when the caller explicitly passed a non-default bN_range
        self.bw_scales = [_bw_cfg[i] for i in range(n_formants)]
        # ABLATION: if learned_placement, the formant (F,B) → pole map is a small MLP
        # instead of the analytic r=exp(-pi*B/sr), theta=2*pi*F/sr. The control handle
        # (F1/F2 in Hz) is unchanged, so manipulation still works — but placement is
        # learned, which is what isolates the contribution of the analytic poles.
        self.learned_placement = learned_placement
        if learned_placement:
            self.placement_net = nn.Sequential(
                nn.Linear(3, 32), nn.GELU(),
                nn.Linear(32, 32), nn.GELU(),
                nn.Linear(32, 2))            # -> (r_logit, theta_logit)
        # conj parameterisation: guarantees |pole| < max_abs_pole for learned sections
        self.logits2biquad = get_logits2biquads("conj", max_abs_pole)
        # control layout: [log_gain, (alpha if use_tilt,)] + (F, B) per formant
        self.n_formant_ctrl = (1 + int(use_tilt)) + 2 * n_formants

        self.ctrl = wrap_ctrl_fn(
            split_size=(self.n_ctrl,),
            trsfm_fn=lambda x: (x,),   # passthrough raw ctrl; forward() decodes it
        )

    @property
    def n_ctrl(self):
        # [log_gain, (alpha,)] + (F, B) per formant + 2*n_learned (conj logits)
        return self.n_formant_ctrl + 2 * self.n_learned

    # ---- analytic biquad builders (frame-level) ----

    def _tilt_biquad(self, alpha: torch.Tensor) -> torch.Tensor:
        """1st-order real pole as a degenerate biquad [1, -alpha, 0]."""
        a1 = -alpha
        a2 = torch.zeros_like(alpha)
        return torch.stack([torch.ones_like(alpha), a1, a2], dim=-1)

    def _formant_biquad(self, f: torch.Tensor, bw: torch.Tensor) -> torch.Tensor:
        """Conjugate pole pair (F, BW) → biquad [1, a1, a2], radius-clamped."""
        r     = torch.exp(-math.pi * bw / self.sr).clamp(max=self.max_abs_pole)
        theta = 2 * math.pi * f / self.sr
        a1    = -2 * r * torch.cos(theta)
        a2    = r * r
        return torch.stack([torch.ones_like(a1), a1, a2], dim=-1)

    def _formant_biquad_learned(self, f: torch.Tensor, bw: torch.Tensor,
                                idx: int) -> torch.Tensor:
        """ABLATION variant: MLP maps (F, BW, formant-id) → (r, theta) → conjugate biquad,
        in place of the analytic placement formula. Stability guaranteed by sigmoid bounds."""
        nyq = self.sr / 2.0
        fid = idx / max(1, self.n_formants - 1)
        feat = torch.stack([f / nyq, bw / nyq, torch.full_like(f, fid)], dim=-1)
        out = self.placement_net(feat)
        r = torch.sigmoid(out[..., 0]) * self.max_abs_pole
        theta = torch.sigmoid(out[..., 1]) * math.pi
        a1 = -2 * r * torch.cos(theta)
        a2 = r * r
        return torch.stack([torch.ones_like(a1), a1, a2], dim=-1)

    def _decode_params(self, x: torch.Tensor) -> dict:
        """Raw logits (..., n_ctrl) → interpretable parameters."""
        out = {"log_gain": x[..., 0]}
        base = 1  # index after log_gain
        if self.use_tilt:
            out["alpha"] = torch.tanh(x[..., 1]) * self.max_abs_pole
            base = 2
        for i in range(self.n_formants):
            lo, hi = self.f_ranges[i]
            off, sc = self.bw_scales[i]
            out[f"f{i+1}"] = torch.sigmoid(x[..., base + 2 * i])     * (hi - lo) + lo
            out[f"b{i+1}"] = torch.sigmoid(x[..., base + 2 * i + 1]) * sc + off
        return out

    def _logits_to_gain_lpc(self, logits: AudioTensor) -> tuple:
        """
        Map encoder logits → (gain, a) for LTVMinimumPhaseFilterPrecise.forward.
        Builds all biquads at frame level, cascades in coefficient domain.
        Returns AudioTensors (gain (B,T), a (B,T,order)) keeping hop_length.
        """
        x = logits.as_tensor()                          # (B, T, n_ctrl)
        p = self._decode_params(x)

        # analytic sections: (optional tilt) + n_formants formant resonances
        if self.learned_placement:
            formant_bqs = [self._formant_biquad_learned(p[f"f{i+1}"], p[f"b{i+1}"], i)
                           for i in range(self.n_formants)]
        else:
            formant_bqs = [self._formant_biquad(p[f"f{i+1}"], p[f"b{i+1}"])
                           for i in range(self.n_formants)]
        if self.use_tilt:
            tilt_bq = self._tilt_biquad(p["alpha"])          # (B,T,3)
            analytic_bq = torch.stack([tilt_bq, *formant_bqs], dim=-2)  # (B,T,1+n_formants,3)
        else:
            analytic_bq = torch.stack(formant_bqs, dim=-2)              # (B,T,n_formants,3)

        # learned sections (conj → guaranteed stable)
        learned_logits = x[..., self.n_formant_ctrl:].reshape(
            *x.shape[:-1], self.n_learned, 2)
        learned_bq = self.logits2biquad(learned_logits)  # (B,T,n_learned,3)

        all_bq = torch.cat([analytic_bq, learned_bq], dim=-2)  # (B,T,3+n_learned,3)

        # coefficient-domain cascade → single AR polynomial (drop leading 1)
        a = biquads2lpc(all_bq)                          # (B,T, 2*(3+n_learned))
        gain = torch.exp(p["log_gain"])                  # (B,T)

        return logits.new_tensor(gain), logits.new_tensor(a)

    def forward(self, ex: AudioTensor, ctrl: AudioTensor) -> AudioTensor:
        # decode ctrl → (gain, a) inside forward so that manipulating ctrl
        # (set_formant_params) re-derives the cascade. Single sample_wise_lpc.
        gain, a = self._logits_to_gain_lpc(ctrl)
        return super().forward(ex, gain, a)

    # ---- inverse / re-synthesis hooks for excitation flow matching ----

    def reverse(self, y: AudioTensor, ctrl: AudioTensor) -> AudioTensor:
        """Inverse-filter a real waveform y → true filter-input excitation.
        x1 = A(z) * y, the FIR inverse of the all-pole cascade 1/A(z). This is how the
        flow-matching TARGET excitation is recovered from the ground-truth recording.
        Returns the excitation on the gain-included filter-input scale (matches x0=src*gain)."""
        _, a = self._logits_to_gain_lpc(ctrl)
        up_a = a.reduce_hop_length().as_tensor()
        fir = torch.cat([torch.ones_like(up_a[..., :1]), up_a], dim=-1)  # [1, a1, a2, ...]
        yt = y.as_tensor()
        T = min(yt.shape[1], fir.shape[1])
        x1 = fir_filt(yt[:, :T], fir[:, :T])
        return AudioTensor(x1)

    def synth_from_excitation(self, x_input: AudioTensor, ctrl: AudioTensor) -> AudioTensor:
        """Re-synthesize waveform from a (gain-included) filter-input excitation via the
        all-pole cascade 1/A(z). Inference path: flow-refined excitation → speech. Mirrors
        forward() but consumes the excitation directly (gain already baked in, no re-mul)."""
        _, a = self._logits_to_gain_lpc(ctrl)
        up_a = a.reduce_hop_length().as_tensor()
        xt = x_input.as_tensor()
        T = min(xt.shape[1], up_a.shape[1])
        y = sample_wise_lpc(xt[:, :T], up_a[:, :T])
        return AudioTensor(y)

    def get_formant_params(self, ctrl: torch.Tensor) -> dict:
        """Decode ctrl logits → interpretable Hz values (for logging/manipulation)."""
        return self._decode_params(ctrl)

    def set_formant_params(self, ctrl: torch.Tensor, f1=None, f2=None, f3=None,
                           alpha=None, b1=None, b2=None, b3=None) -> torch.Tensor:
        """Write target values back into ctrl logits → modified copy (manipulation).
        ctrl channels: [gain, (alpha if use_tilt,) (f, b) per formant, learned...].
        f3/b3 ignored when n_formants < 3; alpha ignored when use_tilt=False."""
        out = ctrl.clone()
        base = 1  # index after log_gain
        if self.use_tilt:
            if alpha is not None:
                a = max(min(float(alpha) / self.max_abs_pole, 0.999), -0.999)
                out[..., 1] = math.atanh(a)
            base = 2
        fs, bs = [f1, f2, f3], [b1, b2, b3]
        for i in range(self.n_formants):
            lo, hi = self.f_ranges[i]
            off, sc = self.bw_scales[i]
            if fs[i] is not None:  out[..., base + 2 * i]     = _inv_sigmoid(fs[i], lo, hi)
            if bs[i] is not None:  out[..., base + 2 * i + 1] = _inv_sigmoid(bs[i], off, off + sc)
        return out

    def scale_formants(self, ctrl: torch.Tensor, f1_scale=1.0, f2_scale=1.0,
                       f3_scale=1.0, alpha_scale=1.0, gain_scale=1.0,
                       alpha_delta=0.0) -> torch.Tensor:
        """Per-frame RELATIVE scaling: multiply each decoded parameter track by a ratio and
        write it back, preserving the contour. Covers formants F1/F2/F3, spectral tilt
        (alpha) and gain (energy). For sentence material where an absolute target is
        meaningless — shift the whole track by ±x%."""
        out = ctrl.clone()
        p = self._decode_params(ctrl)
        base = 2 if self.use_tilt else 1
        for i, sc in enumerate((f1_scale, f2_scale, f3_scale)[:self.n_formants]):
            if sc == 1.0:
                continue
            lo, hi = self.f_ranges[i]
            fn = (p[f"f{i+1}"] * sc).clamp(lo + 1.0, hi - 1.0)
            out[..., base + 2 * i] = torch.log((fn - lo) / (hi - fn))
        if gain_scale != 1.0:                       # energy: log_gain += log(scale)
            out[..., 0] = out[..., 0] + math.log(gain_scale)
        if self.use_tilt and (alpha_scale != 1.0 or alpha_delta != 0.0):
            a = (p["alpha"] * alpha_scale + alpha_delta).clamp(
                -0.999 * self.max_abs_pole, 0.999 * self.max_abs_pole)
            out[..., 1] = torch.atanh(a / self.max_abs_pole)
        return out


class VocalTractCascadeARMA(VocalTractCascade):
    """Pole–zero (ARMA) vocal tract — VocalTractCascade plus a numerator B(z).

    Adds `n_zeros` conjugate anti-formant biquads (interpretable Fz/Bz), cascaded via
    biquads2lpc into a numerator polynomial B(z). H(z)=B(z)/A(z) is applied two-stage: the
    parent's fast all-pole 1/A(z) (torchlpc kernel) then B(z) as a cheap LTV FIR (`fir_filt`).
    Zeros give the **anti-formants** (nasals / laterals / nasalised vowels) that an all-pole
    tract cannot represent, and are exposed as a controllable "nasalisation" dimension.
    (A unified philtorch `lpv.lfilter` rational filter was tried but needs Helion GPU kernels,
    absent here, so it ran ~1000× slower — the two-stage uses the proven fast path instead.)

    Low-risk by construction: zeros are an FIR/MA part ⇒ **unconditionally stable** and fully
    differentiable. The extra 2·n_zeros ctrl channels are appended AFTER the learned-pole
    logits, so the inherited pole/formant decoders and every manipulation method
    (`get/set/scale_formants`) are untouched. `n_zeros = 0` ⇒ **identical to the parent** (the
    parent's all-pole forward is used verbatim). At init the zero logits ≈ 0 ⇒ B(z) ≈ flat ⇒
    behaves ≈ all-pole.

    Note: the excitation-flow hooks `reverse()` / `synth_from_excitation()` are inherited
    unchanged and stay all-pole (they ignore the zeros); the standard forward synthesis path
    is full ARMA. See docs/armax_lf_aria_nasal.md.
    """

    def __init__(self, *args, n_zeros: int = 2, max_abs_zero: float = 0.999,
                 nasal_fz_range=(700, 2500), nasal_bz_range=(40, 600), **kwargs):
        # parent __init__ registers ctrl with the pole-only width (n_zeros unset ⇒ getattr→0)
        super().__init__(*args, **kwargs)
        self.n_zeros = n_zeros
        self.max_abs_zero = max_abs_zero
        # Each zero is an interpretable conjugate ANTI-FORMANT (Fz, Bz), so it can be read out
        # and manipulated (the "nasalisation" knob). Small Bz ⇒ zero near the unit circle ⇒
        # deep spectral notch ⇒ strong nasalisation; large Bz ⇒ shallow ⇒ ~oral.
        self.nasal_fz_range = nasal_fz_range
        self.nasal_bz_range = nasal_bz_range
        if n_zeros > 0:                          # re-register ctrl now that n_ctrl is wider
            self.ctrl = wrap_ctrl_fn(
                split_size=(self.n_ctrl,), trsfm_fn=lambda x: (x,))

    @property
    def n_ctrl(self):
        # parent (n_formant_ctrl + 2·n_learned) + 2·n_zeros = (Fz, Bz) per zero
        return VocalTractCascade.n_ctrl.fget(self) + 2 * getattr(self, "n_zeros", 0)

    def _n_pole_ctrl(self) -> int:
        return self.n_formant_ctrl + 2 * self.n_learned

    def _logits_to_gain_lpc(self, logits: AudioTensor):
        # hand the parent only the pole channels; trailing zero channels handled in _numerator
        if getattr(self, "n_zeros", 0) == 0:
            return super()._logits_to_gain_lpc(logits)
        x = logits.as_tensor()
        return super()._logits_to_gain_lpc(logits.new_tensor(x[..., : self._n_pole_ctrl()]))

    def _zero_biquad(self, fz: torch.Tensor, bz: torch.Tensor) -> torch.Tensor:
        """Conjugate-zero pair (Fz, Bz) → numerator biquad [1, b1, b2]; radius clamped < max_abs_zero."""
        r = torch.exp(-math.pi * bz / self.sr).clamp(max=self.max_abs_zero)
        theta = 2 * math.pi * fz / self.sr
        b1 = -2 * r * torch.cos(theta)
        b2 = r * r
        return torch.stack([torch.ones_like(b1), b1, b2], dim=-1)

    def _decode_zeros(self, x: torch.Tensor):
        """trailing zero logits → (Fz, Bz) per zero, each (..., n_zeros) in Hz."""
        zl = x[..., self._n_pole_ctrl():].reshape(*x.shape[:-1], self.n_zeros, 2)
        flo, fhi = self.nasal_fz_range
        blo, bhi = self.nasal_bz_range
        fz = torch.sigmoid(zl[..., 0]) * (fhi - flo) + flo
        bz = torch.sigmoid(zl[..., 1]) * (bhi - blo) + blo
        return fz, bz

    def _numerator(self, logits: AudioTensor) -> AudioTensor:
        """zero (Fz,Bz) → numerator coeffs b = [b1, b2, …] (leading 1 dropped), as AudioTensor."""
        fz, bz = self._decode_zeros(logits.as_tensor())
        zbq = torch.stack([self._zero_biquad(fz[..., i], bz[..., i])
                           for i in range(self.n_zeros)], dim=-2)   # (B,T,n_zeros,3)
        b = biquads2lpc(zbq)                     # (B,T, 2·n_zeros)
        return logits.new_tensor(b)

    # ---- nasalisation control (manipulate the anti-formants) ----

    def get_nasal_params(self, ctrl: torch.Tensor) -> dict:
        """Decode → anti-formant frequency Fz and bandwidth Bz (Hz) per zero."""
        fz, bz = self._decode_zeros(ctrl)
        return {"fz": fz, "bz": bz}

    def set_nasal_params(self, ctrl: torch.Tensor, fz=None, bz=None,
                         zero_idx: int = 0) -> torch.Tensor:
        """Place / deepen an anti-formant: write target centre Fz (Hz) and/or bandwidth Bz (Hz)
        into the chosen zero's ctrl channels → modified copy. Scalars broadcast over all frames;
        smaller Bz ⇒ zero nearer the unit circle ⇒ deeper notch ⇒ stronger nasalisation. Used by
        the controllable-nasalisation demo (oral→nasal continuum). Values clamped to the
        configured nasal_fz_range / nasal_bz_range."""
        if self.n_zeros == 0:
            return ctrl
        out = ctrl.clone()
        base = self._n_pole_ctrl()
        flo, fhi = self.nasal_fz_range
        blo, bhi = self.nasal_bz_range
        if fz is not None:
            out[..., base + 2 * zero_idx]     = _inv_sigmoid(
                min(max(float(fz), flo + 1.0), fhi - 1.0), flo, fhi)
        if bz is not None:
            out[..., base + 2 * zero_idx + 1] = _inv_sigmoid(
                min(max(float(bz), blo + 1.0), bhi - 1.0), blo, bhi)
        return out

    def scale_nasalization(self, ctrl: torch.Tensor, nasal_scale: float = 1.0) -> torch.Tensor:
        """Per-frame nasalisation strength: scale the anti-formant bandwidth Bz (→ notch depth).
        nasal_scale > 1 ⇒ smaller Bz ⇒ deeper notch ⇒ more nasal; < 1 ⇒ more oral. Fz unchanged."""
        if self.n_zeros == 0 or nasal_scale == 1.0:
            return ctrl
        out = ctrl.clone()
        _, bz = self._decode_zeros(ctrl)
        blo, bhi = self.nasal_bz_range
        base = self._n_pole_ctrl()
        for i in range(self.n_zeros):
            bn = (bz[..., i] / nasal_scale).clamp(blo + 1e-3, bhi - 1e-3)
            out[..., base + 2 * i + 1] = torch.log((bn - blo) / (bhi - bn))   # inverse sigmoid
        return out

    def forward(self, ex: AudioTensor, ctrl: AudioTensor) -> AudioTensor:
        # Two-stage H(z)=B(z)/A(z): the parent's fast all-pole 1/A(z) (torchlpc CUDA kernel),
        # then the numerator B(z) as a cheap LTV FIR. Chosen over a unified philtorch
        # `lpv.lfilter` because that needs Helion GPU kernels (not installed here) and otherwise
        # falls back to a Python recurrence that is ~1000× slower. The two-stage uses the same
        # proven fast kernel as the all-pole v1–v6 models; zeros (FIR) are unconditionally stable.
        y = super().forward(ex, ctrl)            # gain · 1/A(z) (pole-only via our _logits_to_gain_lpc)
        if getattr(self, "n_zeros", 0) == 0:
            return y
        up_b = self._numerator(ctrl).reduce_hop_length().as_tensor()
        fir = torch.cat([torch.ones_like(up_b[..., :1]), up_b], dim=-1)   # [1, b1, b2, …]
        yt = y.as_tensor()
        T = min(yt.shape[1], fir.shape[1])
        return AudioTensor(fir_filt(yt[:, :T], fir[:, :T]))


# ─────────────────────────────────────────────
# Mode 2: Frequency-domain A_k (for DDSP H+N)
# ─────────────────────────────────────────────

class AnalyticHarmonics(nn.Module):
    """
    Parameterized harmonic amplitudes:

      log A_k = log|H_F1(k·F0)| + log|H_F2(k·F0)|
              + log|G_tilt(k·F0)| + residual_k

    The model predicts: alpha (tilt), F1, B1, F2, B2, residual (K-dim).
    The formant contribution is computed analytically from (F, B) and F0.

    Used as the harm_oscillator in HarmonicPlusNoiseSynth.
    """

    def __init__(self, sr: int = 16000, n_harmonics: int = 80,
                 f1_range=(200, 900), f2_range=(800, 2800)):
        super().__init__()
        self.sr          = sr
        self.n_harmonics = n_harmonics
        self.f1_range    = f1_range
        self.f2_range    = f2_range

    @property
    def n_ctrl(self):
        # alpha, F1, B1, F2, B2, residual[K]
        return 5 + self.n_harmonics

    def forward(self, f0: AudioTensor, ctrl: AudioTensor) -> torch.Tensor:
        """
        f0   : (B, T) Hz, sample-level
        ctrl : (B, T_frames, n_ctrl)
        Returns harmonic waveform (B, T).
        """
        ctrl_up = ctrl.reduce_hop_length().as_tensor()
        T = f0.as_tensor().shape[1]
        ctrl_up = ctrl_up[:, :T]
        f0_t    = f0.as_tensor()[:, :T]

        # decode ctrl
        f1_lo, f1_hi = self.f1_range
        f2_lo, f2_hi = self.f2_range

        alpha    = torch.tanh(ctrl_up[..., 0:1]) * 0.99
        f1       = torch.sigmoid(ctrl_up[..., 1:2]) * (f1_hi - f1_lo) + f1_lo
        b1       = torch.sigmoid(ctrl_up[..., 2:3]) * 300 + 30
        f2       = torch.sigmoid(ctrl_up[..., 3:4]) * (f2_hi - f2_lo) + f2_lo
        b2       = torch.sigmoid(ctrl_up[..., 4:5]) * 400 + 30
        residual = ctrl_up[..., 5:]                    # (B, T, K)

        # harmonic frequencies: (B, T, K)
        k      = torch.arange(1, self.n_harmonics + 1,
                               device=f0_t.device, dtype=f0_t.dtype)
        k_freq = k * f0_t.unsqueeze(-1)                # (B, T, K)

        # analytic log amplitude
        log_A  = formant_log_mag_at_harmonics(k_freq, f1, b1, self.sr)
        log_A += formant_log_mag_at_harmonics(k_freq, f2, b2, self.sr)
        log_A += tilt_log_mag_at_harmonics(k_freq, alpha, self.sr)
        log_A += residual                               # learned correction

        A = torch.exp(log_A).clamp(max=100)            # (B, T, K)

        # synthesize harmonics
        # phase accumulation
        phase_inc = 2 * math.pi * k_freq / self.sr     # (B, T, K)
        phase = torch.cumsum(phase_inc, dim=1)          # (B, T, K)

        # voiced mask from F0
        voiced = (f0_t > 0).unsqueeze(-1).float()      # (B, T, 1)
        wave   = (A * voiced * torch.sin(phase)).sum(-1)  # (B, T)
        return wave

    def get_formant_params(self, ctrl: torch.Tensor) -> dict:
        f1_lo, f1_hi = self.f1_range
        f2_lo, f2_hi = self.f2_range
        return {
            "alpha": torch.tanh(ctrl[..., 0]) * 0.99,
            "f1":    torch.sigmoid(ctrl[..., 1]) * (f1_hi - f1_lo) + f1_lo,
            "b1":    torch.sigmoid(ctrl[..., 2]) * 300 + 30,
            "f2":    torch.sigmoid(ctrl[..., 3]) * (f2_hi - f2_lo) + f2_lo,
            "b2":    torch.sigmoid(ctrl[..., 4]) * 400 + 30,
        }


# ─────────────────────────────────────────────
# Mode 3: OscillatorInterface drop-in (for DDSP harm_oscillator)
# ─────────────────────────────────────────────

class AnalyticHarmonicsOscillator(OscillatorInterface):
    """
    Drop-in replacement for AdditiveSynthesizer in HarmonicPlusNoiseSynth.

    Synthesizes harmonics with amplitudes shaped by analytic formant model:
      log A_k = log|H_F1(k·F0)| + log|H_F2(k·F0)| + log|G_tilt(k·F0)| + residual_k

    n_ctrl = 5 (alpha, F1, B1, F2, B2) + n_harmonics (residual per harmonic)

    Integrate into HarmonicPlusNoiseSynth as harm_oscillator:
      harm_oscillator:
        class_path: models.analytic_filter.AnalyticHarmonicsOscillator
        init_args:
          sr: 16000
          n_harmonics: 80
    """

    def __init__(self, sr: int = 16000, n_harmonics: int = 80,
                 f1_range=(150, 1000), f2_range=(600, 3200)):
        super().__init__()
        self.sr          = sr
        self.n_harmonics = n_harmonics
        self.f1_range    = f1_range
        self.f2_range    = f2_range
        self.n_formant_ctrl = 5   # [alpha,f1,b1,f2,b2] — per-harmonic residual after
        self.ctrl = wrap_ctrl_fn(
            split_size=(self.n_ctrl,),
            trsfm_fn=lambda x: (x,),
        )

    @property
    def n_ctrl(self) -> int:
        return 5 + self.n_harmonics

    def forward(self, phase: AudioTensor, ctrl: AudioTensor) -> AudioTensor:
        """
        phase : AudioTensor (B, T_samples), normalized F0/sr ∈ [0, 0.5]
        ctrl  : AudioTensor (B, T_frames, n_ctrl) — raw logits from encoder
        Returns AudioTensor (B, T_samples) harmonic waveform.
        """
        ctrl_up = ctrl.reduce_hop_length().as_tensor()   # (B, T_ctrl, n_ctrl)
        T = phase.as_tensor().shape[1]
        ctrl_up = ctrl_up[:, :T]                        # clip to max sample len
        T = ctrl_up.shape[1]                            # actual length (may be < input T)
        f0_hz   = phase.as_tensor()[:, :T] * self.sr   # (B, T) in Hz

        f1_lo, f1_hi = self.f1_range
        f2_lo, f2_hi = self.f2_range
        alpha    = torch.tanh(ctrl_up[..., 0:1]) * 0.99
        f1       = torch.sigmoid(ctrl_up[..., 1:2]) * (f1_hi - f1_lo) + f1_lo
        b1       = torch.sigmoid(ctrl_up[..., 2:3]) * 300 + 30
        f2       = torch.sigmoid(ctrl_up[..., 3:4]) * (f2_hi - f2_lo) + f2_lo
        b2       = torch.sigmoid(ctrl_up[..., 4:5]) * 400 + 30
        residual = ctrl_up[..., 5:]                      # (B, T, K)

        k      = torch.arange(1, self.n_harmonics + 1,
                               device=f0_hz.device, dtype=f0_hz.dtype)
        k_freq = k * f0_hz.unsqueeze(-1)                 # (B, T, K)

        log_A  = formant_log_mag_at_harmonics(k_freq, f1, b1, self.sr)
        log_A += formant_log_mag_at_harmonics(k_freq, f2, b2, self.sr)
        log_A += tilt_log_mag_at_harmonics(k_freq, alpha, self.sr)
        log_A += residual
        A = torch.exp(log_A).clamp(max=100)              # (B, T, K)

        # cumulative phase from instantaneous frequency
        phase_inc = 2 * math.pi * k_freq / self.sr
        cum_phase = torch.cumsum(phase_inc, dim=1)

        voiced = (f0_hz > 0).unsqueeze(-1).float()
        wave = (A * voiced * torch.sin(cum_phase)).sum(-1)  # (B, T)
        wave = wave / math.sqrt(self.n_harmonics)           # normalize power

        return AudioTensor(wave)

    def get_formant_params(self, ctrl: torch.Tensor) -> dict:
        f1_lo, f1_hi = self.f1_range
        f2_lo, f2_hi = self.f2_range
        return {
            "alpha": torch.tanh(ctrl[..., 0]) * 0.99,
            "f1":    torch.sigmoid(ctrl[..., 1]) * (f1_hi - f1_lo) + f1_lo,
            "b1":    torch.sigmoid(ctrl[..., 2]) * 300 + 30,
            "f2":    torch.sigmoid(ctrl[..., 3]) * (f2_hi - f2_lo) + f2_lo,
            "b2":    torch.sigmoid(ctrl[..., 4]) * 400 + 30,
        }

    def set_formant_params(self, ctrl: torch.Tensor, f1=None, f2=None,
                           alpha=None, b1=None, b2=None) -> torch.Tensor:
        """Write target values back into ctrl logits → modified copy (manipulation).
        ctrl channels: [alpha, f1, b1, f2, b2, residual...]."""
        out = ctrl.clone()
        f1_lo, f1_hi = self.f1_range
        f2_lo, f2_hi = self.f2_range
        if alpha is not None:
            a = max(min(float(alpha) / 0.99, 0.999), -0.999)
            out[..., 0] = math.atanh(a)
        if f1 is not None:  out[..., 1] = _inv_sigmoid(f1, f1_lo, f1_hi)
        if b1 is not None:  out[..., 2] = _inv_sigmoid(b1, 30, 330)
        if f2 is not None:  out[..., 3] = _inv_sigmoid(f2, f2_lo, f2_hi)
        if b2 is not None:  out[..., 4] = _inv_sigmoid(b2, 30, 430)
        return out
