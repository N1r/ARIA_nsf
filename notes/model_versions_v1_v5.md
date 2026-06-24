# ARIA-GOLF model versions v1 → v5: structural differences

A figure-oriented reference. All numbers verified from `cfg/ae/decoder/aria_golf_v*_16k.yaml`,
`cfg/ae/f024_aria_v*_16k.yaml`, `models/sf.py`, `models/filters.py`,
`models/analytic_filter.py` (16 kHz / F024 configs).

All five versions share the **same encoder, glottal oscillator, noise generator,
room filter and MSS loss**. The differences are confined to two places:

1. **`end_filter` (VocalTractCascade)** — the analytic vocal-tract model (v1→v3).
2. **excitation topology** — how the harmonic source and noise are combined (v3→v5).

---

## 1. Shared backbone (identical across v1–v5)

```
                          true F0 (used, not predicted)
                              │
  audio x ──► Encoder ────────┼─────────────────────────────────┐
            (UNet + BiLSTM)   │  per-frame control params        │
                              ▼                                   ▼
                     ┌─────────────────┐                 ┌─────────────────┐
        phase ─────► │ Glottal-flow    │  harm_osc       │  White noise    │
                     │ oscillator (LF) │ ───────────►    │  generator      │
                     └─────────────────┘                 └─────────────────┘
                              ▲                                   │
                          voicing gate                            ▼
                                                          ┌─────────────────┐
                                                          │  noise_filter H │
                                                          └─────────────────┘
                              ... combine (see §3) ...  ──► src
                                                                  │
                                                                  ▼
                                                      ┌───────────────────────┐
                                                      │ end_filter            │
                                                      │ VocalTractCascade     │  (see §2)
                                                      └───────────────────────┘
                                                                  │
                                                                  ▼
                                                          room_filter (LTI, len 128)
                                                                  │
                                                                  ▼
                                                              output x̂
```

| Component | Setting (all versions) |
|---|---|
| Encoder | `VocoderParameterEncoderInterface`, UNet backbone (`n_fft=512`, `hop=160`, channels `[32,64,128,256]`, strides `[4,4,4,4]`), BiLSTM hidden 256 × 3 layers, dropout 0.1 |
| F0 / voicing | `learn_f0=false`, `learn_voicing=false` — uses **ground-truth F0**, voicing from F0>0 |
| Harmonic source | `DownsampledIndexedGlottalFlowTable` — LF glottal-flow model (derivative, `R_d∈[0.3,2.7]`, `lf_v2`, 2048-pt table, 64 ch). **Not** a naive pulse train, but still perfectly periodic (zero jitter). `trainable=false` |
| Noise source | `StandardNormalNoise` (white) |
| Room filter | `LTIAcousticFilter`, length 128, FFT conv |
| MSS loss | `n_ffts=[127,255,511,1023]`, Hann |
| ~Params | ≈ 5.5–5.6 M (v4 slightly larger: extra aperiodicity head) |

> Encoder output width is auto-derived from the decoder's declared parameter
> splits (`decoder.split_sizes_and_trsfms`), so adding a head (v4) automatically
> widens the encoder's final projection.

---

## 2. `end_filter` = VocalTractCascade — the analytic vocal tract (v1 → v3)

Control layout: `[log_gain, (alpha if tilt), (F,B) per formant, learned biquads…]`.
The first 1–2 formants (F1, F2) are **analytic, range-constrained** (this is the
manipulation knob); the rest are `n_learned` free biquad sections.

| | spectral **tilt** (alpha) | `n_learned` biquads | **F1** range (Hz) | **F2** range (Hz) | **B1** range (Hz) | **B2** range (Hz) | AR order |
|---|:---:|:---:|---|---|---|---|:---:|
| **v1** | ✅ on | 6 | 150–1300 | 600–3200 | 30–330 *(wide, default)* | 30–430 *(wide, default)* | 17 |
| **v2** | ❌ off | 7 | 200–1000 | 700–3000 | 50–200 *(tight)* | 50–250 *(tight)* | 18 |
| **v3** | ✅ on | 6 | 150–1300 | 600–3200 | 50–200 *(tight)* | 50–250 *(tight)* | 18 |
| **v4** | ✅ on | 6 | 150–1300 | 600–3200 | 50–200 *(tight)* | 50–250 *(tight)* | 18 |
| **v5** | ✅ on | 6 | 150–1300 | 600–3200 | 50–200 *(tight)* | 50–250 *(tight)* | 18 |

**v4 and v5 use the v3 vocal tract unchanged** — they differ from v3 only in the
excitation (§3).

---

## 3. Excitation topology — how harmonic + noise combine (the v3 → v5 change)

Let `harm` = voicing-gated glottal flow, `noise` = white noise,
`H` = noise spectral filter (zero-phase FIR, `n_mag=128`),
`A` = aperiodicity filter (zero-phase FIR, `n_mag=128`).

### v1 / v2 / v3 — purely additive (`subtract_harmonics = false`)

```
   harm ───────────────────────────────►(+)──► src
                                          ▲
   noise ──► H (exp, unbounded) ──────────┘
```
$$ \text{src} = \text{harm} + H(\text{noise}) $$

Noise can only **raise the noise floor**; it cannot suppress the coherent
harmonics, which the glottal source emits all the way to Nyquist. → **high-band
harmonic-stripe artifact** (see `docs/f024_highband_artifact.md`).

### v4 — decoupled band-aperiodicity head (`SourceFilterSynthAP`, NEW)

```
   harm ──►(I − A)────────────────────────►(+)──► src
                                            ▲
   noise ──► H (exp) ──► A (sigmoid+prior) ─┘
```
$$ \text{src} = (I-A)\,\text{harm} \;+\; A\,H(\text{noise}) $$

- `H` (unbounded, exp) keeps shaping the **noise spectral envelope** (colour/level).
- `A(f) = \sigma(\text{logits} + \text{freq\_bias}) \in [0,1]` is an explicit
  **band aperiodicity** (inverse-HNR). The same `A` weights harmonic by `(1−A)`
  and noise by `A`, per frequency band.
- `freq_bias` is a fixed rising prior (`−2` at DC → `+2` at Nyquist, `hf_bias=2.0`):
  low band starts harmonic-dominated, high band starts noise-dominated.
- The two concerns are **decoupled**: `H` = noise colour, `A` = harmonic/noise mix.

### v5 — minimal crossfade (`subtract_harmonics = true` + bounded `H`)

```
   harm ──►(I − H)────────────►(+)──► src
                                ▲
   noise ──► H (sigmoid) ───────┘
```
$$ \text{src} = \text{harm} + H(\text{noise}) - H(\text{harm}) = (I-H)\,\text{harm} + H\,\text{noise} $$

- One filter `H` does double duty: noise spectral shape **and** aperiodicity.
- `H`'s magnitude is bounded to `[0,1]` (`mag_activation: sigmoid`) so `(I−H)`
  stays a well-posed crossfade (the reason the crossfade was historically disabled
  was that `H=exp(·)` is unbounded → `(I−H)` could go negative).
- Limitation vs v4: noise level is **coupled** to harmonic level per band (no
  separate noise gain), and one filter must serve both roles.

### Noise / aperiodicity filter summary

| | noise filter `H` | magnitude activation | extra aperiodicity head `A` |
|---|---|---|---|
| v1–v3 | `LTVZeroPhaseFIRFilter`, n_mag 128 | `exp` (unbounded) | — |
| **v4** | `LTVZeroPhaseFIRFilter`, n_mag 128 | `exp` (unbounded) | `AperiodicityFIRFilter`, n_mag 128, `sigmoid`+rising prior |
| **v5** | `LTVZeroPhaseFIRFilter`, n_mag 128 | **`sigmoid`** (bounded [0,1]) | — (H itself is the crossfade) |

---

## 4. Training / supervision differences

| | formant_loss_weight | formant_smooth_weight | residual_reg_weight | warm-start |
|---|:---:|:---:|:---:|---|
| v1 | ~0.5 *(light)* | — | — | from scratch |
| v2 | 1.5 | — | 0.02 | from scratch |
| v3 | 2.0 | 0.05 | 0.02 | from scratch |
| v4 | 2.0 | 0.05 | 0.02 | from scratch |
| v5 | 2.0 | 0.05 | 0.02 | from scratch |

All: batch 128, 40k steps (fast recipe), Adam lr 2e-4, F024 16 kHz.

---

## 5. Design narrative (for figure captions)

| ver | one-line intent | what changed | trade-off |
|---|---|---|---|
| **v1** | first analytic-formant ARIA | tilt on, wide BW, light supervision | good reconstruction, **loose** formant control |
| **v2** | tighten formant control | drop tilt, tighten BW, ↑supervision, F1→[200,1000] | controllable but **lost spectral tilt**; F1 range too narrow |
| **v3** | best of both (current base) | **restore tilt**, keep tight BW, F1 back to [150,1300], strongest supervision + smoothness | excellent manipulation; **residual high-band harmonic artifact** |
| **v4** | fix artifact, principled | add explicit **decoupled band-aperiodicity** head (HNR control) | +1 head; most expressive aperiodicity |
| **v5** | fix artifact, minimal | re-enable crossfade with **bounded** noise filter | zero new params; noise level coupled per band |

The v3→v4/v5 step is the structural fix for the high-band artifact: v1–v3 cannot
dissolve high-frequency harmonics into aspiration noise (additive only); v4/v5 give
the model an explicit per-band harmonic-vs-noise mixing weight.

---

## 6. Suggested figures

1. **Lineage / ablation tree** — v1 → v2 → v3 (vocal-tract refinements) → {v4, v5}
   (excitation fix), with the one-line intents from §5.
2. **Signal-flow comparison** — the three boxes in §3 (additive / decoupled-AP /
   crossfade) side by side; this is the core architecture figure.
3. **VocalTractCascade table** as §2 (tilt, n_learned, F/B ranges).
4. **Spectrogram strip** — reference vs v3 vs v4 vs v5 on a voiced F024 segment,
   highlighting the 4–8 kHz band (artifact present in v3, dissolved in v4/v5).
5. **Aperiodicity curve** — plot the learned `A(f)` (v4) / `H(f)` (v5) for a voiced
   frame, showing the rising harmonic→noise transition with frequency.
</content>
