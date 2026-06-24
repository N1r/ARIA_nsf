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

The table below is for the **16 kHz / F024** configs (where v4/v5 live). Tilt and
F/B ranges are the same at every sample rate, but **`n_learned` scales with sample
rate** (see note ▼).

| | spectral **tilt** (alpha) | `n_learned` (16k) | **F1** range (Hz) | **F2** range (Hz) | **B1** range (Hz) | **B2** range (Hz) | LPC order (16k) |
|---|:---:|:---:|---|---|---|---|:---:|
| **v1** | ✅ on | 6 | 150–1300 | 600–3200 | 30–330 *(wide, default)* | 30–430 *(wide, default)* | 18 |
| **v2** | ❌ off | 7 | 200–1000 | 700–3000 | 50–200 *(tight)* | 50–250 *(tight)* | 18 |
| **v3** | ✅ on | 6 | 150–1300 | 600–3200 | 50–200 *(tight)* | 50–250 *(tight)* | 18 |
| **v4** | ✅ on | 6 | 150–1300 | 600–3200 | 50–200 *(tight)* | 50–250 *(tight)* | 18 |
| **v5** | ✅ on | 6 | 150–1300 | 600–3200 | 50–200 *(tight)* | 50–250 *(tight)* | 18 |

> **LPC order convention.** Order = `2 × (n_sections)` where sections = (tilt if on)
> + `n_formants` (=2) + `n_learned`. At 16 k all variants = `2×(1+2+6)=18` (v2:
> `2×(2+7)=18`, tilt removed but +1 learned compensates). Note the tilt section is a
> **degenerate biquad** `[1, −α, 0]` (one real pole), so the *effective* pole count
> is 17 for the tilt versions; an older v1 config comment reports "17" for this
> reason. Same filter, two counting conventions — this doc uses the nominal
> `2×n_sections` (matching the 22 k/24k configs, which all report 22).

> **⚠️ `n_learned` is sample-rate-dependent** — higher SR → higher Nyquist → more
> poles to cover the extra high-frequency resonances → higher LPC/AR order.
> The (F1,F2) analytic formants and tilt are unchanged; only the number of free
> learned biquads grows. **v4/v5 are F024-only (16 kHz, n_learned 6).**
>
> | SR (speaker) | LPC order | v1 `n_learned` | v2 | v3 |
> |---|:---:|:---:|:---:|:---:|
> | **16k** (F024) | 18 | **6** | 7 | **6** |
> | **22k** (LJSpeech) | 22 | **8** | 9 | **8** |
> | **24k** (CSMSC) | 22 | **8** | 9 | **8** |
>
> So the main CSMSC/LJSpeech models use **n_learned = 8** (v1/v3) — this is likely
> the "8" you remember. v2 adds +1 learned biquad (7→/9) to compensate for the
> removed tilt pole and keep the AR order matched.

**v4 and v5 use the v3 vocal tract unchanged** — they differ from v3 only in the
excitation (§3).

---

## 3. Excitation topology — how harmonic + noise combine (the v3 → v5 change)

Let `harm` = voicing-gated glottal flow, `noise` = white noise (`randn`, **not**
amplitude-modulated by the harmonic), `H` = noise spectral filter (zero-phase FIR,
`n_mag=128`), `A` = aperiodicity filter (zero-phase FIR, `n_mag=128`).

> **Read this first — what "src" is and where the LPC sits.**
> All five versions build a single **source / excitation** signal `src` by combining
> the harmonic and the (pre-filtered) noise **at the source level**, and *then* pass
> the **combined** `src` through the LPC vocal tract **once**:
>
> ```
>   harm ─┐
>         ├─(combine, §3)─► src ──► end_filter ──► room_filter ──► x̂
>  noise ─┘                       (LPC all-pole       (global
>                                  vocal tract,        LTI FIR)
>                                  LTVMinimumPhase)
> ```
>
> It is **`LPC(harm + noise)`, NOT `LPC(harm) + LPC(noise)`** in implementation
> (one filter pass). Because `end_filter` is **linear**, the two are mathematically
> equal — but the code sums at the source and filters once. `end_filter`
> (`VocalTractCascade`) is an LTV **all-pole / LPC** filter (`models/sf.py:64`).
>
> Note the **asymmetry before the LPC** (for v1–v4): the noise carries its *own*
> pre-filter `H` (and, in v4, `A`); the harmonic does not. So spectrally
> `harm ∝ glottal(f)·|LPC(f)|` while `noise ∝ |H(f)|·|LPC(f)|`. **v5 is the
> exception** — there the same `H` also shapes the harmonic via the `(I−H)`
> crossfade, so the harmonic is *not* unfiltered. The diagrams below
> therefore stop at `src` — the `end_filter → room_filter` tail is identical for all
> versions and for the harmonic/noise branches alike (both are filtered by the same
> vocal tract because they are summed first). The v3→v4→v5 change is *only* in how
> `src` is formed.

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

| | formant_loss_weight | formant_smooth_weight | residual_reg_weight | batch × steps | data passes |
|---|:---:|:---:|:---:|:---:|:---:|
| v1 | **0 (off)** | — | — | 64 × 40k | 1× |
| v2 | 1.5 | — | 0.02 | 64 × 80k | 2× |
| v3 | 2.0 | 0.05 | 0.02 | 128 × 40k | 2× |
| v4 | 2.0 | 0.05 | 0.02 | 128 × 40k | 2× |
| v5 | 2.0 | 0.05 | 0.02 | 128 × 40k | 2× |

All: Adam lr 2e-4, from scratch, F024 16 kHz. (batch128×40k ≡ batch64×80k data passes — the "fast recipe".)

> **⚠️ v1 has NO explicit formant supervision** (`formant_loss_weight=0`, verified
> from `runs/aria_golf_16k/config.yaml`). Its F1/F2 emerge purely from the MSS
> reconstruction loss. The eval-HTML label "sup 0.5" for v1 is **inaccurate** — the
> real distinction is unsupervised (v1) → supervised 1.5 (v2) → 2.0 + smoothness
> (v3+). v1 also saw half the data passes of the others.

---

## 5. Design narrative (for figure captions)

| ver | one-line intent | what changed | trade-off |
|---|---|---|---|
| **v1** | first analytic-formant ARIA | tilt on, wide BW, **no formant supervision** (F1/F2 emerge from reconstruction) | good reconstruction, **loose/unconstrained** formant control |
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
