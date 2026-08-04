import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union

from .synth import OscillatorInterface
from .filters import LTVFilterInterface, FilterInterface
from .noise import NoiseInterface
from .audiotensor import AudioTensor
from .ctrl import PassThrough, Synth


class SourceFilterSynth(Synth):
    def __init__(
        self,
        harm_oscillator: OscillatorInterface,
        noise_generator: NoiseInterface,
        noise_filter: Union[LTVFilterInterface, PassThrough],
        end_filter: Union[LTVFilterInterface, PassThrough],
        room_filter: Union[FilterInterface, PassThrough] = None,
        subtract_harmonics: bool = True,
    ):
        super().__init__()
        self.subtract_harmonics = subtract_harmonics

        # Time-varying components
        self.harm_oscillator = harm_oscillator
        self.noise_generator = noise_generator
        self.noise_filter = noise_filter
        self.end_filter = end_filter

        # Room filter
        self.room_filter = room_filter if room_filter is not None else PassThrough()

    def forward(
        self,
        phase: AudioTensor,
        harm_oscillator_params: Tuple[AudioTensor, ...],
        noise_generator_params: Tuple[AudioTensor, ...],
        noise_filter_params: Tuple[AudioTensor, ...],
        end_filter_params: Tuple[AudioTensor, ...],
        voicing: Optional[AudioTensor] = None,
        target: Optional[AudioTensor] = None,
        **other_params
    ) -> AudioTensor:
        # Time-varying components
        harm_osc = self.harm_oscillator(phase, *harm_oscillator_params)
        if voicing is not None:
            assert torch.all(voicing >= 0) and torch.all(voicing <= 1)
            voicing = F.threshold(voicing, 0.5, 0)
            harm_osc = harm_osc * voicing

        src = harm_osc + self.noise_filter(
            self.noise_generator(harm_osc, *noise_generator_params),
            *noise_filter_params
        )

        if self.subtract_harmonics:
            src = src - self.noise_filter(harm_osc, *noise_filter_params)

        if target is not None:
            src, target_src = self.end_filter.reverse(src, target, *end_filter_params)
            return src, target_src
        return self.room_filter(self.end_filter(src, *end_filter_params))

    def excitation(
        self,
        phase: AudioTensor,
        harm_oscillator_params: Tuple[AudioTensor, ...],
        noise_generator_params: Tuple[AudioTensor, ...],
        noise_filter_params: Tuple[AudioTensor, ...],
        voicing: Optional[AudioTensor] = None,
        **other_params
    ) -> AudioTensor:
        """Return the pre-vocal-tract-filter source signal `src` (glottal flow + shaped
        noise), i.e. the deterministic excitation that feeds end_filter. This is x0, the
        informed prior for excitation flow matching. Mirrors the src computation in
        forward() exactly so the flow head sees the same signal the filter would."""
        harm_osc = self.harm_oscillator(phase, *harm_oscillator_params)
        if voicing is not None:
            voicing = F.threshold(voicing, 0.5, 0)
            harm_osc = harm_osc * voicing
        src = harm_osc + self.noise_filter(
            self.noise_generator(harm_osc, *noise_generator_params),
            *noise_filter_params
        )
        if self.subtract_harmonics:
            src = src - self.noise_filter(harm_osc, *noise_filter_params)
        return src


class SourceFilterSynthAP(SourceFilterSynth):
    """v4: source-filter synth with an explicit, decoupled band-aperiodicity head.

    v3 builds the excitation purely additively (src = harm + H(noise)), so the
    coherent harmonics produced by the glottal oscillator are never suppressed at
    high frequency — the noise filter can only raise the noise floor, not dissolve
    the harmonic stripes. This is the structural cause of the F024 high-band
    harmonic artifact.

    Here `noise_filter` (unbounded) keeps shaping the noise *spectral envelope*,
    while a separate `aperiodicity_filter` predicts a bounded A(f) in [0,1] that
    crossfades harmonic vs noise per band:

        src = (I - A)·harm + A·noise_filter(noise)

    A(f) is the band aperiodicity / inverse-HNR (WORLD-style). The two concerns
    are decoupled: H = noise colour/level, A = how much of each band is noise."""

    def __init__(self, *args, aperiodicity_filter, **kwargs):
        # we implement the harmonic/noise mix ourselves; never use the base subtract path
        kwargs["subtract_harmonics"] = False
        super().__init__(*args, **kwargs)
        self.aperiodicity_filter = aperiodicity_filter

    def _build_src(
        self,
        phase,
        harm_oscillator_params,
        noise_generator_params,
        noise_filter_params,
        aperiodicity_filter_params,
        voicing,
    ):
        harm_osc = self.harm_oscillator(phase, *harm_oscillator_params)
        if voicing is not None:
            assert torch.all(voicing >= 0) and torch.all(voicing <= 1)
            voicing = F.threshold(voicing, 0.5, 0)
            harm_osc = harm_osc * voicing

        noise = self.noise_filter(
            self.noise_generator(harm_osc, *noise_generator_params),
            *noise_filter_params,
        )
        # aperiodicity crossfade: same response A applied to both branches so the
        # harmonic gets (1 - A) and the noise gets A, exactly per frequency band.
        harm_ap = self.aperiodicity_filter(harm_osc, *aperiodicity_filter_params)
        noise_ap = self.aperiodicity_filter(noise, *aperiodicity_filter_params)
        return (harm_osc - harm_ap) + noise_ap

    def forward(
        self,
        phase: AudioTensor,
        harm_oscillator_params: Tuple[AudioTensor, ...],
        noise_generator_params: Tuple[AudioTensor, ...],
        noise_filter_params: Tuple[AudioTensor, ...],
        aperiodicity_filter_params: Tuple[AudioTensor, ...],
        end_filter_params: Tuple[AudioTensor, ...],
        voicing: Optional[AudioTensor] = None,
        target: Optional[AudioTensor] = None,
        **other_params
    ) -> AudioTensor:
        src = self._build_src(
            phase, harm_oscillator_params, noise_generator_params,
            noise_filter_params, aperiodicity_filter_params, voicing,
        )
        if target is not None:
            src, target_src = self.end_filter.reverse(src, target, *end_filter_params)
            return src, target_src
        return self.room_filter(self.end_filter(src, *end_filter_params))

    def excitation(
        self,
        phase: AudioTensor,
        harm_oscillator_params: Tuple[AudioTensor, ...],
        noise_generator_params: Tuple[AudioTensor, ...],
        noise_filter_params: Tuple[AudioTensor, ...],
        aperiodicity_filter_params: Tuple[AudioTensor, ...],
        voicing: Optional[AudioTensor] = None,
        **other_params
    ) -> AudioTensor:
        return self._build_src(
            phase, harm_oscillator_params, noise_generator_params,
            noise_filter_params, aperiodicity_filter_params, voicing,
        )
