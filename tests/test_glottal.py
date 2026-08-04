import pytest
import torch

from aris.models.audiotensor import AudioTensor
from aris.models.synth import (
    GlottalFlowTable,
    IndexedGlottalFlowTable,
    WeightedGlottalFlowTable,
)


@pytest.mark.parametrize("table_type", ["flow", "derivative"])
@pytest.mark.parametrize("normalize_method", [None, "constant_power", "peak"])
@pytest.mark.parametrize("align_peak", [True, False])
def test_glottal_construction(table_type, normalize_method, align_peak):
    glottal = GlottalFlowTable(
        table_type=table_type, normalize_method=normalize_method, align_peak=align_peak
    )
    assert glottal is not None


def test_glottal_forward():
    hop_size = 80
    frames = 201
    phase = AudioTensor(torch.linspace(0.001, 0.005, frames).unsqueeze(0), hop_size)

    indexed = IndexedGlottalFlowTable()
    index_weight = AudioTensor(torch.rand(1, frames), hop_size)
    indexed_output = indexed(phase, index_weight)
    assert indexed_output.shape == (1, (frames - 1) * hop_size + 1)

    weighted = WeightedGlottalFlowTable()
    mixture_weight = AudioTensor(torch.randn(1, frames, 100).softmax(-1), hop_size)
    weighted_output = weighted(phase, mixture_weight)
    assert weighted_output.shape == indexed_output.shape
