import torch

from models.audiotensor import AudioTensor


def test_basic():
    x = AudioTensor(torch.randn(1, 100), hop_length=10)
    assert x.size() == (1, 100)
    assert x.hop_length == 10

    x1 = x.reduce_hop_length()
    assert isinstance(x1, AudioTensor), x1
    assert x1.shape == (1, 991)
    assert x1.hop_length == 1

    x2 = x.reduce_hop_length(5)
    assert isinstance(x2, AudioTensor), x2
    assert x2.shape == (1, 496)
    assert x2.hop_length == 2

    x3 = x1 + x2 * x
    assert isinstance(x3, AudioTensor)
    assert x3.shape == (1, 991)
    assert x3.hop_length == 1
