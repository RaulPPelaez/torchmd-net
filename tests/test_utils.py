from os.path import join, exists
from pytest import mark, raises
import pytest
import torch
from torchmdnet.utils import make_splits
from torchmdnet.models.utils import CUDAGraphModule

def sum_lengths(*args):
    return sum(map(len, args))


def test_make_splits_outputs():
    result = make_splits(100, 0.7, 0.2, 0.1, 1234)
    assert len(result) == 3
    assert isinstance(result[0], torch.Tensor)
    assert isinstance(result[1], torch.Tensor)
    assert isinstance(result[2], torch.Tensor)
    assert len(result[0]) == 70
    assert len(result[1]) == 20
    assert len(result[2]) == 10
    assert sum_lengths(*result) == len(torch.unique(torch.cat(result)))
    assert max(map(max, result)) == 99
    assert min(map(min, result)) == 0


@mark.parametrize("dset_len", [5, 1000])
@mark.parametrize("ratio1", [0.0, 0.3])
@mark.parametrize("ratio2", [0.0, 0.3])
@mark.parametrize("ratio3", [0.0, 0.3])
def test_make_splits_ratios(dset_len, ratio1, ratio2, ratio3):
    train, val, test = make_splits(dset_len, ratio1, ratio2, ratio3, 1234)
    assert sum_lengths(train, val, test) <= dset_len
    assert len(train) == round(ratio1 * dset_len)
    assert len(val) == round(ratio2 * dset_len)
    # simply multiplying and rounding ratios can lead to values larger than dset_len,
    # which make_splits should account for by removing one sample from the test set
    if (
        round(ratio1 * dset_len) + round(ratio2 * dset_len) + round(ratio3 * dset_len)
        > dset_len
    ):
        assert len(test) == round(ratio3 * dset_len) - 1
    else:
        assert len(test) == round(ratio3 * dset_len)


def test_make_splits_sizes():
    assert sum_lengths(*make_splits(100, 70, 20, 10, 1234)) == 100
    assert sum_lengths(*make_splits(100, 70, 20, None, 1234)) == 100
    assert sum_lengths(*make_splits(100, 70, None, 10, 1234)) == 100
    assert sum_lengths(*make_splits(100, None, 20, 10, 1234)) == 100
    assert sum_lengths(*make_splits(100, 70, 20, 0.1, 1234)) == 100
    assert sum_lengths(*make_splits(100, 70, 20, 0.05, 1234)) == 95


def test_make_splits_save_load(tmpdir):
    path = join(tmpdir, "splits.npz")
    train, val, test = make_splits(100, 0.7, 0.2, 0.1, 1234, filename=path)
    assert exists(path)
    trainl, vall, testl = make_splits(None, None, None, None, None, splits=path)
    assert (train == trainl).all() and (val == vall).all() and (test == testl).all()


def test_make_splits_order():
    train, val, test = make_splits(
        100, 0.7, 0.2, 0.1, 1234, order=torch.arange(100, 0, -1, dtype=torch.int)
    )
    assert (train == torch.arange(100, 30, -1, dtype=torch.int)).all()
    assert (val == torch.arange(30, 10, -1, dtype=torch.int)).all()
    assert (test == torch.arange(10, 0, -1, dtype=torch.int)).all()


def test_make_splits_errors():
    with raises(AssertionError):
        make_splits(100, 0.5, 0.5, 0.5, 1234)
    with raises(AssertionError):
        make_splits(100, 50, 50, 50, 1234)
    with raises(AssertionError):
        make_splits(100, None, None, 5, 1234)
    with raises(AssertionError):
        make_splits(100, 60, 60, None, 1234)


def test_cuda_graph_module():
    # Skip if CUDA is not available
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    class GraphableModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            pass
        def forward(self, x):
            return 2*x
    module = GraphableModule()
    x = torch.randn(10, requires_grad=True, device="cuda")
    y = module(x)
    graph_module = CUDAGraphModule(module, fallback_to_eager=False)
    ygraph = graph_module(x)
    assert torch.allclose(ygraph, y)



def test_cuda_graph_module_with_fallback():
    # Skip if CUDA is not available
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    class UnGraphableModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            pass
        def forward(self, x):
            max_val = torch.max(x).cpu()
            torch.cuda.synchronize()
            return x*max_val
    module = UnGraphableModule()
    x = torch.randn(10, requires_grad=True, device='cuda')
    y = module(x)
    graph_module = CUDAGraphModule(module, fallback_to_eager=True)
    assert torch.allclose(graph_module(x), y)
