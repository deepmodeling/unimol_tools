import numpy as np
import torch
from unimol_tools.models.nnmodel import NNModel


def test_collect_data_with_dict():
    X = {"a": np.arange(5), "b": np.arange(5) * 2}
    y = np.arange(5)
    idx = np.array([0, 2, 4])
    x_out, y_out = NNModel.collect_data(None, X, y, idx)
    assert isinstance(x_out, dict)
    assert np.array_equal(x_out["a"], X["a"][idx])
    assert np.array_equal(x_out["b"], X["b"][idx])
    assert torch.equal(y_out, torch.tensor(y[idx]))