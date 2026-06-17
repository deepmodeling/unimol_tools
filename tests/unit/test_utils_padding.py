import torch

from unimol_tools.utils.util import pad_1d_tokens, pad_2d, pad_coords


def test_pad_1d_tokens_right_and_multiple():
    values = [torch.tensor([1, 2]), torch.tensor([3])]

    result = pad_1d_tokens(values, pad_idx=0, pad_to_multiple=4)

    assert result.tolist() == [[1, 2, 0, 0], [3, 0, 0, 0]]


def test_pad_1d_tokens_left_pad_to_length():
    values = [torch.tensor([1, 2]), torch.tensor([3])]

    result = pad_1d_tokens(values, pad_idx=-1, left_pad=True, pad_to_length=3)

    assert result.tolist() == [[-1, 1, 2], [-1, -1, 3]]


def test_pad_2d_preserves_square_blocks():
    values = [
        torch.tensor([[1, 2], [3, 4]]),
        torch.tensor([[5]]),
    ]

    result = pad_2d(values, pad_idx=0, pad_to_length=3)

    assert result.shape == (2, 3, 3)
    assert result[0].tolist() == [[1, 2, 0], [3, 4, 0], [0, 0, 0]]
    assert result[1].tolist() == [[5, 0, 0], [0, 0, 0], [0, 0, 0]]


def test_pad_2d_left_pads_feature_matrices():
    values = [
        torch.tensor([[[1, 10], [2, 20]], [[3, 30], [4, 40]]]),
        torch.tensor([[[5, 50]]]),
    ]

    result = pad_2d(values, pad_idx=-1, dim=2, left_pad=True, pad_to_length=3)

    assert result.shape == (2, 3, 3, 2)
    assert result[0, 1:, 1:, :].tolist() == values[0].tolist()
    assert result[1, 2:, 2:, :].tolist() == values[1].tolist()
    assert result[1, :2, :, :].eq(-1).all()


def test_pad_coords_left_pad_and_multiple():
    values = [
        torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        torch.tensor([[7.0, 8.0, 9.0]]),
    ]

    result = pad_coords(values, pad_idx=0.0, left_pad=True, pad_to_multiple=4)

    assert result.shape == (2, 4, 3)
    assert torch.equal(result[0, 2:], values[0])
    assert torch.equal(result[1, 3:], values[1])
    assert result[1, :3].eq(0.0).all()
