import math

import pytest
import torch

from unimol_tools.models.loss import (
    FocalLoss,
    FocalLossWithLogits,
    GHMC_Loss,
    GHMR_Loss,
    MAEwithNan,
    myCrossEntropyLoss,
)
from unimol_tools.utils.dynamic_loss_scaler import DynamicLossScaler


def test_mae_with_nan_ignores_nan_targets():
    pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    target = torch.tensor([[2.0, float("nan")], [1.0, 4.0]])

    loss = MAEwithNan(pred, target)

    assert torch.isclose(loss, torch.tensor(1.0))


def test_focal_loss_matches_manual_formula():
    pred = torch.tensor([0.9, 0.2])
    target = torch.tensor([1.0, 0.0])
    alpha = 0.25
    gamma = 2.0

    loss = FocalLoss(pred, target, alpha=alpha, gamma=gamma)

    two_class_target = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    two_class_pred = torch.tensor([[0.1, 0.9], [0.8, 0.2]]).clamp(1e-5, 1.0)
    expected = -alpha * two_class_target * (1 - two_class_pred).pow(gamma) * two_class_pred.log()
    assert torch.isclose(loss, expected.sum(dim=1).mean())


def test_focal_loss_with_logits_ignores_nan_targets():
    logits = torch.tensor([10.0, -10.0, 0.0])
    target = torch.tensor([1.0, 0.0, float("nan")])

    loss = FocalLossWithLogits(logits, target)

    assert torch.isfinite(loss)
    assert loss.item() < 1e-4


def test_cross_entropy_loss_flattens_targets():
    logits = torch.tensor([[3.0, 0.1], [0.2, 2.0]])
    target = torch.tensor([[0], [1]])

    loss = myCrossEntropyLoss(logits, target)

    assert torch.isclose(loss, torch.nn.CrossEntropyLoss()(logits, target.flatten()))


def test_ghm_losses_are_finite_and_update_state():
    x = torch.tensor([[0.0], [1.0], [-1.0]])
    cls_target = torch.tensor([[0.0], [1.0], [0.0]])
    reg_target = torch.tensor([[0.5], [0.0], [-0.5]])

    ghmc = GHMC_Loss(bins=5, alpha=0.5)
    ghmr = GHMR_Loss(bins=5, alpha=0.5, mu=0.02)

    first = ghmc(x, cls_target)
    second = ghmc(x, cls_target)
    reg_loss = ghmr(x, reg_target)

    assert torch.isfinite(first)
    assert torch.isfinite(second)
    assert torch.isfinite(reg_loss)
    assert ghmc._last_bin_count is not None
    assert ghmr._last_bin_count is not None


def test_dynamic_loss_scaler_scales_and_unscales_gradients():
    scaler = DynamicLossScaler(init_scale=8.0)
    param = torch.nn.Parameter(torch.tensor([2.0]))
    param.grad = torch.tensor([16.0])

    scaled = scaler.scale(torch.tensor(0.5))
    scaler.unscale_([param])

    assert torch.isclose(scaled, torch.tensor(4.0))
    assert torch.isclose(param.grad, torch.tensor([2.0]))


def test_dynamic_loss_scaler_grows_after_window():
    scaler = DynamicLossScaler(init_scale=2.0, scale_factor=2.0, scale_window=1)

    scaler.update()

    assert scaler.loss_scale == 4.0


def test_dynamic_loss_scaler_reduces_and_reports_overflow():
    scaler = DynamicLossScaler(init_scale=8.0, scale_factor=2.0, tolerance=0.0)

    with pytest.raises(OverflowError, match="setting loss scale to: 4.0"):
        scaler.check_overflow(float("inf"))

    assert scaler.loss_scale == 4.0


def test_dynamic_loss_scaler_minimum_raises_floating_point_error():
    scaler = DynamicLossScaler(
        init_scale=1.0,
        scale_factor=2.0,
        tolerance=0.0,
        min_loss_scale=0.75,
    )

    with pytest.raises(FloatingPointError, match="Minimum loss scale reached"):
        scaler.check_overflow(math.nan)

    assert scaler.loss_scale == 1.0
