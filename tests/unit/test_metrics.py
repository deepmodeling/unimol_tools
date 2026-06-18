import numpy as np
import torch

from unimol_tools.utils.metrics import (
    cal_nan_metric,
    multi_acc,
    log_loss_with_label,
    reg_preasonr,
    reg_spearmanr,
    Metrics,
)


def test_cal_nan_metric_ignores_nan():
    y_true = np.array([[1.0, np.nan], [2.0, 3.0]])
    y_pred = np.array([[1.5, 0.0], [2.5, 4.0]])
    mse = lambda a, b: ((a - b) ** 2).mean()
    res = cal_nan_metric(y_true, y_pred, metric_func=mse)
    assert np.isclose(res, 0.625)


def test_basic_metric_functions():
    assert np.isclose(reg_preasonr([1, 2], [1, 2]), 1.0)
    assert np.isclose(reg_spearmanr([1, 2], [1, 2]), 1.0)
    y_true = np.array([[0], [1], [2]])
    y_pred = np.array([[0.1, 0.9, 0.0], [0.2, 0.6, 0.2], [0.1, 0.2, 0.7]])
    assert np.isclose(multi_acc(y_true, y_pred), 2 / 3)
    ll = log_loss_with_label([0, 1], [[0.8, 0.2], [0.2, 0.8]], labels=[0, 1])
    assert ll >= 0.0


def test_metrics_classification_threshold():
    metric = Metrics('classification', metrics_str='acc')
    target = np.array([[0], [1], [1], [0]])
    pred = np.array([[0.2], [0.8], [0.9], [0.1]])
    th = metric.calculate_single_classification_threshold(target, pred, step=5)
    assert np.isclose(th, 0.3)


def test_metrics_calculation():
    cls = Metrics('classification', metrics_str='acc')
    out = cls.cal_metric(np.array([[1], [0]]), np.array([[0.6], [0.3]]))
    assert 'acc' in out and out['acc'] == 1.0

    reg = Metrics('regression', metrics_str='mae')
    out = reg.cal_metric(np.array([[1.0], [2.0]]), np.array([[1.5], [2.0]]))
    assert 'mae' in out and np.isclose(out['mae'], 0.25)
