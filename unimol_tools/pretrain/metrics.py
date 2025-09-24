"""Lightweight metrics aggregation inspired by Uni-Core.

This module provides a small subset of Uni-Core's ``logging.metrics``
functionality. It supports hierarchical aggregation contexts via a context
manager and allows scalar values to be logged with optional weights.  Logged
values can later be retrieved as averages for a given aggregation name.

The implementation is intentionally minimal but keeps the same public API used
throughout Uni-Mol so existing training code can rely on the familiar
``metrics.log_scalar`` and ``metrics.aggregate`` helpers.
"""

from __future__ import annotations

import contextlib
import uuid
from collections import OrderedDict
from typing import Dict, Iterable


class _AverageMeter:
    """Tracks a weighted average for a single metric."""

    def __init__(self) -> None:
        self.sum = 0.0
        self.count = 0.0

    def update(self, value: float, weight: float) -> None:
        self.sum += float(value) * weight
        self.count += weight

    def get_value(self) -> float:
        return self.sum / self.count if self.count else 0.0

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0.0


class _Aggregator:
    """Container holding meters for a particular aggregation context."""

    def __init__(self) -> None:
        self.meters: Dict[str, _AverageMeter] = {}

    def log_scalar(self, key: str, value: float, weight: float) -> None:
        meter = self.meters.setdefault(key, _AverageMeter())
        meter.update(value, weight)

    def get_smoothed_values(self) -> Dict[str, float]:
        return {k: m.get_value() for k, m in self.meters.items()}

    def reset(self) -> None:
        for m in self.meters.values():
            m.reset()


# ---------------------------------------------------------------------------
# Global registry of aggregators and utilities mirroring Uni-Core's API
# ---------------------------------------------------------------------------
_aggregators: "OrderedDict[str, _Aggregator]" = OrderedDict()
_active_aggregators: "OrderedDict[str, _Aggregator]" = OrderedDict()


def reset() -> None:
    """Reset all metrics aggregators and create the default context."""
    _aggregators.clear()
    _active_aggregators.clear()
    default = _Aggregator()
    _aggregators["default"] = default
    _active_aggregators["default"] = default


reset()


@contextlib.contextmanager
def aggregate(name: str | None = None, new_root: bool = False):
    """Context manager to aggregate metrics under ``name``.

    Args:
        name: Aggregation key. If ``None`` a temporary aggregator is created.
        new_root: If ``True`` this context becomes the new root, ignoring any
            previously active aggregators.
    """

    if name is None:
        name = str(uuid.uuid4())
        assert name not in _aggregators
        agg = _Aggregator()
        temp_name = True
    else:
        agg = _aggregators.setdefault(name, _Aggregator())
        temp_name = False

    if new_root:
        backup = _active_aggregators.copy()
        _active_aggregators.clear()

    _active_aggregators[name] = agg
    try:
        yield agg
    finally:
        _active_aggregators.pop(name, None)
        if temp_name:
            _aggregators.pop(name, None)
        if new_root:
            _active_aggregators.clear()
            _active_aggregators.update(backup)


def _iter_active() -> Iterable[_Aggregator]:
    return _active_aggregators.values()


def log_scalar(key: str, value: float, weight: float = 1.0, round: int | None = None) -> None:
    """Log a scalar value to all active aggregators."""

    for agg in _iter_active():
        agg.log_scalar(key, value, weight)


def get_smoothed_values(name: str) -> Dict[str, float]:
    """Return the averaged metrics for the given aggregation name."""

    if name not in _aggregators:
        return {}
    return _aggregators[name].get_smoothed_values()


def reset_meters(name: str) -> None:
    """Reset meters for the specified aggregation."""

    if name in _aggregators:
        _aggregators[name].reset()


# ---------------------------------------------------------------------------
# Compatibility layer so callers can ``from metrics import metrics`` and use
# attribute access (e.g. ``metrics.log_scalar``) like in Uni-Core.
# ---------------------------------------------------------------------------
class _MetricsProxy:
    aggregate = staticmethod(aggregate)
    log_scalar = staticmethod(log_scalar)
    get_smoothed_values = staticmethod(get_smoothed_values)
    reset_meters = staticmethod(reset_meters)
    reset = staticmethod(reset)


metrics = _MetricsProxy()


__all__ = [
    "aggregate",
    "get_smoothed_values",
    "log_scalar",
    "reset",
    "reset_meters",
    "metrics",
]