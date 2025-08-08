from __future__ import annotations
from collections import defaultdict
from typing import Dict, List, Optional


class MetricsAggregator:
    """Minimal metrics logger inspired by Uni-Core's metrics module.

    It accumulates scalar values with optional weights and allows fetching
    averaged results. The implementation is intentionally lightweight and
    only provides the subset of functionality required by UniMol pretraining.
    """

    def __init__(self) -> None:
        self.meters: Dict[str, List[float]] = defaultdict(lambda: [0.0, 0.0])

    def log_scalar(self, key: str, value: float, weight: float = 1.0, round: Optional[int] = None) -> None:
        """Log a scalar value.

        Args:
            key: Metric name.
            value: Observed value.
            weight: Contribution weight for averaging.
            round: Unused, present for API compatibility.
        """
        meter = self.meters[key]
        meter[0] += float(value) * weight
        meter[1] += weight

    def get_smoothed_values(self) -> Dict[str, float]:
        """Return averaged metrics and reset internal storage."""
        result = {k: (v[0] / v[1] if v[1] else 0.0) for k, v in self.meters.items()}
        return result

    def reset(self) -> None:
        self.meters.clear()


# Expose a module-level singleton similar to unicore.metrics
metrics = MetricsAggregator()