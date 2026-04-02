"""Prometheus measurements for acquisition decisions and fallback paths."""

from __future__ import annotations

from dataclasses import dataclass

from prometheus_client import CollectorRegistry, Counter, Histogram


@dataclass
class Metrics:
    """An isolated metric registry, suitable for one service instance."""

    registry: CollectorRegistry
    requests: Counter
    failures: Counter
    latency: Histogram

    @classmethod
    def create(cls) -> Metrics:
        """Register decision counters and the request-latency histogram."""
        registry = CollectorRegistry()
        return cls(
            registry,
            Counter(
                "geodiscover_plans_total", "Acquisition decisions", ["mode"], registry=registry
            ),
            Counter(
                "geodiscover_fallbacks_total", "Fallback decisions", ["reason"], registry=registry
            ),
            Histogram(
                "geodiscover_plan_seconds",
                "Decision latency",
                registry=registry,
                buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.14, 0.25, 0.5, 1, 2, 5),
            ),
        )
