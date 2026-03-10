"""Forward-only timing must compare the same discretized physics."""

from __future__ import annotations

import numpy as np

from geodiscover.cache import ForwardCache
from geodiscover.cache_benchmark import compare_forward_cache
from geodiscover.types import Vector


def test_cache_comparison_counts_real_calls_and_separates_approximation_error() -> None:
    calls = []

    def solve(model: Vector) -> Vector:
        calls.append(model.copy())
        return model**2

    cache = ForwardCache(solve, {"kind": "square"}, quantization=0.5)
    report = compare_forward_cache(cache, [np.array([0.26]), np.array([0.74])], repeats=2)
    assert report["calls_per_condition"] == 4
    assert report["forward_cache_delta"] == {"hits": 3, "misses": 1, "backend_failures": 0}
    assert len(calls) == 9  # four diagnostic calls, four uncached calls, one cache miss
    assert report["quantization_prediction_rmse"] > 0
    assert report["speedup"] > 0
    assert report["order"] == ["uncached", "cached"]


def test_equal_sums_do_not_hide_corrupt_cached_candidate_responses() -> None:
    import pytest

    model = np.array([1.0, 2.0])
    cache = ForwardCache(lambda value: value, {"kind": "identity"})
    cache(model)
    key = next(iter(cache.memory))
    expires, _ = cache.memory[key]
    cache.memory[key] = (expires, np.array([2.0, 1.0]))
    with pytest.raises(ValueError, match="cached responses disagree"):
        compare_forward_cache(cache, [model])
