"""Cache expiry and Redis outage regressions with an in-memory backend."""

from __future__ import annotations

import numpy as np
import pytest

from geodiscover.cache import ForwardCache
from geodiscover.resilience import CircuitBreaker
from geodiscover.types import Vector


class MemoryBackend:
    """Byte storage with observable calls and an injected connection outage."""

    def __init__(self) -> None:
        self.values: dict[str, bytes] = {}
        self.reads = self.writes = self.deletes = 0
        self.online = True
        self.ttls: list[int] = []

    def get(self, key: str) -> bytes | None:
        self.reads += 1
        if not self.online:
            raise ConnectionError("controlled Redis outage")
        return self.values.get(key)

    def set(self, key: str, value: bytes, *, ex: int) -> bool:
        self.writes += 1
        if not self.online:
            raise ConnectionError("controlled Redis outage")
        self.values[key] = value
        self.ttls.append(ex)
        return True

    def delete(self, *keys: str) -> int:
        self.deletes += 1
        removed = sum(key in self.values for key in keys)
        for key in keys:
            self.values.pop(key, None)
        return removed


def test_memory_response_expires_at_the_ttl_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [100.0]
    monkeypatch.setattr("geodiscover.cache.time.monotonic", lambda: clock[0])
    calls: list[Vector] = []

    def solve(model: Vector) -> Vector:
        calls.append(model.copy())
        return model * 2

    cache = ForwardCache(solve, {"operator": "twice"}, ttl_seconds=2)
    model = np.array([1.0])
    np.testing.assert_array_equal(cache(model), [2.0])
    cache(model)
    assert len(calls) == 1
    clock[0] = 102.0
    cache(model)
    assert len(calls) == 2
    assert cache.stats()["hits"] == 1
    assert cache.stats()["misses"] == 2


def test_invalidate_removes_both_local_and_backend_responses() -> None:
    backend = MemoryBackend()
    calls: list[Vector] = []

    def solve(model: Vector) -> Vector:
        calls.append(model.copy())
        return model * 3

    model = np.array([2.0])
    cache = ForwardCache(solve, {"operator": "triple"}, redis_client=backend, ttl_seconds=7)
    cache(model)
    assert backend.ttls == [7]
    cache.invalidate(model)
    assert backend.values == {}
    assert backend.deletes == 1
    cache(model)
    assert len(calls) == 2
    assert cache.misses == 2


def test_new_worker_reads_a_serialized_backend_response() -> None:
    backend = MemoryBackend()
    model = np.array([2.0])
    first = ForwardCache(lambda m: m * 4, {"operator": "four"}, redis_client=backend)
    first(model)

    def unexpected_solve(_model: Vector) -> Vector:
        raise AssertionError("a valid Redis response should avoid a second solve")

    second = ForwardCache(unexpected_solve, {"operator": "four"}, redis_client=backend)
    np.testing.assert_array_equal(second(model), [8.0])
    assert second.hits == 1 and second.misses == 0


def test_transient_reads_retry_with_exponential_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    delays: list[float] = []
    monkeypatch.setattr("geodiscover.resilience.time.sleep", delays.append)

    class FlakyBackend(MemoryBackend):
        def get(self, key: str) -> bytes | None:
            if self.reads < 2:
                self.reads += 1
                raise ConnectionError("controlled temporary outage")
            return super().get(key)

    backend = FlakyBackend()
    breaker = CircuitBreaker(attempts=3, delay_seconds=0.01)
    cache = ForwardCache(
        lambda m: m, {"operator": "identity"}, redis_client=backend, breaker=breaker
    )
    np.testing.assert_array_equal(cache(np.array([1.0])), [1.0])
    assert backend.reads == 3
    assert delays == [0.01, 0.02]
    assert cache.backend_failures == 0


def test_open_circuit_bypasses_redis_and_recovers_after_its_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [100.0]
    monkeypatch.setattr("geodiscover.resilience.time.monotonic", lambda: clock[0])
    backend = MemoryBackend()
    backend.online = False
    breaker = CircuitBreaker(threshold=1, attempts=1, recovery_seconds=10, delay_seconds=0)
    cache = ForwardCache(
        lambda m: m * 2, {"operator": "twice"}, redis_client=backend, breaker=breaker
    )
    np.testing.assert_array_equal(cache(np.array([1.0])), [2.0])
    np.testing.assert_array_equal(cache(np.array([2.0])), [4.0])
    assert backend.reads == 1
    assert cache.backend_failures >= 2
    assert cache.misses == 2
    clock[0] = 111.0
    backend.online = True
    cache(np.array([3.0]))
    assert backend.reads == 2
    assert backend.writes == 1
    assert breaker.failures == 0


def test_corrupt_backend_bytes_are_replaced_by_a_fresh_solve() -> None:
    backend = MemoryBackend()
    model = np.array([1.0])
    first = ForwardCache(lambda m: m, {"operator": "identity"}, redis_client=backend)
    first(model)
    key = next(iter(backend.values))
    backend.values[key] = b"not a numpy forward result"
    fresh = ForwardCache(lambda m: m, {"operator": "identity"}, redis_client=backend)
    np.testing.assert_array_equal(fresh(model), model)
    assert fresh.backend_failures == 1
    assert fresh.misses == 1
