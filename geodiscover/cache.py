"""Noiseless forward caching with explicit model-quantization semantics."""

from __future__ import annotations

import hashlib
import io
import json
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, TypeVar, cast

import numpy as np

from .constants import DEFAULT_CACHE_SIZE, DEFAULT_CACHE_TTL
from .logging import get_logger
from .resilience import BackendUnavailable, CircuitBreaker
from .types import Forward, JsonMap, Vector

if TYPE_CHECKING:
    from .core import Problem


T = TypeVar("T")
logger = get_logger(__name__)


class CacheBackend(Protocol):
    """Synchronous byte storage used by the solve cache."""

    def get(self, key: str) -> bytes | None:
        """Read a serialized forward response."""
        ...

    def set(self, key: str, value: bytes, *, ex: int) -> object:
        """Write a response with a server-side expiration."""
        ...

    def delete(self, *keys: str) -> int:
        """Invalidate specific response keys."""
        ...


class ForwardCache:
    """Cache trusted solves, falling back locally when Redis is unavailable."""

    def __init__(
        self,
        solver: Forward,
        identity: JsonMap,
        quantization: float | None = None,
        redis_client: CacheBackend | None = None,
        max_entries: int = DEFAULT_CACHE_SIZE,
        ttl_seconds: int = DEFAULT_CACHE_TTL,
        breaker: CircuitBreaker | None = None,
    ) -> None:
        if quantization is not None and (not np.isfinite(quantization) or quantization <= 0):
            raise ValueError("quantization must be positive. use None for exact solve keys")
        if max_entries < 1 or ttl_seconds < 1:
            raise ValueError("cache capacity and TTL must be positive")
        self.solver, self.quantization, self.redis = solver, quantization, redis_client
        self.identity = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)
        self.namespace = hashlib.sha256(self.identity.encode()).hexdigest()
        self.max_entries, self.ttl = max_entries, ttl_seconds
        self.breaker = breaker or CircuitBreaker()
        self.memory: OrderedDict[str, tuple[float, Vector]] = OrderedDict()
        self.hits = self.misses = self.backend_failures = 0

    def _key(self, model: Vector) -> tuple[str, Vector]:
        representative = np.asarray(model, dtype="<f8")
        if representative.ndim != 1 or not np.isfinite(representative).all():
            raise ValueError("model must be a finite vector")
        if self.quantization is not None:
            representative = np.rint(representative / self.quantization) * self.quantization
        representative = np.ascontiguousarray(representative, dtype="<f8")
        digest = hashlib.sha256(
            str(representative.shape).encode()
            + str(self.quantization).encode()
            + representative.tobytes()
        ).hexdigest()
        return f"geodiscover:forward:v2:{self.namespace}:{digest}", representative

    def _backend(self, operation: Callable[[], T]) -> T:
        transient: tuple[type[Exception], ...] = (OSError, ConnectionError, TimeoutError)
        if self.redis is not None:
            try:
                from redis.exceptions import ConnectionError as RedisConnectionError
                from redis.exceptions import TimeoutError as RedisTimeoutError

                transient += (RedisConnectionError, RedisTimeoutError)
            except ImportError:
                pass
        return self.breaker.call(operation, transient)

    def solve_uncached(self, model: Vector) -> Vector:
        """Solve the same representative without consulting either cache tier."""
        _, representative = self._key(model)
        value = np.asarray(self.solver(representative), float)
        if value.ndim != 1 or not np.isfinite(value).all():
            raise ValueError("solver response must be a finite observation vector")
        return value.copy()

    def __call__(self, model: Vector) -> Vector:
        key, representative = self._key(model)
        cached = self.memory.get(key)
        if cached is not None and cached[0] > time.monotonic():
            self.memory.move_to_end(key)
            self.hits += 1
            return cached[1].copy()
        self.memory.pop(key, None)
        backend = self.redis
        if backend is not None:
            try:
                payload = self._backend(lambda: backend.get(key))
                if payload is not None:
                    value = np.asarray(
                        np.load(io.BytesIO(payload), allow_pickle=False), dtype=float
                    )
                    if value.ndim != 1 or not np.isfinite(value).all():
                        raise ValueError("cached response must be a finite vector")
                    self.hits += 1
                    self._remember(key, value)
                    return value.copy()
            except (BackendUnavailable, ValueError, EOFError) as exc:
                self.backend_failures += 1
                logger.warning("cache read bypassed", extra={"fields": {"error": str(exc)}})
        self.misses += 1
        value = np.asarray(self.solver(representative), dtype=float)
        if value.ndim != 1 or not np.isfinite(value).all():
            raise ValueError("solver response must be a finite observation vector")
        self._remember(key, value)
        if backend is not None:
            buffer = io.BytesIO()
            np.save(buffer, value, allow_pickle=False)
            try:
                self._backend(lambda: backend.set(key, buffer.getvalue(), ex=self.ttl))
            except BackendUnavailable as exc:
                self.backend_failures += 1
                logger.warning("cache write bypassed", extra={"fields": {"error": str(exc)}})
        return value.copy()

    def _remember(self, key: str, value: Vector) -> None:
        self.memory[key] = (time.monotonic() + self.ttl, value.copy())
        self.memory.move_to_end(key)
        while len(self.memory) > self.max_entries:
            self.memory.popitem(last=False)

    def invalidate(self, model: Vector) -> None:
        """Discard a model response locally and in the configured backend."""
        key, _ = self._key(model)
        self.memory.pop(key, None)
        backend = self.redis
        if backend is not None:
            try:
                self._backend(lambda: backend.delete(key))
            except BackendUnavailable as exc:
                self.backend_failures += 1
                logger.warning(
                    "cache invalidation deferred to TTL", extra={"fields": {"error": str(exc)}}
                )

    def stats(self) -> JsonMap:
        """Return actual cache accesses and backend failures."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "backend_failures": self.backend_failures,
            "entries": len(self.memory),
            "quantization": self.quantization,
            "ttl_seconds": self.ttl,
        }


def cached_linear_forward(
    problem: Problem, quantization: float | None = None, redis_url: str | None = None
) -> ForwardCache:
    """Build a linear solve cache whose key includes every operator coefficient."""
    backend = None
    if redis_url:
        from redis import Redis

        backend = cast(
            CacheBackend, Redis.from_url(redis_url, socket_timeout=1, socket_connect_timeout=1)
        )
    return ForwardCache(
        lambda m: problem.operator @ m + problem.offset,
        {
            "kind": "linear",
            "operator": problem.operator.tolist(),
            "offset": problem.offset.tolist(),
            "numpy_version": np.__version__,
        },
        quantization,
        backend,
    )
