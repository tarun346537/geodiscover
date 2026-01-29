"""Bounded backend retries and circuit-breaker state."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from .constants import (
    BACKEND_ATTEMPTS,
    BACKEND_DELAY_SECONDS,
    CIRCUIT_RECOVERY_SECONDS,
    CIRCUIT_THRESHOLD,
)

T = TypeVar("T")


class BackendUnavailable(RuntimeError):
    """A backend exhausted its retry budget or its circuit remains open."""


@dataclass
class CircuitBreaker:
    """Stop repeated calls while an unavailable backend recovers."""

    threshold: int = CIRCUIT_THRESHOLD
    recovery_seconds: float = CIRCUIT_RECOVERY_SECONDS
    attempts: int = BACKEND_ATTEMPTS
    delay_seconds: float = BACKEND_DELAY_SECONDS
    failures: int = 0
    open_until: float = 0.0

    def __post_init__(self) -> None:
        if (
            self.threshold < 1
            or self.attempts < 1
            or self.delay_seconds < 0
            or self.recovery_seconds < 0
        ):
            raise ValueError("backend retry limits must be positive and delays nonnegative")

    def call(self, operation: Callable[[], T], transient: tuple[type[Exception], ...]) -> T:
        """Retry transient failures, opening the circuit after repeated outages."""
        if time.monotonic() < self.open_until:
            raise BackendUnavailable("backend circuit is open. using a local solve")
        for attempt in range(self.attempts):
            try:
                result = operation()
                self.failures = 0
                return result
            except transient as exc:
                if attempt + 1 < self.attempts:
                    time.sleep(self.delay_seconds * 2**attempt)
                else:
                    self.failures += 1
                    if self.failures >= self.threshold:
                        self.open_until = time.monotonic() + self.recovery_seconds
                    raise BackendUnavailable(
                        "backend retries exhausted. check the Redis connection"
                    ) from exc
        raise BackendUnavailable("backend retry budget must be positive")
