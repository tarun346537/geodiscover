"""Measured epoch timing and optional NVML-backed device telemetry."""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Self

from .constants import UTILIZATION_POLL_SECONDS
from .types import JsonMap

if TYPE_CHECKING:
    import torch

    from .cache import ForwardCache


class EpochProfiler:
    """Collect measured phase timing and optional CUDA telemetry."""

    def __init__(
        self, device: str | torch.device = "cpu", sample_seconds: float = UTILIZATION_POLL_SECONDS
    ) -> None:
        self.device = str(device)
        self.sample_seconds = sample_seconds
        self.result: JsonMap = {}
        self.samples: list[float] = []
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None

    def __enter__(self) -> Self:
        import torch

        self.torch = torch
        self.cuda = self.device.startswith("cuda") and torch.cuda.is_available()
        self.gpu_error = None
        if self.cuda:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)

            def sample() -> None:
                """Sample the current device utilization."""
                while not self.stop.is_set():
                    try:
                        self.samples.append(float(torch.cuda.utilization(self.device)))
                    except Exception as exc:
                        self.gpu_error = type(exc).__name__
                        break
                    self.stop.wait(self.sample_seconds)

            self.thread = threading.Thread(target=sample, name="gpu-util-sampler", daemon=True)
            self.thread.start()
        self.started, self.cpu_started = time.perf_counter(), time.process_time()
        return self

    def __exit__(self, *_exc: object) -> None:
        if self.cuda:
            self.torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - self.started
        cpu = time.process_time() - self.cpu_started
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=1)
        self.result.update(
            wall_seconds=elapsed,
            process_cpu_seconds=cpu,
            device=self.device,
            gpu_utilization_percent_mean=sum(self.samples) / len(self.samples)
            if self.samples
            else None,
            gpu_utilization_samples=len(self.samples),
            peak_cuda_allocated_bytes=int(self.torch.cuda.max_memory_allocated(self.device))
            if self.cuda
            else None,
            telemetry_source="device-wide NVML utilization samples"
            if self.samples
            else (f"GPU telemetry unavailable: {self.gpu_error}" if self.cuda else "CPU execution"),
        )


def cache_delta(cache: ForwardCache, before: JsonMap) -> JsonMap:
    """Measure accesses since the previous cache snapshot."""
    stats = cache.stats()
    return {k: stats[k] - before.get(k, 0) for k in ("hits", "misses", "backend_failures")}
