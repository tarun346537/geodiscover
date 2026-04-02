"""Measure real HTTP acquisition throughput, latency, failures and fallbacks."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import httpx
import numpy as np

from .config import Settings
from .constants import LOAD_CONCURRENCY, LOAD_REQUESTS, LOAD_WARMUP
from .logging import configure_logging, get_logger
from .types import JsonMap

logger = get_logger(__name__)


async def run_load_test(
    client: httpx.AsyncClient,
    payload: JsonMap,
    requests: int = LOAD_REQUESTS,
    concurrency: int = LOAD_CONCURRENCY,
    warmup: int = LOAD_WARMUP,
) -> JsonMap:
    """Measure HTTP decisions, errors, latency and fallback frequency."""
    if requests < 1 or concurrency < 1 or warmup < 0:
        raise ValueError("positive requests/concurrency and nonnegative warmup required")
    warmup_errors = 0
    for _ in range(warmup):
        try:
            response = await client.post("/plan", json=payload)
            warmup_errors += int(response.status_code != 200)
        except httpx.HTTPError:
            warmup_errors += 1
    next_index = 0
    latencies: list[float] = []
    statuses: Counter[str] = Counter()
    errors: Counter[str] = Counter()
    modes: Counter[str] = Counter()

    async def worker() -> None:
        """Issue assigned requests through a shared HTTP connection pool."""
        nonlocal next_index
        while next_index < requests:
            next_index += 1
            started = time.perf_counter()
            try:
                response = await client.post("/plan", json=payload)
                statuses[str(response.status_code)] += 1
                if response.status_code == 200:
                    try:
                        data = response.json()
                        action, mode = data.get("action"), data.get("mode")
                        if (
                            type(action) is not int
                            or action < 0
                            or mode not in ("mcts", "policy", "greedy_bayesian_fallback")
                            or action in payload.get("selected", [])
                            or (
                                payload.get("problem")
                                and action >= len(payload["problem"]["operator"])
                            )
                        ):
                            raise ValueError("unexpected response schema")
                        modes[data["mode"]] += 1
                    except (ValueError, TypeError, AttributeError):
                        errors["invalid_response"] += 1
                else:
                    errors["http_error"] += 1
            except httpx.HTTPError as exc:
                errors[type(exc).__name__] += 1
            latencies.append(time.perf_counter() - started)

    started = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(min(concurrency, requests))))
    elapsed = time.perf_counter() - started
    successes = sum(modes.values())
    fallbacks = modes.get("greedy_bayesian_fallback", 0)
    return {
        "schema_version": 1,
        "kind": "http_load_test",
        "generated_at": datetime.now(UTC).isoformat(),
        "requests": requests,
        "concurrency": concurrency,
        "warmup_requests": warmup,
        "warmup_errors": warmup_errors,
        "elapsed_seconds": elapsed,
        "requests_per_second": len(latencies) / elapsed,
        "successful_decisions_per_second": successes / elapsed,
        "successful_decisions": successes,
        "error_count": sum(errors.values()),
        "errors_by_kind": dict(errors),
        "status_codes": dict(statuses),
        "modes": dict(modes),
        "fallback_count": fallbacks,
        "fallback_fraction": fallbacks / successes if successes else None,
        "latency_seconds_p50": float(np.quantile(latencies, 0.5)),
        "latency_seconds_p95": float(np.quantile(latencies, 0.95)),
        "latency_seconds_p99": float(np.quantile(latencies, 0.99)),
        "latency_seconds_max": max(latencies),
        "payload_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
        "timing_scope": "Client scheduling and HTTP round trips across all attempted requests.",
    }


def main() -> None:
    """Run the command-line workflow and write its result artifacts."""
    settings = Settings.load()
    configure_logging(settings.log_level)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=settings.api_url)
    parser.add_argument("--payload", type=Path, default=Path("configs/request-small.json"))
    parser.add_argument("--requests", type=int, default=LOAD_REQUESTS)
    parser.add_argument("--concurrency", type=int, default=LOAD_CONCURRENCY)
    parser.add_argument("--warmup", type=int, default=LOAD_WARMUP)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--output", type=Path, default=settings.output_dir / "loadtest.json")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    payload = json.loads(args.payload.read_text())

    async def run() -> JsonMap:
        """Run the asynchronous local measurement job."""
        import httpx

        limits = httpx.Limits(
            max_connections=args.concurrency, max_keepalive_connections=args.concurrency
        )
        async with httpx.AsyncClient(
            base_url=args.url, timeout=args.timeout, limits=limits, trust_env=False
        ) as client:
            return await run_load_test(
                client, payload, args.requests, args.concurrency, args.warmup
            )

    report = asyncio.run(run())
    report["target_url"] = args.url
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    logger.info("workflow complete", extra={"fields": report})


if __name__ == "__main__":
    main()
