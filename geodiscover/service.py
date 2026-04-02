"""Bounded FastAPI acquisition service with greedy Bayesian failover."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from contextvars import copy_context
from pathlib import Path
from typing import Literal
from uuid import uuid4

import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, ConfigDict, Field

from .config import Settings
from .constants import DEFAULT_PARTICLES, DEFAULT_WORKERS
from .core import Problem, greedy_action, posterior
from .logging import configure_logging, get_logger, request_id
from .metrics import Metrics
from .planning import plan_acquisition
from .types import BeliefState, JsonMap, Planner

logger = get_logger(__name__)


class ProblemInput(BaseModel):
    """Constrained API representation of a survey."""

    model_config = ConfigDict(extra="forbid")
    operator: list[list[float]] = Field(min_length=1, max_length=128)
    noise_std: list[float] = Field(min_length=1, max_length=128)
    prior_mean: list[float] = Field(min_length=1, max_length=64)
    prior_covariance: list[list[float]] = Field(min_length=1, max_length=64)
    offset: list[float] | None = Field(default=None, max_length=128)

    def problem(self) -> Problem:
        """Validate the API survey and return the numerical planning problem."""
        if any(len(row) > 64 for row in self.operator + self.prior_covariance):
            raise ValueError("model dimension exceeds service limit")
        return Problem(**self.model_dump())


class PlanRequest(BaseModel):
    """Acquisition history, planning strategy and request limits."""

    model_config = ConfigDict(extra="forbid")
    problem: ProblemInput | None = None
    selected: list[int] = Field(default_factory=list, max_length=128)
    observations: list[float] = Field(default_factory=list, max_length=128)
    remaining_budget: int = Field(default=1, ge=1, le=16)
    simulations: int = Field(default=128, ge=1, le=4096)
    timeout_ms: int = Field(default=500, ge=10, le=2000)
    seed: int = Field(default=42, ge=0, le=2147483647)
    strategy: Literal["mcts", "policy"] = "mcts"


def _local_plan(
    problem: Problem,
    belief: BeliefState,
    budget: int,
    simulations: int,
    seed: int,
    deadline: float | None,
) -> tuple[int, JsonMap]:
    return plan_acquisition(problem, belief, budget, simulations, seed, deadline)


def create_app(
    planner: Planner | None = None,
    checkpoint: str | Path | None = None,
    max_workers: int = DEFAULT_WORKERS,
    problem_config: JsonMap | None = None,
    particles: int = DEFAULT_PARTICLES,
) -> FastAPI:
    """Build the decision API with bounded planning workers."""
    planner = planner or _local_plan
    measurements = Metrics.create()
    registry, requests, failures, latency = (
        measurements.registry,
        measurements.requests,
        measurements.failures,
        measurements.latency,
    )
    gate = threading.BoundedSemaphore(max_workers)
    executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="survey-plan")

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        """Release planning workers when the service shuts down."""
        yield
        executor.shutdown(wait=False, cancel_futures=True)
        if hasattr(planner, "close"):
            planner.close()

    app = FastAPI(title="GeoDiscover acquisition planner", version="0.1.0", lifespan=lifespan)
    policy = None
    if checkpoint and Path(checkpoint).is_file():
        from .policy import load_policy

        policy, _ = load_policy(checkpoint)
    app.state.policy = policy
    configured_problem = Problem.from_dict(problem_config) if problem_config else None
    configured_ensemble = None
    if configured_problem is not None and configured_problem.metadata.get("simpeg_dc"):
        from .nonlinear import particle_prior
        from .simpeg_adapter import forward_for_problem

        configured_ensemble = particle_prior(
            configured_problem, forward_for_problem(configured_problem), particles, seed=42
        ).to_ensemble()

    @app.middleware("http")
    async def request_trace(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Attach a request ID and record HTTP completion as structured data."""
        identifier = request.headers.get("x-request-id", uuid4().hex)[:128]
        token = request_id.set(identifier)
        started = time.perf_counter()
        try:
            response = await call_next(request)
            response.headers["x-request-id"] = identifier
            logger.info(
                "http request complete",
                extra={
                    "fields": {
                        "method": request.method,
                        "path": request.url.path,
                        "status": response.status_code,
                        "seconds": time.perf_counter() - started,
                    }
                },
            )
            return response
        finally:
            request_id.reset(token)

    @app.get("/health")
    def health() -> JsonMap:
        """Report survey and policy readiness."""
        return {
            "status": "ok",
            "policy_loaded": policy is not None,
            "solver": "simpeg_particle_belief" if configured_ensemble else "linear_gaussian",
            "configured_survey": configured_problem is not None,
        }

    @app.get("/metrics")
    def metrics() -> Response:
        """Expose Prometheus counters and latency histograms."""
        return Response(content=generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    @app.post("/plan")
    async def plan(request: PlanRequest) -> JsonMap:
        """Validate a request and return a primary or fallback acquisition."""
        started = time.monotonic()
        try:
            problem = request.problem.problem() if request.problem else configured_problem
            if problem is None:
                raise ValueError("provide a problem or configure GEO_PROBLEM_JSON")
            if not request.problem and configured_ensemble is not None:
                from .nonlinear import particle_posterior

                belief: BeliefState = particle_posterior(
                    problem, configured_ensemble, request.selected, request.observations
                )
            else:
                belief = posterior(problem, request.selected, request.observations)
            if request.remaining_budget > problem.n_actions - len(belief.selected):
                raise ValueError("budget exceeds remaining measurements")
        except (ValueError, TypeError, np.linalg.LinAlgError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        reason: str | None = None
        statistics: JsonMap = {}
        mode: str = request.strategy
        if not gate.acquire(blocking=False):
            reason = "workers_busy"
        else:

            def compute() -> tuple[int, JsonMap]:
                """Run the configured planner within its worker slot."""
                try:
                    if request.strategy == "policy":
                        if policy is None:
                            raise LookupError("policy checkpoint is not configured")
                        return policy.choose(problem, belief, request.remaining_budget), {}
                    return planner(
                        problem,
                        belief,
                        request.remaining_budget,
                        request.simulations,
                        request.seed,
                        started + request.timeout_ms / 1000,
                    )
                finally:
                    gate.release()

            future = asyncio.get_running_loop().run_in_executor(
                executor, copy_context().run, compute
            )
            try:
                action, statistics = await asyncio.wait_for(
                    asyncio.shield(future), request.timeout_ms / 1000
                )
                if isinstance(action, (bool, np.bool_)) or not isinstance(
                    action, (int, np.integer)
                ):
                    raise ValueError("planner action must be a non-boolean integer")
                action = int(action)
                if action in belief.selected or not 0 <= action < problem.n_actions:
                    raise ValueError("planner returned unavailable action")
            except TimeoutError:
                reason = "timeout"
                # Retrieve eventual exceptions. Cooperative MCTS stops at its
                # deadline. A hung custom worker holds its bounded slot rather
                # than allowing an unbounded queue of abandoned tasks.
                future.add_done_callback(
                    lambda done: done.exception() if not done.cancelled() else None
                )
            except Exception:
                reason = "planner_failure"
        if reason:
            action = greedy_action(problem, belief)
            mode = "greedy_bayesian_fallback"
            failures.labels(reason).inc()
        requests.labels(mode).inc()
        latency.observe(time.monotonic() - started)
        return {
            "action": int(action),
            "mode": mode,
            "fallback_reason": reason,
            "expected_information_gain": float(belief.information_gains(problem)[action]),
            "statistics": statistics,
            "provenance": "live_request",
        }

    return app


def _configured_app() -> FastAPI:
    settings = Settings.load()
    configure_logging(settings.log_level)
    planner = None
    if settings.ray_address:
        from .parallel import RayPlanner

        planner = RayPlanner(settings.ray_address)
    problem_config = (
        json.loads(settings.problem_path.read_text()) if settings.problem_path else None
    )
    return create_app(
        planner=planner,
        checkpoint=settings.checkpoint,
        max_workers=settings.workers,
        problem_config=problem_config,
        particles=settings.particles,
    )


app = _configured_app()
