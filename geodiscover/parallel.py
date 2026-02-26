"""Optional Ray planner callable with remote timeout cancellation."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

from .planning import plan_acquisition
from .types import BeliefState, JsonMap

if TYPE_CHECKING:
    from .core import Problem


def _remote_plan(
    problem: Problem,
    belief: BeliefState,
    budget: int,
    simulations: int,
    seed: int,
    timeout_seconds: float,
) -> tuple[int, JsonMap]:
    return plan_acquisition(
        problem, belief, budget, simulations, seed, time.monotonic() + timeout_seconds
    )


class RayPlanner:
    """Dispatch planning to cancellable Ray worker tasks."""

    def __init__(self, address: str = "auto") -> None:
        import ray

        self.ray = ray
        self.owns_runtime = not ray.is_initialized()
        if self.owns_runtime:
            ray.init(
                address=address,
                include_dashboard=False,
                runtime_env={"py_modules": [str(Path(__file__).parent)]},
            )
        self.task = ray.remote(num_cpus=1)(_remote_plan)

    def __call__(
        self,
        problem: Problem,
        belief: BeliefState,
        budget: int,
        simulations: int,
        seed: int,
        deadline: float,
    ) -> tuple[int, JsonMap]:
        timeout = deadline - time.monotonic()
        if timeout <= 0:
            raise TimeoutError("deadline expired before remote submission")
        ref = self.task.remote(problem, belief, budget, simulations, seed, timeout)
        try:
            return self.ray.get(ref, timeout=timeout)
        except Exception:
            self.ray.cancel(ref, force=True)
            raise

    def close(self) -> None:
        """Stop the Ray runtime owned by this planner."""
        if self.owns_runtime:
            self.ray.shutdown()
