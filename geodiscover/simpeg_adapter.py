"""Optional 2.5D DC forward solver and finite-difference local linearization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import ArrayLike

from .cache import CacheBackend, ForwardCache
from .config import Settings
from .core import Problem
from .logging import configure_logging, get_logger
from .types import JsonMap, Vector

logger = get_logger(__name__)


class SimPEGDC:
    """Model vector is natural-log electrical conductivity per mesh cell.

    Each candidate is one ABMN dipole acquisition. Coordinates have (x, z)
    components on a tensor mesh. The highest z mesh boundary is the surface.
    The tensor mesh has a flat surface and all cells active. Prepare survey
    coordinates and field quality checks before constructing the configuration.
    """

    def __init__(self, config: JsonMap) -> None:
        import simpeg
        from discretize import TensorMesh
        from pymatsolver import SolverLU
        from simpeg import maps
        from simpeg.electromagnetics.static import resistivity as dc

        self.config = json.loads(json.dumps(config, allow_nan=False))
        # Backend credentials are not physics settings or shareable metadata.
        self.config.pop("redis_url", None)
        widths = [np.asarray(config[k], float) for k in ("hx", "hz")]
        if any(x.ndim != 1 or len(x) < 3 or np.any(x <= 0) for x in widths):
            raise ValueError("positive hx/hz mesh widths with >=3 cells required")
        self.mesh = TensorMesh(widths, x0=np.asarray(config["origin"], float))
        if self.mesh.nC > 10000:
            raise ValueError("explicitly review adapter resource limits before using >10000 cells")
        sources = []
        for item in config["candidates"]:
            coords = {k: np.asarray(item[k], float) for k in ("a", "b", "m", "n")}
            if any(v.shape != (2,) or not np.isfinite(v).all() for v in coords.values()):
                raise ValueError("ABMN coordinates must be finite (x,z) pairs")
            receiver = dc.receivers.Dipole(
                locations_m=coords["m"][None, :], locations_n=coords["n"][None, :], data_type="volt"
            )
            sources.append(
                dc.sources.Dipole(
                    [receiver],
                    location_a=coords["a"],
                    location_b=coords["b"],
                    current=float(item.get("current", 1)),
                )
            )
        if not sources:
            raise ValueError("at least one DC candidate required")
        survey = dc.Survey(sources)
        self.simulation = dc.Simulation2DCellCentered(
            self.mesh,
            survey=survey,
            sigmaMap=maps.ExpMap(nP=self.mesh.nC),
            solver=SolverLU,
            nky=int(config.get("nky", 7)),
            bc_type=config.get("boundary_condition", "Robin"),
        )
        identity = {
            "kind": "simpeg_dc_2_5d",
            "version": simpeg.__version__,
            "config": self.config,
            "mapping": "ExpMap",
            "solver": "pymatsolver.SolverLU",
        }
        redis_client = None
        if config.get("redis_url"):
            from redis import Redis

            redis_client = Redis.from_url(
                config["redis_url"], socket_timeout=1, socket_connect_timeout=1
            )
        self.forward = ForwardCache(
            self._solve,
            identity,
            config.get("quantization"),
            cast(CacheBackend | None, redis_client),
        )

    def _solve(self, model: Vector) -> Vector:
        if model.shape != (self.mesh.nC,):
            raise ValueError("log-conductivity model must have one entry per mesh cell")
        return np.asarray(self.simulation.dpred(model), float)

    def linearize(
        self,
        reference: Vector,
        noise_std: ArrayLike,
        prior_std: float = 0.25,
        epsilon: float = 1e-3,
    ) -> Problem:
        """Central differences cost 2*n_cells+1 forward solves. Use small meshes."""
        if epsilon <= 0 or not np.isfinite(epsilon) or prior_std <= 0:
            raise ValueError("positive finite epsilon and prior_std required")
        if self.forward.quantization is not None:
            raise ValueError("disable quantization for derivative construction")
        reference = np.asarray(reference, float)
        baseline = self.forward(reference)
        columns = []
        for i in range(reference.size):
            delta = np.zeros_like(reference)
            delta[i] = epsilon
            columns.append(
                (self.forward(reference + delta) - self.forward(reference - delta)) / (2 * epsilon)
            )
        operator = np.column_stack(columns)
        return Problem(
            operator,
            noise_std,
            reference,
            np.eye(reference.size) * prior_std**2,
            baseline - operator @ reference,
            {
                "kind": "simpeg_dc_local_linearization",
                "simpeg_dc": self.config,
                "reference": reference.tolist(),
                "finite_difference_epsilon": epsilon,
            },
        )


def forward_for_problem(
    problem: Problem, redis_url: str | None = None, quantization: float | None = None
) -> ForwardCache:
    """Build a cached simulator matching the survey configuration."""
    config = problem.metadata.get("simpeg_dc")
    if config:
        if redis_url:
            config = dict(config, redis_url=redis_url)
        if quantization is not None:
            config = dict(config, quantization=quantization)
        return SimPEGDC(config).forward
    from .cache import cached_linear_forward

    return cached_linear_forward(problem, quantization=quantization, redis_url=redis_url)


def main() -> None:
    """Run the command-line workflow and write its result artifacts."""
    settings = Settings.load()
    configure_logging(settings.log_level)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    adapter = SimPEGDC(config)
    reference = np.full(adapter.mesh.nC, np.log(config.get("reference_conductivity", 0.01)))
    problem = adapter.linearize(reference, config["noise_std"], config.get("prior_std", 0.25))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(problem.to_dict(), indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
