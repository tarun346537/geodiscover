"""Train and export a search-distilled acquisition policy."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import torch

from .config import Settings
from .constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_BUDGET,
    DEFAULT_EPOCHS,
    DEFAULT_PARTICLES,
    DEFAULT_PATIENCE,
    DEFAULT_RL_EPISODES,
    DEFAULT_SEED,
)
from .core import Problem
from .distillation import distill as distill
from .distillation import imitation_loss as imitation_loss
from .logging import configure_logging, get_logger
from .policy import AcquisitionPolicy
from .reinforcement import refine_rl as refine_rl
from .tracking import RunTracker, save_checkpoint
from .train_data import belief_for_row as belief_for_row
from .train_data import file_sha256 as file_sha256
from .train_data import model_fingerprint
from .train_data import read_models as read_models
from .train_data import read_plans as read_plans
from .types import JsonMap

logger = get_logger(__name__)


def main() -> None:
    """Run imitation learning, simulator refinement and portable model export."""
    settings = Settings.load()
    configure_logging(settings.log_level)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plans", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--rl-episodes", type=int, default=DEFAULT_RL_EPISODES)
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--problem", type=Path)
    parser.add_argument("--models", type=Path)
    parser.add_argument("--redis-url", default=settings.redis_url)
    parser.add_argument("--cache-quantization", type=float)
    parser.add_argument("--backend", choices=("auto", "linear", "particle"), default="auto")
    parser.add_argument("--particles", type=int, default=DEFAULT_PARTICLES)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--track", type=Path)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    if args.models and not args.problem:
        parser.error("--models requires --problem")
    torch.manual_seed(args.seed)
    torch.set_num_threads(settings.cpu_threads)
    rows = read_plans(args.plans)
    if (
        args.rl_episodes
        and not args.problem
        and any(
            row["problem"].get("metadata", {}).get("kind") != "synthetic_potential_field_proxy"
            for row in rows
        )
    ):
        parser.error("configured-survey plans require --problem for simulator refinement")
    problem = Problem.from_dict(json.loads(args.problem.read_text())) if args.problem else None
    models = None
    validation_models = None
    if args.models:
        assert problem is not None
        models = read_models(args.models, problem)
        validation_models = read_models(args.models, problem, split="validation")
    checkpoint_dir = args.checkpoint_dir or args.output.parent / f"{args.output.stem}-state"
    tracker = RunTracker(
        args.track or args.output.with_suffix(".steps.jsonl"), args.run_id or args.output.stem
    )
    policy = AcquisitionPolicy().to(args.device)
    distillation_path, rl_path = (
        checkpoint_dir / "distillation.pt",
        checkpoint_dir / "refinement.pt",
    )
    history = distill(
        policy,
        rows,
        args.epochs,
        seed=args.seed,
        batch_size=args.batch_size,
        checkpoint_path=distillation_path,
        resume=args.resume,
        patience=args.patience,
        tracker=tracker,
    )
    refinement = refine_rl(
        policy,
        args.rl_episodes,
        args.budget,
        args.seed + 1000000,
        problem=problem,
        models=models,
        redis_url=args.redis_url,
        backend=args.backend,
        particles=args.particles,
        cache_quantization=args.cache_quantization,
        checkpoint_path=rl_path,
        resume=args.resume and rl_path.is_file(),
        patience=args.patience,
        validation_models=validation_models,
        tracker=tracker,
    )
    checkpoint: JsonMap = {
        "schema_version": 1,
        "kind": "geodiscover_acquisition_policy",
        "provenance": "new_training_run",
        "seed": args.seed,
        "config": {
            k: str(v) if isinstance(v, Path) else v
            for k, v in vars(args).items()
            if k != "redis_url"
        },
        "torch_version": str(torch.__version__),
        "device": args.device,
        "generated_at": datetime.now(UTC).isoformat(),
        "hidden_size": policy.hidden_size,
        "state_dict": {key: value.cpu() for key, value in policy.state_dict().items()},
        "plans_sha256": file_sha256(args.plans),
        "model_dataset_sha256": file_sha256(args.models) if args.models else None,
        "training_model_sha256": [model_fingerprint(model) for model in models or []],
        "validation_model_sha256": [model_fingerprint(model) for model in validation_models or []],
        "distillation": history,
        "rl": refinement,
    }
    save_checkpoint(args.output, checkpoint)
    report = {key: value for key, value in checkpoint.items() if key != "state_dict"}
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    logger.info(
        "policy training complete",
        extra={
            "fields": {
                "checkpoint": str(args.output),
                "epochs_completed": len(history),
                "rl_episodes_completed": len(refinement),
            }
        },
    )


if __name__ == "__main__":
    main()
