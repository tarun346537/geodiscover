"""Streaming teacher-plan exports and measured search accounting."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from .types import JsonMap


def write_plans(
    path: str | Path, batches: Iterable[tuple[int, list[JsonMap]]], config: JsonMap
) -> JsonMap:
    """Version 2 JSONL stores each scenario/ensemble once, then its plan states."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    report: JsonMap = {
        "schema_version": 1,
        "kind": "teacher_export",
        "config": config,
        "generated_at": datetime.now(UTC).isoformat(),
        "episodes": 0,
        "plans": 0,
        "nodes_expanded": 0,
        "search_simulations": 0,
        "worker_seconds": 0.0,
        "forward_cache": {"hits": 0, "misses": 0, "backend_failures": 0},
    }
    started = time.perf_counter()
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for _, rows in batches:
            first = rows[0]
            header = {
                k: first[k]
                for k in ("scenario_id", "split", "provenance", "problem", "belief_backend")
            }
            header.update(schema_version=2, record_type="scenario")
            if "particle_ensemble" in first:
                header["particle_ensemble"] = first["particle_ensemble"]
            records = [header]
            for row in rows:
                record = {
                    k: row[k]
                    for k in (
                        "scenario_id",
                        "selected",
                        "observations",
                        "remaining_budget",
                        "teacher_action",
                        "search",
                    )
                }
                record.update(schema_version=2, record_type="plan")
                records.append(record)
                report["plans"] += 1
                report["nodes_expanded"] += row["search"]["nodes_expanded"]
                report["search_simulations"] += row["search"]["simulations"]
                report["worker_seconds"] += row["search"]["worker_seconds"]
                for k, value in row["search"]["forward_cache_delta"].items():
                    report["forward_cache"][k] += value
            report["episodes"] += 1
            for record in records:
                line = json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n"
                stream.write(line)
                digest.update(line.encode("utf-8"))
    report["wall_seconds"] = time.perf_counter() - started
    report["plans_sha256"] = digest.hexdigest()
    path.with_suffix(".run.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report
