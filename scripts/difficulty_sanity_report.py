"""
Run repeated deterministic baseline trials and report per-task score stability.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline.baseline_inference import BaselineAgent


Z_95 = 1.96


def confidence_interval_95(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    mu = mean(values)
    if len(values) < 2:
        return {"mean": mu, "std": 0.0, "ci95_low": mu, "ci95_high": mu}
    sigma = stdev(values)
    margin = Z_95 * (sigma / math.sqrt(len(values)))
    return {
        "mean": mu,
        "std": sigma,
        "ci95_low": mu - margin,
        "ci95_high": mu + margin,
    }


def build_report(
    trial_outputs: List[Dict[str, Any]],
    task_ids: List[str],
    trials: int,
    provider: str,
    model: str,
    temperature: float,
    base_seed: int,
) -> Dict[str, Any]:
    per_task_scores: Dict[str, List[float]] = {task_id: [] for task_id in task_ids}
    trial_averages: List[float] = []

    for output in trial_outputs:
        trial_averages.append(float(output.get("average_score", 0.0)))
        for task_id in task_ids:
            score = float(output["results"][task_id]["score"])
            per_task_scores[task_id].append(score)

    task_stats = {
        task_id: confidence_interval_95(scores)
        for task_id, scores in per_task_scores.items()
    }

    ordering = sorted(
        task_stats.items(),
        key=lambda item: item[1]["mean"],
        reverse=True,
    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "trials": trials,
            "provider": provider,
            "model": model,
            "temperature": temperature,
            "base_seed": base_seed,
            "task_ids": task_ids,
        },
        "overall": confidence_interval_95(trial_averages),
        "per_task": task_stats,
        "mean_ordering_desc": [task_id for task_id, _ in ordering],
    }


def print_report(report: Dict[str, Any]) -> None:
    cfg = report["config"]
    print("=" * 72)
    print("Difficulty Sanity Report")
    print("=" * 72)
    print(
        f"Trials={cfg['trials']} | Provider={cfg['provider']} | "
        f"Model={cfg['model']} | Temp={cfg['temperature']} | Seed={cfg['base_seed']}"
    )
    print()
    overall = report["overall"]
    print(
        "Overall average score: "
        f"{overall['mean']:.3f} (95% CI [{overall['ci95_low']:.3f}, {overall['ci95_high']:.3f}])"
    )
    print()
    print("Per-task score stability:")
    for task_id, stats in report["per_task"].items():
        print(
            f"- {task_id}: mean={stats['mean']:.3f}, std={stats['std']:.3f}, "
            f"95% CI [{stats['ci95_low']:.3f}, {stats['ci95_high']:.3f}]"
        )
    print()
    print("Mean ordering (desc): " + " > ".join(report["mean_ordering_desc"]))
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate difficulty sanity report from repeated baseline trials.")
    parser.add_argument("--provider", type=str, default="groq", choices=["openai", "groq"])
    parser.add_argument("--model", type=str, default="openai/gpt-oss-120b")
    parser.add_argument("--env-url", type=str, default="http://localhost:8000")
    parser.add_argument(
        "--task-ids",
        type=str,
        nargs="+",
        default=[
            "task1_security_basic",
            "task2_quality_logic",
            "task3_advanced_review",
            "task4_session_auth_medium",
            "task5_async_pipeline_hard",
            "task6_data_export_hard",
        ],
    )
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=str,
        default="difficulty_sanity_report.json",
        help="Output path for report JSON",
    )
    args = parser.parse_args()

    trial_outputs: List[Dict[str, Any]] = []
    for i in range(args.trials):
        trial_seed = args.seed + i
        print(f"\nRunning deterministic trial {i + 1}/{args.trials} (seed={trial_seed})")
        agent = BaselineAgent(
            provider=args.provider,
            model=args.model,
            env_base_url=args.env_url,
            temperature=args.temperature,
            seed=trial_seed,
        )
        trial_outputs.append(agent.run_evaluation(args.task_ids))

    report = build_report(
        trial_outputs=trial_outputs,
        task_ids=args.task_ids,
        trials=args.trials,
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
        base_seed=args.seed,
    )

    output_path = Path(args.output)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print_report(report)
    print(f"\nSaved report to {output_path}")


if __name__ == "__main__":
    main()

