#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import yaml


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CCO CPU precheck only. Real eval is GPU-based.")
    parser.add_argument("--report-path", default="eval/reports/precheck.json")
    parser.add_argument("--markdown-path", default="eval/reports/precheck.md")
    args = parser.parse_args()

    root = repo_root()
    thresholds = load_yaml(root / "eval" / "thresholds.yaml")
    shapes = load_yaml(root / "bench" / "shapes.yaml")
    baseline_exists = (root / "eval" / "baselines" / "0.0.0.json").exists()

    report = {
        "repo": "Cuda-Compute-OSS",
        "mode": "cpu-precheck",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "thresholds_loaded": bool(thresholds),
        "shapes_loaded": bool(shapes),
        "baseline_exists": baseline_exists,
        "verdict": "pass" if thresholds and shapes and baseline_exists else "fail",
        "notes": [
            "CPU precheck validates repo configuration only.",
            "Real transform evaluation and matrix computation are GPU-based in run_eval_gpu.py.",
        ],
    }

    report_path = root / args.report_path
    markdown_path = root / args.markdown_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)

    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown = "\n".join(
        [
            "# CCO CPU Precheck",
            "",
            f"- verdict: `{report['verdict']}`",
            f"- thresholds loaded: `{report['thresholds_loaded']}`",
            f"- shapes loaded: `{report['shapes_loaded']}`",
            f"- baseline exists: `{report['baseline_exists']}`",
        ]
    )
    markdown_path.write_text(markdown + "\n", encoding="utf-8")

    print("CCO cpu precheck")
    print(f"json report: {report_path}")
    print(f"markdown report: {markdown_path}")
    print(f"verdict: {report['verdict']}")


if __name__ == "__main__":
    main()
