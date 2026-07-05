#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

from score import summary_score


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare a current CCO eval report against a baseline.")
    parser.add_argument("--baseline", default="eval/baselines/0.0.0.json")
    parser.add_argument("--report", default="eval/reports/gpu-latest.json")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    baseline = load_json(repo_root / args.baseline)
    report = load_json(repo_root / args.report)
    baseline_score = summary_score(baseline)
    report_score = summary_score(report)

    print("CCO compare_versions")
    print(f"baseline transform: {baseline.get('transform_version', 'unknown')}")
    print(f"report transform:   {report.get('transform_version', 'unknown')}")
    print(f"baseline verdict:   {baseline.get('verdict', 'unknown')}")
    print(f"report verdict:     {report.get('verdict', 'unknown')}")
    print(f"case count:         {len(report.get('cases', []))}")
    print(f"baseline score:     {baseline_score:.8f}")
    print(f"report score:       {report_score:.8f}")


if __name__ == "__main__":
    main()
