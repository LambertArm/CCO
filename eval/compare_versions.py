#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare a current CCO eval report against a baseline.")
    parser.add_argument("--baseline", default="eval/baselines/0.0.0.json")
    parser.add_argument("--report", default="eval/reports/latest.json")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    baseline = load_json(repo_root / args.baseline)
    report = load_json(repo_root / args.report)

    print("CCO compare_versions")
    print(f"baseline transform: {baseline.get('transform_version', 'unknown')}")
    print(f"report transform:   {report.get('transform_version', 'unknown')}")
    print(f"baseline verdict:   {baseline.get('verdict', 'unknown')}")
    print(f"report verdict:     {report.get('verdict', 'unknown')}")
    print(f"shape count:        {len(report.get('shapes', []))}")


if __name__ == "__main__":
    main()
