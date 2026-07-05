#!/usr/bin/env bash
set -euo pipefail

python3 eval/run_eval.py
python3 eval/compare_versions.py
