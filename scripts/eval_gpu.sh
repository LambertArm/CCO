#!/usr/bin/env bash
set -euo pipefail

python3 eval/run_eval_gpu.py
python3 eval/compare_versions.py --report eval/reports/gpu-latest.json
