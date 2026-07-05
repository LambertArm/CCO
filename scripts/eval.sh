#!/usr/bin/env bash
set -euo pipefail

if python3 - <<'PY'
import sys
try:
    import torch
    sys.exit(0 if torch.cuda.is_available() else 1)
except Exception:
    sys.exit(1)
PY
then
    echo "CCO eval: GPU detected, running real GPU eval"
    python3 eval/run_eval_gpu.py
    python3 eval/compare_versions.py --report eval/reports/gpu-latest.json
else
    echo "CCO eval: no GPU detected, running CPU precheck only"
    python3 eval/run_eval.py
fi
