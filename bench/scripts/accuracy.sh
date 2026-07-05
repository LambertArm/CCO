#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/_common.sh"
ROOT="$(repo_root)"

python3 "$ROOT/eval/run_eval.py"
