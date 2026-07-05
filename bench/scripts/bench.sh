#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/_common.sh"
ROOT="$(repo_root)"

"$ROOT/build/cco_bench_transform"
"$ROOT/build/cco_bench_compute"
"$ROOT/build/cco_bench_pipeline"
