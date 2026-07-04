#!/usr/bin/env bash
set -euo pipefail

./build/cco_bench_transform
./build/cco_bench_compute
./build/cco_bench_pipeline
