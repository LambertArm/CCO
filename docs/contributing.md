# Contributing

## Main Rule

Changes should be easy to compare against the previous transform version.

Right now that means comparing against version `0.0.0`.

## Typical Contribution

1. implement or refine a transform in `kernels/src/transform/`
2. update any shared registry wiring if needed
3. run benchmarks: `bench/scripts/bench.sh`
4. run eval: `python3 eval/run_eval.py`
5. include the results in the PR

## Scope Guidance

- put reusable CUDA compute in `kernels/src/compute/`
- put transform evolution in `kernels/src/transform/`
- put correctness-oriented CPU truth code in `runtime/src/reference/`
- put orchestration and pipeline logic in `runtime/src/`

CCO currently targets matrix-only transforms and GEMM only.
