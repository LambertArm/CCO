# Contributing

## Main Rule

Changes should be easy to compare against the previous transform version.

Right now that means comparing against version `0.0.0`.

## Typical Contribution

1. implement or refine a transform in `kernels/src/transform/`
2. update any shared registry wiring if needed
3. run benchmarks: `bench/scripts/bench.sh`
4. run eval: `bash scripts/eval.sh`
5. include the results in the PR

## PR Bot Flow

1. every PR update triggers CPU-only precheck
2. failing precheck PRs are auto-closed before they consume GPU time
3. passing precheck PRs get added to the ordered GPU eval queue
4. when a GPU machine is available, run the queue worker to evaluate queued PRs one by one in arrival order

## Scope Guidance

- put attention-specific kernels in `kernels/src/attention/`
- put transform evolution in `kernels/src/transform/`
- put correctness-oriented CPU truth code in `runtime/src/reference/`
- put orchestration and pipeline logic in `runtime/src/`

CCO currently targets matrix-only transforms applied directly to attention.
