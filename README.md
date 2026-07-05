# Cuda-Compute-OSS

CCO is a CUDA-first open source repo for transformed attention optimization.

The repo is organized around a transform-first attention workflow:

- `kernels/` holds the CUDA transform and attention kernels.
- `runtime/` holds host orchestration, pipeline logic, and CPU references.
- `bench/` measures raw performance.
- `eval/` runs the real GPU-based evaluation system.
- `eval/` also includes a lightweight CPU precheck for local non-GPU machines.
- GitHub workflows gate PRs with CPU precheck first and then feed accepted PRs into an ordered GPU eval queue.

The current structure is intentionally inspired by two ideas:

- `leimao/CUDA-GEMM-Optimization` for explicit version-by-version progress
- `gittensor-ai-lab/sparkinfer` for engineering discipline around bench and eval

## Repository Layout

```text
Cuda-Compute-OSS/
├── docs/
├── kernels/
│   ├── include/cco/
│   └── src/
├── runtime/
│   ├── include/cco/
│   └── src/
├── tests/
├── bench/
├── eval/
├── examples/
└── scripts/
```

## First Version

The repo currently keeps only the first transform version in `kernels/src/transform/`.

- `0.0.0` implemented in `00_baseline.cu`

Future versions can be added later, but the repo now stays intentionally focused on a single transform baseline and one direct transformed-attention pipeline.

## Development Order

CCO follows this order:

1. transform `Q`, `K`, and `V` from `(n, d)` into `(n', d')`
2. run attention in the reduced space
3. reconstruct the output back to `(n, d)`
4. compare the transformed result against exact attention on GPU

`(n, d)` and `(n', d')` are intentionally flexible. The baseline requires only that `(n', d')` cleanly divide `(n, d)`.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build
```

## Documentation

- [`docs/overview.md`](docs/overview.md)
- [`docs/transform-ladder.md`](docs/transform-ladder.md)
- [`docs/evaluation.md`](docs/evaluation.md)
- [`docs/contributing.md`](docs/contributing.md)
- [`docs/gpu-runner-setup.md`](docs/gpu-runner-setup.md)
- [`bench/README.md`](bench/README.md)
- [`eval/README.md`](eval/README.md)
- [`kernels/README.md`](kernels/README.md)
- [`runtime/README.md`](runtime/README.md)

## Current Scope

CCO currently tracks one baseline transform version, one transformed-attention path, and a kernel/runtime split designed for versioned optimization work.

The authoritative evaluation path is GPU-based: exact attention vs transformed attention, bounded accuracy, relative error, latency, and memory are all measured on GPU in `eval/run_eval_gpu.py`.

By default, the GPU eval runs 10 random Q/K/V trials per shape and dtype, scores them one by one, and sums those trial scores into the final result.

For local non-GPU development, `eval/run_eval.py` is only a lightweight precheck.

For a single automatic entrypoint, run `bash scripts/eval.sh`. It chooses CPU precheck on CPU-only machines and real GPU eval on CUDA machines.

For pull requests, the repo now uses a two-stage bot flow:

1. CPU-only precheck on every PR update
2. ordered GPU queue processing later on a rented or self-hosted GPU machine

To prepare a rented GPU machine for the queue worker, use:

```bash
bash scripts/setup_gpu_runner.sh
```

Default attention shapes are `n=4096`, `d=256`, `n'=1024`, and `d'=64`, but any valid transform target can be configured.

If `n'` and `d'` are omitted, CCO now asks the active transform version for its default transformed-shape policy.
