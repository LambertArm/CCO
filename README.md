# Cuda-Compute-OSS

CCO is a CUDA-first open source repo for GEMM optimization and matrix-only transform strategies.

The repo is organized around two primary subsystems:

- `kernels/` holds the CUDA matrix transform implementation and GEMM kernels.
- `runtime/` holds host orchestration, pipeline logic, and CPU references.
- `bench/` measures raw performance.
- `eval/` runs the real GPU-based evaluation system.
- `eval/` also includes a lightweight CPU precheck for local non-GPU machines.
- GitHub workflows run eval automatically and can comment results on pull requests.

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

The repo currently keeps only the first matrix transform version in `kernels/src/transform/`.

- `0.0.0` implemented in `00_baseline.cu`

Future versions can be added later, but the repo now stays intentionally focused on a single matrix transform baseline and GEMM-only compute path so the structure is clean from the start.

## Development Order

CCO follows this order:

1. improve GEMM with mathematical matrix transform methods
2. evaluate the transform directly against exact GEMM
3. only after the GEMM path is useful, reuse it inside attention

An attention area exists under `kernels/src/attention/`, but it is secondary and intentionally scaffolded for later integration.

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
- [`bench/README.md`](bench/README.md)
- [`eval/README.md`](eval/README.md)
- [`kernels/README.md`](kernels/README.md)
- [`runtime/README.md`](runtime/README.md)

## Current Scope

CCO currently tracks one baseline matrix transform version, one GEMM-focused compute path, and a kernel/runtime split designed for future optimization work. Attention is present only as a future consumer of the GEMM work.

The authoritative evaluation path is GPU-based: exact GEMM vs transformed GEMM, bounded accuracy, relative error, latency, and memory are all measured on GPU in `eval/run_eval_gpu.py`.

For local non-GPU development, `eval/run_eval.py` is only a lightweight precheck.
