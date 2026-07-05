# Cuda-Compute-OSS

CCO is a CUDA-first open source repo for GEMM optimization and matrix-only transform strategies.

The repo is organized around two primary subsystems:

- `kernels/` holds the CUDA matrix transform implementation and GEMM kernels.
- `runtime/` holds host orchestration, pipeline logic, and CPU references.
- `bench/` measures raw performance.
- `eval/` compares versions with stable rules.

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

CCO currently tracks one baseline matrix transform version, one GEMM-focused compute path, and a kernel/runtime split designed for future optimization work.
