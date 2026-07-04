# Cuda-Compute-OSS

CCO is a CUDA-first open source repo for matrix multiply optimization.

The repo is organized around two primary subsystems:

- `kernels/` holds the versioned CUDA transform and compute kernels.
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
├── scripts/
└── archive/legacy-python/
```

## First Version

The repo currently keeps only the first transform version in `kernels/src/transform/`.

- `0.0.0` implemented in `00_baseline.cu`

Future versions can be added later, but the repo now stays intentionally focused on a single baseline transform so the kernel/runtime structure is clean from the start.

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
- [`kernels/README.md`](kernels/README.md)
- [`runtime/README.md`](runtime/README.md)

## Legacy Prototype

The previous Python prototype is preserved in [`archive/legacy-python/`](archive/legacy-python/) so earlier experiments remain available while the repo transitions to the CUDA-first layout.
