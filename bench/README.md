# CCO Bench

`bench/` holds raw benchmark entry points and shape definitions.

This is the measurement layer, not the policy layer:

- `bench/` answers: what happened in one run?
- `eval/` answers: is this version acceptable compared with the baseline?

## Current Layout

```text
bench/
├── README.md
├── bench_transform.cpp
├── bench_compute.cpp
├── bench_pipeline.cpp
├── shapes.yaml
└── scripts/
```

## Commands

```bash
bench/scripts/bench.sh
bench/scripts/accuracy.sh
```

The current benchmark binaries are still scaffolds, but this layout is ready for the first real GEMM and matrix-transform measurements.
