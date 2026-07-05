# CCO Bench

`bench/` holds raw benchmark entry points and shape definitions.

This is the measurement layer, not the policy layer:

- `bench/` answers: what happened in one run?
- `eval/` answers: is this version acceptable compared with the baseline?

The active benchmark contract is flexible transformed attention: `(n, d) -> (n', d') -> attention -> (n, d)`.

## Current Layout

```text
bench/
├── README.md
├── bench_transform.cpp
├── bench_attention.cpp
├── bench_pipeline.cpp
├── shapes.yaml
└── scripts/
```

## Commands

```bash
bench/scripts/bench.sh
bench/scripts/accuracy.sh
```

The current benchmark binaries are lightweight, but this layout is ready for direct transformed-attention measurements.

`bench_attention.cpp` and `bench_pipeline.cpp` now exercise the active transformed-attention path.
