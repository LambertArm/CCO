# CCO Eval

`eval/` is the correctness and regression layer for CCO.

It is intentionally modeled after the role `eval/` plays in `sparkinfer`, but scaled down to fit CCO's current scope:

- one matrix transform version: `0.0.0`
- GEMM-only compute path
- kernel/runtime split

## Current Layout

```text
eval/
├── README.md
├── eval.py
├── run_eval.py
├── compare_versions.py
├── thresholds.yaml
├── baselines/
│   └── 0.0.0.json
└── reports/
```

## Commands

```bash
python3 eval/run_eval.py
python3 eval/compare_versions.py
```

## Output

The current scaffold writes:

- `eval/reports/latest.json`
- `eval/reports/latest.md`

As the kernels mature, this should evolve toward:

- exact GEMM correctness checks
- transform reconstruction error
- latency and memory measurements
- baseline vs current regression gating
