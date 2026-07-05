# CCO Eval

`eval/` is the real correctness and regression layer for CCO.

It is intentionally modeled after the role `eval/` plays in `sparkinfer`, but adapted to CCO's current scope:

- one matrix transform version: `0.0.0`
- GEMM-only compute path
- kernel/runtime split
- attention treated as a later integration target

## Current Layout

```text
eval/
├── README.md
├── eval.py
├── run_eval.py
├── compare_versions.py
├── score.py
├── thresholds.yaml
├── baselines/
│   └── 0.0.0.json
└── reports/
```

## Commands

```bash
python3 eval/run_eval_gpu.py
python3 eval/compare_versions.py --report eval/reports/gpu-latest.json
python3 eval/run_eval_gpu.py --write-baseline
python3 eval/run_eval.py
```

## Modes

- `run_eval_gpu.py`: the one authoritative real eval path
- `run_eval.py`: CPU precheck only, no matrix computation

## Bot Behavior

GitHub Actions now provide two automation layers:

- `.github/workflows/eval.yml` runs eval and uploads the generated report artifacts
- `.github/workflows/eval-comment.yml` runs eval on pull requests and posts the markdown summary as a PR comment

## Output

The current system writes:

- `eval/reports/gpu-latest.json`
- `eval/reports/gpu-latest.md`

The current implementation measures on GPU:

- exact GEMM
- transformed GEMM via the `0.0.0` matrix reduction
- bounded accuracy and relative error
- latency
- memory

`run_eval_gpu.py` is the rented-GPU path and the real evaluation system for CCO.

`run_eval.py` remains useful as a lightweight local precheck for repo configuration, but it is not a performance or accuracy evaluation path.
