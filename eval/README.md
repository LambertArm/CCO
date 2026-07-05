# CCO Eval

`eval/` is the real correctness and regression layer for CCO.

It is intentionally modeled after the role `eval/` plays in `sparkinfer`, but adapted to CCO's current scope:

- one matrix transform version: `0.0.0`
- transformed-attention compute path
- kernel/runtime split

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
bash scripts/eval.sh
python3 eval/run_eval_gpu.py
python3 eval/run_eval_gpu.py --seed 1234
python3 eval/compare_versions.py --report eval/reports/gpu-latest.json
python3 eval/run_eval_gpu.py --write-baseline
python3 eval/run_eval.py
```

## Modes

- `run_eval_gpu.py`: the one authoritative real eval path
- `run_eval.py`: CPU precheck only, no matrix computation
- `scripts/eval.sh`: automatic chooser between CPU precheck and GPU eval

## Bot Behavior

GitHub Actions now provide two automation layers:

- `.github/workflows/pr-precheck.yml` runs CPU-only precheck on every PR update, comments the result, auto-closes failing PRs, and queues passing PRs
- `.github/workflows/gpu-queue.yml` is the rented-GPU worker that drains queued PRs in arrival order and comments the real GPU eval result for each one

## Output

The current system writes:

- `eval/reports/gpu-latest.json`
- `eval/reports/gpu-latest.md`

The current implementation measures on GPU:

- exact attention
- transformed attention via the `0.0.0` `Q/K/V` transform
- both `float16` and `float32`
- bounded accuracy and relative error
- latency
- memory

The transform target is flexible: the eval harness accepts any `(n', d')` that cleanly divide `(n, d)`.

If a shape case omits `n_transformed` and `d_transformed`, `run_eval_gpu.py` resolves them from the active transform version automatically.

`Q/K/V` are randomly generated from `(n, d)` with values sampled uniformly from `(-25, 25)`. By default the seeds are unpredictable. Pass `--seed` when you want a reproducible eval run.

`run_eval_gpu.py` is the rented-GPU path and the real evaluation system for CCO.

`run_eval.py` remains useful as a lightweight local precheck for repo configuration, but it is not a performance or accuracy evaluation path.
