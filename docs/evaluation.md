# Evaluation

CCO separates measurement from policy.

- `bench/` records raw timings
- `eval/` compares versions and decides whether a change is an improvement

Each evaluated run should record:

- transform version
- attention shape
- transformed attention shape
- dtype
- latency
- reconstruction error
- memory estimate
- notes about the execution environment

Useful commands:

```bash
bash scripts/eval.sh
python3 eval/run_eval_gpu.py
python3 eval/run_eval_gpu.py --seed 1234
python3 eval/compare_versions.py --report eval/reports/gpu-latest.json
python3 eval/run_eval_gpu.py --write-baseline
python3 eval/run_eval.py
```

For CCO, the real evaluation path is GPU-based. Exact attention, transformed attention, accuracy, latency, and memory are all measured on GPU in `run_eval_gpu.py`.

Each GPU eval run should cover both `float16` and `float32`.

`Q/K/V` inputs are generated randomly from `(n, d)` with a uniform value range of `(-25, 25)`. The default mode is unpredictable; use `--seed` when you need reproducible input generation.

Use `python3 eval/run_eval_gpu.py` on a rented or self-hosted GPU machine. That path is intended for repo-level validation and combines:

- GPU exact attention
- GPU transformed attention
- GPU accuracy and error
- GPU latency and memory
- GPU build status
- GPU test status
- GPU benchmark command status

`run_eval.py` is only a CPU precheck and does not compute matrix results.

`bash scripts/eval.sh` is the automatic entrypoint. It detects whether CUDA is available and chooses the correct path.

The baseline transform accepts flexible targets, so eval should always report both `(n, d)` and `(n', d')`.
