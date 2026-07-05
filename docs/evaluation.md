# Evaluation

CCO separates measurement from policy.

- `bench/` records raw timings
- `eval/` compares versions and decides whether a change is an improvement

Each evaluated run should record:

- transform version
- matrix shape
- dtype
- latency
- reconstruction error
- memory estimate
- notes about the execution environment

Useful commands:

```bash
python3 eval/run_eval_gpu.py
python3 eval/compare_versions.py --report eval/reports/gpu-latest.json
python3 eval/run_eval_gpu.py --write-baseline
python3 eval/run_eval.py
```

For CCO, the real evaluation path is GPU-based. Exact GEMM, transformed GEMM, accuracy, latency, and memory are all measured on GPU in `run_eval_gpu.py`.

Use `python3 eval/run_eval_gpu.py` on a rented or self-hosted GPU machine. That path is intended for repo-level validation and combines:

- GPU exact GEMM
- GPU transformed GEMM
- GPU accuracy and error
- GPU latency and memory
- GPU build status
- GPU test status
- GPU benchmark command status

`run_eval.py` is only a CPU precheck and does not compute matrix results.
