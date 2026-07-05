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
python3 eval/run_eval.py
python3 eval/compare_versions.py
```

At the current stage, `bench/` and `eval/` are repo scaffolds. They should evolve together with the first real GEMM and transform implementations.
