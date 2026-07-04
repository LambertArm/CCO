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
