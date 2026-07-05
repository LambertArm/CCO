# CCO Runtime

This directory holds host-side orchestration for CCO.

## Contents

- `src/reference/`: CPU truth paths for correctness checks
- `src/pipeline/`: `Q/K/V` transform -> attention -> reconstruct flow
- `src/dispatch.cpp`: runtime-facing wiring and shared metrics helpers

`runtime/` should coordinate kernels, not replace them.

Current focus:

- host-side wiring for the `0.0.0` transform
- transformed-attention pipeline flow with flexible `(n, d) -> (n', d')` targets
- CPU truth paths used for validation
