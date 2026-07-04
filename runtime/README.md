# CCO Runtime

This directory holds host-side orchestration for CCO.

## Contents

- `src/reference/`: CPU truth paths for correctness checks
- `src/pipeline/`: matrix reduce -> GEMM -> reconstruct flow
- `src/dispatch.cpp`: runtime-facing wiring and shared metrics helpers

`runtime/` should coordinate kernels, not replace them.
