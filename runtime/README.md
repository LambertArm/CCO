# CCO Runtime

This directory holds host-side orchestration for CCO.

## Contents

- `src/reference/`: CPU truth paths for correctness checks
- `src/pipeline/`: matrix reduce -> GEMM -> reconstruct flow
- `src/dispatch.cpp`: runtime-facing wiring and shared metrics helpers

`runtime/` should coordinate kernels, not replace them.

Current focus:

- host-side wiring for the `0.0.0` matrix transform
- GEMM-oriented pipeline flow
- CPU reference stubs for attention-related kernels after GEMM integration
- CPU truth paths used for validation
