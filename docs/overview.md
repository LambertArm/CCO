# Overview

CCO is a CUDA matrix optimization lab.

The core workflow is:

1. reduce or transform an `(n, n)` matrix into a smaller or structured matrix form
2. run GEMM on that transformed matrix representation
3. reconstruct or map results back
4. compare the version against prior versions with stable metrics

This is the primary research loop of CCO.

The repo is split into:

- `kernels/` for CUDA implementation work
- `runtime/` for orchestration and reference logic

The transform strategy remains the main evolving subsystem, but it is matrix-only and evaluated in the context of GEMM.

Attention is a later integration target. The attention subsystem exists so improved GEMM paths can eventually be reused there, but it is not the primary optimization target right now.
