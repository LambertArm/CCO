# Overview

CCO is a CUDA matrix optimization lab.

The core workflow is:

1. reduce or transform an `(n, n)` matrix into a smaller or structured form
2. run compute on that transformed representation
3. reconstruct or map results back
4. compare the version against prior versions with stable metrics

The repo is split into:

- `kernels/` for CUDA implementation work
- `runtime/` for orchestration and reference logic

The transform strategy remains the main evolving subsystem, but it now lives in a layout that better matches a real CUDA optimization repo.
