# Overview

CCO is a CUDA matrix optimization lab.

The core workflow is:

1. transform `Q`, `K`, and `V` from `(n, d)` into `(n', d')`
2. run attention on that transformed representation
3. reconstruct the output back to `(n, d)`
4. compare the version against prior versions with stable metrics

This is the primary research loop of CCO.

The transform target is flexible. In the current baseline, any `(n', d')` that evenly tiles `(n, d)` is allowed.

The repo is split into:

- `kernels/` for CUDA implementation work
- `runtime/` for orchestration and reference logic

The transform strategy remains the main evolving subsystem, and it is evaluated directly in the context of attention.
