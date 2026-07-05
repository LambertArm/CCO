# CCO Kernels

This directory holds CUDA-facing implementation code for CCO.

## Contents

- `src/transform/`: the transform implementation set, currently starting with version `0.0.0`
- `src/attention/`: attention kernels and attention-building blocks
- `src/common/`: shared low-level utilities used by kernels

This folder is the optimization frontier of the repo.

Current focus:

- one flexible matrix-only transform version: `0.0.0`
- direct transformed attention
- clean separation from runtime orchestration
