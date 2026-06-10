# CCO canonical-rerun runtime image — runtime/Dockerfile.
#
# Build ONCE, push to a public registry, pin by CONTENT DIGEST (sha256), record the digest in
# cco.config.json. Every canonical rerun pulls THIS image by digest — nobody rebuilds per run, so
# the harness + oracles + configs + champions are byte-identical to what manifest.json pins. Only
# kernel.py is mounted in at run time (the variable artifact).
#
# v1 posture: runs on a trusted GPU box (cco.config.json gate4.attestation = deferred_v2). For v2
# the published image's identity lands in MRTD under a GPU-TEE; see DESIGN.md §6.
#
#   docker build -t cco-runtime:v1 -f runtime/Dockerfile .
#   docker run --rm --gpus '"device=0"' --network=none \
#       -e PR_HEAD_SHA=<sha> \
#       -v "$PWD/kernel.py:/cco/kernel.py:ro" -v "$PWD/out:/out" cco-runtime:v1

ARG CUDA_IMAGE=nvidia/cuda:12.8.0-runtime-ubuntu24.04
FROM ${CUDA_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1

# Toolchain: python + a C compiler/headers (Triton JITs C at run time); nvidia-smi ships in base.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-dev python3-pip build-essential ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# uv for reproducible installs
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /cco

# Pinned GPU stack (cu128). Exact versions keep the measured image reproducible.
ARG TORCH_VERSION=2.8.0
ARG TRITON_VERSION=3.4.0
RUN uv pip install --system --index-url https://download.pytorch.org/whl/cu128 \
        "torch==${TORCH_VERSION}+cu128" \
    && uv pip install --system "triton==${TRITON_VERSION}" numpy "tomli; python_version < '3.11'"
# numpy is not imported by CCO code; it is installed so torch's numpy-interop paths are available.

# Bake the LOCKED competition surface (everything manifest.json pins). kernel.py is NOT part of the
# locked surface — it is mounted read-only at run time as the variable artifact (the COPY below is
# only a self-contained smoke-test seed; the run-time mount overrides it).
COPY benchmark.py ./benchmark.py
COPY cco ./cco
COPY references ./references
COPY kernel_configs ./kernel_configs
COPY champions ./champions
COPY cco.config.json payload-schema.json pyproject.toml ./
COPY runtime/gpu.config runtime/start.sh ./runtime/
RUN chmod +x runtime/start.sh
COPY kernel.py ./kernel.py

ENTRYPOINT ["/cco/runtime/start.sh"]
