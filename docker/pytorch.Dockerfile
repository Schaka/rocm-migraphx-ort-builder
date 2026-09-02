# syntax=docker/dockerfile:1
#
# Produces a torch wheel in /pytorch/dist. Independent of MIGraphX -- it only
# needs the HIP/rocBLAS/MIOpen/rocRAND stack already in the base image (or the
# gfx900/gfx906/gfx90c rebuild from the rocblas target).
#
# PyTorch is built (or fetched) here rather than reusing AMD's rocm/pytorch
# image: that image's current tag ships TheRock's pip-packaged ROCm SDK (no
# /opt/rocm, no /opt/rocm/llvm/bin/clang) instead of the classic layout
# MIGraphX's build expects. Reusing it would mean two independent ROCm runtime
# stacks in one image (TheRock's for torch, classic for MIGraphX/ORT), risking
# conflicting libamdhip64.so builds loaded into the same process when the app
# touches both -- it does. One consistent stack is worth the extra build time.
FROM rocblas

ARG ROCM_ARCH
ARG PYTORCH_VERSION
ARG BUILD_PARALLEL_LEVEL
ARG USE_PREBUILT
ARG ROCM_RELEASE
ARG LEGACY_GCN_ARCHES

RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake ninja-build build-essential \
        libopenblas-dev pkg-config libdrm-dev ccache \
    && rm -rf /var/lib/apt/lists/*

# The wheel this target produces has to load into the final image's venv, which
# is pinned to 3.12 -- so build against that same 3.12, not the base image's
# native python3.
RUN uv venv /build-venv --python 3.12 --seed \
    && uv pip install --python /build-venv/bin/python3 \
        numpy pyyaml typing_extensions requests setuptools wheel six build
ENV PATH=/build-venv/bin:$PATH

# Kept as its own layer: the decision is network-bound and cheap, the build below
# is neither, and separating them means a retry of the build doesn't re-resolve
# a floating nightly index to a different answer.
RUN --mount=type=bind,source=scripts/torch-package-build-decide.sh,target=/scripts/torch-package-build-decide.sh \
    /scripts/torch-package-build-decide.sh \
        pytorch "${PYTORCH_VERSION}" "${ROCM_ARCH}" "${USE_PREBUILT}" "${ROCM_RELEASE}" \
        > /tmp/pytorch-decision.txt \
    && cat /tmp/pytorch-decision.txt

RUN --mount=type=cache,target=/root/.ccache,id=pytorch-ccache \
    --mount=type=bind,source=scripts/lib,target=/scripts/lib \
    --mount=type=bind,source=scripts/rocm-devrelease-snapshot.py,target=/scripts/rocm-devrelease-snapshot.py \
    --mount=type=bind,source=scripts/build/pytorch.sh,target=/scripts/build/pytorch.sh \
    /scripts/build/pytorch.sh
