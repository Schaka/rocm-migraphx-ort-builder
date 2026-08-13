# syntax=docker/dockerfile:1
#
# torchaudio wheel: prebuilt wheel first, from-source build of ROCm/audio when no
# matching one exists. Same shared tier logic / ABI gate / source build as
# torchvision (scripts/lib/torch-companion.sh); only the apt dependencies and the
# package name differ.
#
# Note torchaudio's wheel is non-device-specific -- the same one works for every
# GPU target, so no ROCM_ARCH is passed to the decision script (see
# companion_decide_arch). It is still built per-arch by CI because the pipeline
# itself is per-arch, and rebuilding the same wheel choice per arch costs
# nothing beyond the one extra decision-script call.
FROM rocblas AS builder

ENV TORCH_COMPANION=torchaudio

# Declared but unused: torchaudio's wheel is arch-independent, so no arch is
# passed to the decision script (companion_decide_arch). Kept so the shared
# ROCM_ARCH build-arg is consumed rather than warned about.
ARG ROCM_ARCH
ARG PYTORCH_VERSION
ARG BUILD_PARALLEL_LEVEL
ARG USE_PREBUILT
ARG ROCM_RELEASE

RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake ninja-build build-essential pkg-config ccache \
        libsndfile1-dev libsndfile1 sox libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

RUN uv venv /build-venv --python 3.12 --seed \
    && uv pip install --python /build-venv/bin/python3 \
        numpy pyyaml typing_extensions requests setuptools wheel
ENV PATH=/build-venv/bin:$PATH

# See torchvision.Dockerfile for why pytorch's dist dir is bind-mounted whole
# rather than COPYed.
RUN --mount=type=bind,from=pytorch,source=/pytorch/dist,target=/wheels-torch \
    --mount=type=bind,source=scripts/lib,target=/scripts/lib \
    --mount=type=bind,source=scripts/torch-package-build-decide.sh,target=/scripts/torch-package-build-decide.sh \
    --mount=type=bind,source=scripts/generate-torch-constraints.sh,target=/scripts/generate-torch-constraints.sh \
    --mount=type=bind,source=scripts/build/torch-companion.sh,target=/scripts/build/torch-companion.sh \
    /scripts/build/torch-companion.sh

RUN --mount=type=bind,from=pytorch,source=/pytorch/dist,target=/wheels-torch \
    --mount=type=bind,source=scripts/lib,target=/scripts/lib \
    --mount=type=bind,source=scripts/torch-package-build-decide.sh,target=/scripts/torch-package-build-decide.sh \
    --mount=type=bind,source=scripts/build/torch-companion-verify.sh,target=/scripts/build/torch-companion-verify.sh \
    /scripts/build/torch-companion-verify.sh

# Publication target -- same reasoning as torchvision.Dockerfile's.
FROM scratch
COPY --from=builder /torchaudio/dist/torchaudio-*.whl /torchaudio/dist/
