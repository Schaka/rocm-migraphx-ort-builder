# syntax=docker/dockerfile:1
#
# torchvision wheel: prebuilt per-arch wheel first, from-source build of
# pytorch/vision when no matching one exists. The tier logic, the ABI gate and
# the source build are shared with torchaudio -- see scripts/lib/torch-companion.sh
# and scripts/build/torch-companion*.sh; only the apt dependencies and the
# package name differ, which is all this file carries.
FROM rocblas AS builder

ENV TORCH_COMPANION=torchvision

ARG ROCM_ARCH
ARG PYTORCH_VERSION
ARG BUILD_PARALLEL_LEVEL
ARG USE_PREBUILT
ARG ROCM_RELEASE

RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake ninja-build build-essential pkg-config ccache \
        libjpeg-dev libpng-dev libfreetype6-dev libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

RUN uv venv /build-venv --python 3.12 --seed \
    && uv pip install --python /build-venv/bin/python3 \
        numpy pyyaml typing_extensions requests setuptools wheel
ENV PATH=/build-venv/bin:$PATH

# pytorch's dist dir is bind-mounted, not COPYed: torch's wheel alone is ~1.7GB,
# and a COPY would commit it as a layer of this stage. The whole directory is
# exposed, not just *.whl -- it may also hold a non-.whl sdist (the `rocm`
# metapackage, when the PIP tier pulled in TheRock's rocm-sdk-* deps).
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

# Publication target -- CI publishes THIS, not `builder`. The builder stage is a
# full ROCm base plus an apt toolchain plus a /build-venv with torch installed in
# it: ~25GB published, of which the final image consumes exactly one wheel. A
# bind mount cannot pull part of an image, so publishing the builder would make
# the final job download all 25GB to read that one file -- on a runner that has
# no room for it. scratch + the wheel is the whole contract.
#
# Everything torchvision, and nothing else: torch, triton, rocm_sdk_* and the
# shared transitive deps that `pip download` also leaves in dist/ are either
# byte-identical duplicates of what the pytorch image already hands the final
# stage, or resolvable from PyPI there. Dropping the duplicate torch wheel cannot
# mask a torch/torchvision mismatch either -- the ABI gate above has already
# imported this exact wheel against that exact torch.
#
# The glob is *torchvision* and NOT torchvision-*, which is a real bug fix rather
# than tidying. On the prebuilt path `pip download` produces TWO torchvision
# wheels:
#   torchvision-<ver>.whl                        -- arch-agnostic python + _C.so
#   amd_torchvision_device_gfx<arch>-<ver>.whl   -- the actual GPU code objects
# torchvision-* matched only the first, so the published component -- and every
# image built from it -- carried a _C.so whose .hip_fatbin is NOBITS, i.e. no
# device code at all. It imports, it registers a CUDA kernel, and then dies on
# the first GPU op with "CUDA error: invalid kernel file" (observed on gfx1201).
# The per-arch device wheel is neither a duplicate of anything the pytorch image
# supplies nor available on PyPI, so nothing downstream could recover it.
#
# Both names contain "torchvision", so this one glob covers the prebuilt path and
# still matches the single wheel a source build produces.
FROM scratch
COPY --from=builder /torchvision/dist/*torchvision*.whl /torchvision/dist/
