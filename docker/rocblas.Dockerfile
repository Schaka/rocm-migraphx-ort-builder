# syntax=docker/dockerfile:1
#
# Serves rocBLAS to every torch-side and MIGraphX target for the legacy GCN
# arches (gfx900/gfx906/gfx90c), where the base image's prebuilt rocBLAS may not
# have usable kernels. The script self-decides (see rocblas.sh):
#   - release builds always rebuild from source -- AMD's pinned stable bases ship
#     no kernels for these arches, and upstream marks them build-passing but not
#     sanity-tested, so a version-targeted build never trusts the prebuilt.
#   - nightly builds use the base's prebuilt kernels when present for this exact
#     base, and rebuild from source when absent (never substituting another
#     version's kernels).
# A no-op passthrough on every other arch, which keeps the base's own prebuilt
# rocBLAS.
#
# Every torch-side and MIGraphX target builds FROM this one, so both see the
# rebuilt rocBLAS when it applies. On gfx900/gfx906/gfx90c, CI overrides this
# target's context with the published rocm-rocblas-builder:<arch> image so the
# (Tensile-heavy, hours-long) rebuild runs once in its own job instead of once
# per dependent job.
#
# Only fires for a single-arch build. A local one-shot bake using the default
# multi-arch ROCM_ARCH list won't match and won't rebuild -- that default is a
# convenience fallback for arches that don't need this fix, not how
# gfx900/gfx906/gfx90c are meant to be built.
FROM python-base

ARG ROCM_ARCH
ARG ROCM_RELEASE
ARG BUILD_PARALLEL_LEVEL
ARG LEGACY_GCN_ARCHES

RUN --mount=type=cache,target=/root/.ccache,id=rocblas-legacy-ccache \
    --mount=type=bind,source=scripts/lib,target=/scripts/lib \
    --mount=type=bind,source=scripts/build/rocblas.sh,target=/scripts/build/rocblas.sh \
    /scripts/build/rocblas.sh
