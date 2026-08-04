# syntax=docker/dockerfile:1
#
# Rebuilds rocBLAS from source for gfx900/gfx906/gfx90c; a no-op passthrough for
# every other arch, which keeps the base image's own prebuilt rocBLAS (that one
# does have their kernels).
#
# AMD's prebuilt rocBLAS package for this ROCm line ships no
# gfx900/gfx906/gfx90c code objects at all (confirmed: no
# amdrocm-blas*-gfx900/-gfx906/-gfx90c apt package exists, and the base image's
# Tensile library on disk has no gfx900/gfx906/gfx90c folder) -- but the *source*
# still carries them, so rebuilding covers them without a ROCm-major downgrade,
# unlike gfx803 (which needs one for Polaris's actual ROCR enumeration break on
# ROCm 7 -- see gfx803/).
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
