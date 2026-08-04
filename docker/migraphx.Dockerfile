# syntax=docker/dockerfile:1
#
# MIGraphX built from source into /opt/rocm, against a ROCm release that has no
# matching prebuilt MIGraphX package (it isn't part of TheRock's package set).
# The ort and final targets COPY the resulting /opt/rocm wholesale.
FROM rocblas

ARG ROCM_ARCH
ARG LEGACY_GCN_ARCHES

RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake ninja-build build-essential pkg-config ccache \
        python3 python3-dev python3-pybind11 \
    && rm -rf /var/lib/apt/lists/*

# TheRock's bundled rocm-cmake modules under /opt/rocm are stale/incomplete for
# building components outside TheRock itself (e.g. missing the
# rocm_add_version_resource macro AMDMIGraphX's CMakeLists.txt needs) -- install
# a fresh copy over the same prefix.
RUN git clone --branch develop --depth 1 \
        https://github.com/ROCm/rocm-cmake.git /rocm-cmake-src \
    && cmake -S /rocm-cmake-src -B /rocm-cmake-src/build -DCMAKE_INSTALL_PREFIX=/opt/rocm \
    && cmake --build /rocm-cmake-src/build --target install \
    && rm -rf /rocm-cmake-src

# Use MIGraphX's own documented from-source build path (rbuild, per
# docs/install/install-migraphx.rst on develop), instead of hand-rolling each
# third-party dependency: rbuild reads requirements.txt and builds every
# dependency itself at the exact pinned commit MIGraphX's CMakeLists.txt expects
# -- notably ROCm/composable_kernel's `codegen` subdir specifically, which is
# what actually produces the composable_kernel_host CMake package (not the same
# artifact as the plain "composable_kernel" apt package), and ROCm/rocMLIR built
# from source, since neither has a matching apt package for this ROCm release.
#
# rbuild's dependency `cget` subclasses urllib.request.FancyURLopener, which
# Python 3.14 (this image's only system python3) removed outright -- unrelated to
# MIGraphX itself, just the base image shipping a python newer than that old tool
# supports. Give rbuild/cget an isolated Python 3.12 via uv instead of touching
# the system interpreter or MIGraphX's own build.
#
# This same venv doubles as the interpreter MIGraphX's python module is built
# against, so the module's ABI tag lines up with the Python 3.12 the torch, ort
# and final targets all use.
RUN uv venv /rbuild-venv --python 3.12 \
    && uv pip install --python /rbuild-venv/bin/python3 \
        https://github.com/RadeonOpenCompute/rbuild/archive/master.tar.gz

# SOURCE_DATE busts this layer's cache when it changes -- develop is a moving
# branch, so without it the clone would reuse a stale commit on every nightly
# rebuild. The compile layer below busts with it, which is intended: fresh
# develop each night. Not needed when MIGRAPHX_REF pins a fixed branch/tag, since
# that ref's own commits already change the cache key when it moves --
# docker-bake.hcl only varies it for develop.
ARG SOURCE_DATE
ARG MIGRAPHX_REF
RUN echo "MIGraphX ${MIGRAPHX_REF} snapshot: ${SOURCE_DATE}" \
    && git clone --branch "${MIGRAPHX_REF}" --depth 1 \
        https://github.com/ROCm/AMDMIGraphX.git /migraphx-src

WORKDIR /migraphx-src

RUN --mount=type=bind,source=scripts/lib,target=/scripts/lib \
    --mount=type=bind,source=scripts/build/migraphx-patch-legacy.sh,target=/scripts/build/migraphx-patch-legacy.sh \
    /scripts/build/migraphx-patch-legacy.sh

ARG BUILD_PARALLEL_LEVEL
# Cache mount (not an image layer) persists ccache's compiled-object cache across
# a *failed* RUN retry -- a plain layer is only committed on success, so without
# this a segfault at job 359/658 (like the one that motivated the
# MemAvailable-based job cap in build-jobs.sh) would force recompiling everything
# from zero on the next attempt. Can't also cache-mount the `build` dir itself:
# rbuild renames/recreates it between dependency phases (LLVM, then MIGraphX
# proper), which fails against a bind-mounted mountpoint.
RUN --mount=type=cache,target=/root/.ccache,id=migraphx-ccache \
    --mount=type=bind,source=scripts/lib,target=/scripts/lib \
    --mount=type=bind,source=scripts/build/migraphx.sh,target=/scripts/build/migraphx.sh \
    /scripts/build/migraphx.sh

RUN --mount=type=bind,source=scripts/build/migraphx-verify-python.sh,target=/scripts/build/migraphx-verify-python.sh \
    /scripts/build/migraphx-verify-python.sh
