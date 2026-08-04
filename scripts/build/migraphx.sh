#!/bin/sh
# Builds MIGraphX (and, through rbuild, its pinned rocMLIR/composable_kernel
# dependencies) into /opt/rocm. Run from /migraphx-src.
# Inputs (build-args): ROCM_ARCH, MIGRAPHX_REF, BUILD_PARALLEL_LEVEL,
# LEGACY_GCN_ARCHES.
set -eu

. /scripts/lib/legacy-arch.sh
. /scripts/lib/build-jobs.sh

# shellcheck disable=SC3045  # dash (this image's /bin/sh) does implement ulimit -s;
# the rocMLIR/LLVM build overflows the default 8MB stack in template-heavy TUs.
ulimit -s unlimited

jobs="$(resolve_build_jobs)"
echo "rocMLIR/LLVM build: using $jobs parallel jobs"

# hipBLASLt's Tensile Logic tree has never had gfx900/gfx906/gfx90c kernels
# (oldest arch present is arcturus/gfx908, the same fact that motivates the
# rocblas stage) -- link against the rocBLAS rebuilt there instead of a
# hipBLASLt with nothing to dispatch to on this hardware. composable_kernel is
# denylisted for all three by CK's own CMakeLists (see migraphx-patch-legacy.sh)
# -- MIGRAPHX_USE_COMPOSABLEKERNEL=Off keeps MIGraphX's own cmake from expecting
# the CK package that was never built.
extra_cmake_args=""
if is_legacy_gcn_arch "${ROCM_ARCH}"; then
    extra_cmake_args="-DMIGRAPHX_USE_HIPBLASLT=Off -DMIGRAPHX_USE_COMPOSABLEKERNEL=Off"
fi

# rbuild shells out to a bare `cget` (relies on PATH, not sys.executable), so
# the venv's bin dir must be on PATH, not just invoked via absolute path.
#
# MIGraphX's own cmake/PythonModules.cmake ignores -DPython3_EXECUTABLE below
# entirely for its python-module target(s): its find_python(version) macro
# instead does a bare `find_program(python<version>-config)` PATH search, once
# per entry in a hardcoded 3.6-3.14 list, and silently skips any version whose
# `-config` script isn't found -- no error, just that version missing from the
# PYTHON_VERSIONS it then builds modules for. /rbuild-venv (a venv) never
# carries a python3.12-config script -- that's an artifact of the underlying
# real interpreter install, not something venvs create -- so without this,
# find_python(3.12) silently fails while find_python(3.14) silently succeeds
# against this base image's own native python3.14-config on PATH, and the only
# migraphx.so that ends up built is cpython-314-tagged: unimportable from every
# 3.12 venv downstream. `uv python find 3.12`'s own bin dir is where the real
# python3.12-config actually lives; PYTHON_DISABLE_VERSIONS=3.14 additionally
# makes the outcome deterministic regardless of what else this base image's
# python3-config search might turn up, rather than relying on 3.12 merely
# winning some discovery-order race against 3.14.
py312_bin="$(dirname "$(uv python find 3.12)")"

CMAKE_BUILD_PARALLEL_LEVEL=$jobs \
PATH="${py312_bin}:/rbuild-venv/bin:$PATH" \
/rbuild-venv/bin/rbuild build -d /migraphx-deps -B build -G Ninja \
    --cxx=/opt/rocm/llvm/bin/clang++ --cc=/opt/rocm/llvm/bin/clang \
    "-DGPU_TARGETS=${ROCM_ARCH}" \
    -DCMAKE_INSTALL_PREFIX=/opt/rocm \
    -DCMAKE_BUILD_TYPE=Release \
    -DMIGRAPHX_ENABLE_PYTHON=On \
    -DPython3_EXECUTABLE=/rbuild-venv/bin/python3 \
    -DPYTHON_DISABLE_VERSIONS=3.14 \
    -DBUILD_TESTING=Off \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    ${extra_cmake_args} \
    -T install

# Stamp the built ref + resolved commit into the image so downstream consumers
# (e.g. AudioMuse-AI's entrypoint) can detect a MIGraphX change across
# BASE_IMAGE bumps and invalidate their compiled-model cache instead of silently
# recompiling against a stale cache forever. Captured here, before
# /migraphx-src is removed below.
echo "${MIGRAPHX_REF} $(git -C /migraphx-src rev-parse HEAD)" > /opt/rocm/migraphx-version.txt

# rbuild's dependency tree (a full rocMLIR/LLVM build, easily tens of GB) and
# the MIGraphX source/build dir are both fully installed into /opt/rocm by now
# -- rm'ing them inside this same RUN (rather than a later one) keeps them out
# of this layer's diff entirely instead of just hiding them behind a whiteout,
# which is what was actually filling the runner's disk during the final stage's
# image export.
rm -rf /migraphx-src /migraphx-deps
