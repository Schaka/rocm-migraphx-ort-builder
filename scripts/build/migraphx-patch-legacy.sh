#!/bin/sh
# Source patches MIGraphX needs on the legacy GCN arches only. Run from
# /migraphx-src. No-op everywhere else.
# Inputs (build-args): ROCM_ARCH, LEGACY_GCN_ARCHES.
set -eux

. /scripts/lib/legacy-arch.sh

if ! is_legacy_gcn_arch "${ROCM_ARCH}"; then
    echo "migraphx: ${ROCM_ARCH} needs no legacy-GCN source patches"
    exit 0
fi

# composable_kernel explicitly denylists gfx900/gfx906 (its CMakeLists.txt:
# CK_UNSUPPORTED_GPU_TARGETS="gfx900;gfx906;gfx90c" -- a single-target request
# against that list generates a dummy target and returns, per CK's own
# CMakeLists, rather than building anything real). Turning it off via
# -DMIGRAPHX_USE_COMPOSABLEKERNEL=Off alone is not sufficient: rbuild builds
# everything requirements.txt lists regardless of MIGraphX's own cmake options,
# so it would still try to build CK's dummy target for these arches. Strip the
# line so rbuild never attempts it.
sed -i '/composable_kernel/d' requirements.txt
if grep -q 'composable_kernel' requirements.txt; then
    echo "FATAL: composable_kernel still listed in MIGraphX requirements.txt after the strip." >&2
    exit 1
fi

# Upstream bug (filed: see gfx900-906-gfx-default-rocblas-bug-report.md),
# reproduced independently of any of our own build flags/caching:
# device_name.hpp guards gfx_default_rocblas()'s DECLARATION behind
# `#if MIGRAPHX_USE_HIPBLASLT`, but lowering.cpp's one call site has no matching
# guard, so -DMIGRAPHX_USE_HIPBLASLT=Off (required on gfx900/gfx906/gfx90c --
# hipBLASLt has no kernels for any of them) fails to compile outright: "no
# member named 'gfx_default_rocblas' in namespace 'migraphx::gpu'".
# hipblaslt_supported() itself already returns a hardcoded false with the flag
# off, which alone makes the enclosing `or` chain unconditionally true at runtime
# regardless of gfx_default_rocblas() -- so replacing the call with a literal
# `true` under the same guard is a semantics-preserving fix, not a behavior
# change. Confirmed this is the only unguarded call site in actually-compiled
# code (the other two references are in test/, excluded by -DBUILD_TESTING=Off).
sed -i \
    's/not hipblaslt_supported() or gpu::gfx_default_rocblas()/not hipblaslt_supported()/' \
    src/targets/gpu/lowering.cpp
grep -q 'not hipblaslt_supported()' src/targets/gpu/lowering.cpp
if grep -q 'gfx_default_rocblas' src/targets/gpu/lowering.cpp; then
    echo "FATAL: gfx_default_rocblas still referenced in lowering.cpp after the patch -- upstream changed the call site." >&2
    exit 1
fi
