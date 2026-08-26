# shellcheck shell=sh
# Sourced, never executed.
#
# Single predicate for the arches that keep the legacy-GCN special-casing spread
# across the rocblas, migraphx and pytorch stages.
#
# The pieces and why:
#   - rocBLAS: rebuilt from source when the base can't serve these arches -- a
#     release build always does (AMD's stable bases ship no kernels for them),
#     a nightly does only when its own base lacks them (see rocblas.sh). TheRock
#     10.x nightly bases do carry their kernels; hipBLASLt never does.
#   - hipBLASLt: off for these arches always -- no gfx900/gfx906/gfx90c kernels
#     exist in any prebuilt package.
#   - composable_kernel: denylisted for these arches at CK's own CMake configure
#     time, so MIGraphX/PyTorch build with it disabled for them always.
#
# The arch LIST lives in docker-bake.hcl (variable "LEGACY_GCN_ARCHES"), which
# passes it to every stage that needs it as a build-arg. It has to live there
# rather than here because bake also uses it to decide the final image's
# TORCH_BLAS_PREFER_HIPBLASLT ENV, and an ENV cannot be computed by a script.
# Extend it there; nothing else hardcodes the arches.

is_legacy_gcn_arch() {
    if [ -z "${LEGACY_GCN_ARCHES:-}" ]; then
        echo "FATAL: LEGACY_GCN_ARCHES is unset -- build through 'docker buildx bake', which passes it." >&2
        exit 2
    fi
    for _legacy in ${LEGACY_GCN_ARCHES}; do
        [ "$1" = "$_legacy" ] && return 0
    done
    return 1
}
