# shellcheck shell=sh
# Sourced, never executed.
#
# Single predicate for the arches that need the rocBLAS-from-source /
# composable_kernel-off / hipBLASLt-off special-casing spread across the
# rocblas, migraphx and pytorch stages: AMD's prebuilt packages for this ROCm
# line ship no gfx900/gfx906/gfx90c code objects at all, in rocBLAS, hipBLASLt,
# or composable_kernel.
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
