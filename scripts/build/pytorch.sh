#!/bin/sh
# Materialises a torch wheel into /pytorch/dist by the first tier that works:
#   1. prebuilt wheel from an AMD nightly/release index (`pip download`)
#   2. a ROCm devreleases snapshot (rocm-devrelease-snapshot.py)
#   3. a full from-source build of AMD's ROCm/pytorch fork
#
# The plan comes from torch-package-build-decide.sh, read from
# /tmp/pytorch-decision.txt (written by the previous, separately cached, RUN):
#   PYTORCH_PLAN:<pip_pkg_spec_or_EMPTY>|<pip_index_or_EMPTY>|<try_snapshot 0|1>|<arch>|<pytorch_ver_num>|<rocm_release>|<source_repo>|<source_branch>
# ("|"-delimited, not ":" -- the index URL field itself contains colons.)
#
# ALL fields are always present, regardless of whether the upfront listing
# checks in that script found anything: this script actually attempts each tier
# in order and moves on if a tier's real command fails, for whatever reason (not
# just "nothing was listed"). A transient index outage or an unanticipated
# dependency gap must not kill the build when a working fallback (down to
# SOURCE, always present) exists.
#
# Every tier lands its output in the same /pytorch/dist so consumers can read it
# without knowing which tier fired.
# Inputs (build-args): ROCM_ARCH, BUILD_PARALLEL_LEVEL, LEGACY_GCN_ARCHES.
set -eu

. /scripts/lib/legacy-arch.sh
. /scripts/lib/build-jobs.sh

mkdir -p /pytorch/dist

clear_dist() {
    rm -f /pytorch/dist/*.whl /pytorch/dist/*.tar.gz
}

# Source-build fallback uses AMD's ROCm/pytorch fork, not upstream
# pytorch/pytorch: it carries ROCm fixes and a newer composable_kernel submodule
# that upstream lags on. Which repo/branch exactly is the decision script's call.
build_pytorch_from_source() {
    repo="$1"
    branch="$2"
    echo "Building PyTorch from source (repo: $repo, branch: $branch)"
    git clone --recursive --branch "${branch}" --depth 1 --shallow-submodules \
        "https://github.com/${repo}.git" /pytorch-src
    cd /pytorch-src
    python3 tools/amd_build/build_amd.py
    # shellcheck disable=SC3045  # dash (this image's /bin/sh) does implement ulimit -s;
    # the PyTorch build overflows the default 8MB stack in template-heavy TUs.
    ulimit -s unlimited

    jobs="$(resolve_build_jobs)"
    echo "PyTorch build: using $jobs parallel jobs"

    ck_gemm=1
    if is_legacy_gcn_arch "${ROCM_ARCH}"; then
        echo "Legacy GCN arch (${ROCM_ARCH}): composable_kernel denylists" \
             "gfx900/gfx906/gfx90c outright (undeclared CK_BUFFER_RESOURCE_3RD_DWORD)" \
             "-- disabling PyTorch's own CK-based bgemm kernels (USE_ROCM_CK_GEMM)."
        ck_gemm=0
    fi

    env USE_ROCM=1 ROCM_HOME=/opt/rocm "PYTORCH_ROCM_ARCH=${ROCM_ARCH}" \
        MAX_JOBS="$jobs" USE_MKLDNN=0 USE_CCACHE=1 \
        USE_FLASH_ATTENTION=0 USE_MEM_EFF_ATTENTION=0 \
        USE_DISTRIBUTED=0 USE_ROCM_CK_GEMM="$ck_gemm" \
        python3 setup.py bdist_wheel
    cp dist/*.whl /pytorch/dist/
    cd /
    rm -rf /pytorch-src
}

rest="$(grep '^PYTORCH_PLAN:' /tmp/pytorch-decision.txt | tail -1 | sed 's/^PYTORCH_PLAN://')"
if [ -z "$rest" ]; then
    echo "FATAL: no PYTORCH_PLAN line in /tmp/pytorch-decision.txt" >&2
    cat /tmp/pytorch-decision.txt >&2
    exit 1
fi
pip_spec=$(echo "$rest" | cut -d'|' -f1)
pip_index=$(echo "$rest" | cut -d'|' -f2)
try_snapshot=$(echo "$rest" | cut -d'|' -f3)
plan_arch=$(echo "$rest" | cut -d'|' -f4)
plan_pytorch_ver=$(echo "$rest" | cut -d'|' -f5)
plan_rocm_release=$(echo "$rest" | cut -d'|' -f6)
plan_source_repo=$(echo "$rest" | cut -d'|' -f7)
plan_source_branch=$(echo "$rest" | cut -d'|' -f8)

got_it=0

if [ -n "$pip_spec" ]; then
    echo "Downloading prebuilt PyTorch: $pip_spec (+ deps) from $pip_index"
    if /build-venv/bin/pip download "$pip_spec" --index-url "$pip_index" -d /pytorch/dist/; then
        got_it=1
    else
        echo "Prebuilt download failed -- trying the next fallback tier instead of failing the build"
        clear_dist
    fi
fi

if [ "$got_it" = "0" ] && [ "$try_snapshot" = "1" ]; then
    if /build-venv/bin/python3 /scripts/rocm-devrelease-snapshot.py \
            /pytorch/dist cp312 linux_x86_64 \
            "$plan_arch" "$plan_pytorch_ver" "$plan_rocm_release"; then
        got_it=1
    else
        echo "No usable devreleases snapshot either -- trying the next fallback tier instead of failing the build"
        clear_dist
    fi
fi

if [ "$got_it" = "0" ]; then
    build_pytorch_from_source "$plan_source_repo" "$plan_source_branch"
fi
