#!/bin/bash
set -eux

# Decide whether to use prebuilt PyTorch wheel or build from source.
# Usage: pytorch-build-decide.sh <pytorch_version> <rocm_arch> [use_prebuilt] [rocm_release]
#
# Args:
#   pytorch_version: PyTorch version tag (v2.13.0, v2.12.0, develop, etc.)
#   rocm_arch:       GPU arch (gfx1033, gfx1100, etc.)
#   use_prebuilt:    Try prebuilt wheels first if set to 1 (default: 0)
#   rocm_release:    ROCm release (7.14, 7.13, etc.). Empty = nightly build

PYTORCH_VERSION=$1
ROCM_ARCH=$2
USE_PREBUILT=${3:-0}
ROCM_RELEASE=${4:-}

log() {
    echo "[pytorch-build] $*" >&2
}

try_download_nightly() {
    local arch=$1
    local pytorch_version=$2  # e.g., v2.13.0 (what we're building)
    local python_ver=$3       # e.g., cp312
    local platform=$4         # e.g., linux_x86_64

    local pytorch_ver_num=${pytorch_version#v}
    local base_url="https://rocm.nightlies.amd.com/whl-multi-arch/amd-torch-device-${arch}/"

    log "Checking nightly wheels: arch=$arch pytorch=$pytorch_ver_num python=$python_ver platform=$platform"

    # Parse directory listing to find matching wheels
    # Look for: amd_torch_device_${arch}-${pytorch_ver_num}+rocm*-${python_ver}-${python_ver}-${platform}.whl
    local matching_wheels
    matching_wheels=$(curl -s "$base_url" 2>/dev/null | \
        grep -oE "href=\"[^\"]*amd_torch_device_${arch}-${pytorch_ver_num}%2Brocm[^\"]*-${python_ver}-${python_ver}-${platform}\.whl\"" | \
        sed 's|href="\.\./||; s/"$//' | \
        sort -V)

    if [ -z "$matching_wheels" ]; then
        log "No nightly wheel found for pytorch=$pytorch_ver_num"
        return 1
    fi

    # Prefer 7.x over 10.x (10.x is bleeding-edge dev). Get latest 7.x if available.
    local wheel_url=$(echo "$matching_wheels" | grep "rocm7\." | tail -1)

    # If no 7.x found, fallback to latest (could be 10.x)
    if [ -z "$wheel_url" ]; then
        log "No rocm7.x nightly found, using latest available"
        wheel_url=$(echo "$matching_wheels" | tail -1)
    else
        log "Found rocm7.x nightly (preferred over rocm10.x)"
    fi

    log "Found nightly wheel (latest): $wheel_url"
    echo "$base_url$wheel_url"
    return 0
}

try_download_release() {
    local arch=$1
    local pytorch_version=$2  # e.g., v2.13.0
    local rocm_release=$3     # e.g., 7.14
    local python_ver=$4       # e.g., cp312
    local platform=$5         # e.g., linux_x86_64

    local pytorch_ver_num=${pytorch_version#v}
    local base_url="https://rocm.devreleases.amd.com/whl-multi-arch/amd-torch-device-${arch}/"

    log "Checking release wheels: arch=$arch pytorch=$pytorch_ver_num rocm=$rocm_release python=$python_ver platform=$platform (EXACT MATCH)"

    # Parse directory listing to find THE ONE release wheel (no .dev, no git hash)
    # Format: amd_torch_device_${arch}-${pytorch_ver_num}+rocm${rocm_release}.0-${python_ver}-${python_ver}-${platform}.whl
    local wheel_url
    wheel_url=$(curl -s "$base_url" 2>/dev/null | \
        grep -oE "href=\"[^\"]*amd_torch_device_${arch}-${pytorch_ver_num}%2Brocm${rocm_release}\.0-${python_ver}-${python_ver}-${platform}\.whl\"" | \
        grep -v "%2Bdev" | \
        sed 's|href="\.\./||; s/"$//' | \
        head -1)

    if [ -z "$wheel_url" ]; then
        log "Wheel not found for exact match: pytorch=$pytorch_ver_num rocm=$rocm_release"
        return 1
    fi

    log "Found matching release wheel: $wheel_url"
    echo "$base_url$wheel_url"
    return 0
}

determine_rocm_pytorch_branch() {
    local pytorch_version=$1

    case "$pytorch_version" in
        v2.13.0)  echo "release/2.13" ;;
        v2.12.0)  echo "release/2.12" ;;
        v2.11.0)  echo "release/2.11" ;;
        v2.10.0)  echo "release/2.10" ;;
        develop)  echo "develop" ;;
        *)        log "Unknown PyTorch version: $pytorch_version"; return 1 ;;
    esac
}

build_from_source() {
    local pytorch_version=$1
    local rocm_arch=$2

    log "Building PyTorch from source (version: $pytorch_version, arch: $rocm_arch)"

    local rocm_pytorch_branch
    rocm_pytorch_branch=$(determine_rocm_pytorch_branch "$pytorch_version") || return 1

    log "Using ROCm/pytorch branch: $rocm_pytorch_branch"

    # Clone happens in the main Dockerfile RUN statement
    # This script just determines the branch
    echo "$rocm_pytorch_branch"
}

# Constants
PYTHON_VERSION="cp312"
PLATFORM="linux_x86_64"

# Main logic
if [ "$USE_PREBUILT" = "1" ]; then
    log "Prebuilt wheels enabled"

    if [ -z "$ROCM_RELEASE" ]; then
        log "Nightly build detected"
        if try_download_nightly "$ROCM_ARCH" "$PYTORCH_VERSION" "$PYTHON_VERSION" "$PLATFORM"; then
            log "Using prebuilt nightly wheel"
            exit 0
        fi
    else
        log "Release build detected (ROCm: $ROCM_RELEASE)"
        if try_download_release "$ROCM_ARCH" "$PYTORCH_VERSION" "$ROCM_RELEASE" "$PYTHON_VERSION" "$PLATFORM"; then
            log "Using prebuilt release wheel"
            exit 0
        fi
    fi

    log "Prebuilt wheel not available, falling back to source build"
fi

# Build from source
build_from_source "$PYTORCH_VERSION" "$ROCM_ARCH"
