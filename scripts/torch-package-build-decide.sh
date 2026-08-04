#!/bin/bash
set -eux

# Decide whether to use prebuilt wheel or build from source for torch packages.
# Handles: pytorch, torchvision, torchaudio.
#
# For prebuilt wheels: outputs "PIP:<package>==<exact_version>:<index_url>"
# The caller feeds this straight to `uv pip install <package>==<version> --index-url <url>`,
# which resolves the correct Python-version wheel AND all transitive deps (rocm-sdk-core,
# triton, etc.) automatically -- pip/uv's own dependency resolver replaces what used to be
# a manual curl-one-wheel-file-and-hope step.
#
# What pip/uv CANNOT do: decide which nightly build to prefer. This script's discovery
# functions sort the index themselves (`sort -V` + take the last match) and take the newest
# match regardless of ROCm major -- nightly is meant to float to whatever AMD actually
# publishes newest. See README.md for why no major-version preference is applied here.
#
# pytorch specifically, in release mode (ROCM_RELEASE set), can also output:
#   "SNAPSHOT:<arch>:<pytorch_ver_num>:<rocm_release>:<source_repo>:<source_branch>"
# meaning: no stable release wheel exists on repo.amd.com for this exact pin. The caller
# should run scripts/rocm-devrelease-snapshot.py with the bundled arch/pytorch_ver_num/
# rocm_release (that script resolves a devreleases nightly snapshot where the whole
# dependency closure is mutually consistent -- see its own header for why); if THAT also
# comes up empty, fall back to a source build using the bundled source_repo/source_branch.
#
# For source builds: outputs "SOURCE:<repo>:<branch>"
#
# torchvision/torchaudio must match whatever pytorch itself actually resolved to, not be
# independently guessed -- pass pytorch's exact resolved version string (e.g.
# "2.13.0+rocm7.15.0a20260728", read off the actual downloaded wheel filename, not re-derived)
# as pytorch_resolved_version so their discovery can find a wheel with the SAME rocm date-tag
# (guaranteeing a same-day, ABI-compatible build) instead of picking their own "latest" and
# hoping it lines up -- it doesn't always: AMD's torchvision index mixes multiple version
# trains (0.23 through an unstable 0.29.0a0 dev-track) with independent, sometimes stale
# nightly cadences, so "latest torchvision" and "latest pytorch" can diverge by over a week.
# If pytorch went SOURCE (no prebuilt wheel found), pytorch_resolved_version is empty and
# torchvision/torchaudio should also go SOURCE -- there's no date to pair against.
#
# Usage: torch-package-build-decide.sh <package> <pytorch_version> <rocm_arch> [use_prebuilt] [rocm_release] [pytorch_resolved_version]
#
# Args:
#   package:                  pytorch | torchvision | torchaudio
#   pytorch_version:          PyTorch version tag (v2.13.0, v2.11.0, develop, etc.) -- used to
#                             pick the SOURCE-fallback branch, as an exact pin for pytorch's own
#                             RELEASE-mode (ROCM_RELEASE set) discovery, and ALSO as an optional
#                             pin for pytorch's own NIGHTLY discovery: if set (e.g. v2.13.0),
#                             nightly discovery filters to wheels of that pytorch version while
#                             still floating on the latest ROCm nightly date/build for it. Empty
#                             (or "develop") means fully floating -- take whatever the latest
#                             nightly build actually is, any pytorch version. Prebuilt wheels
#                             bundle their own ROCm runtime, so pinning pytorch while floating
#                             ROCm carries no ABI conflict risk.
#   rocm_arch:                GPU arch (gfx1100, gfx1201, etc.) - required for pytorch/torchvision, ignored for torchaudio
#   use_prebuilt:             Try prebuilt wheels first if set to 1 (default: 0)
#   rocm_release:             ROCm release (7.14, 7.13, etc.). Empty = nightly build
#   pytorch_resolved_version: torchvision/torchaudio only -- pytorch's exact resolved version
#                             string (from its own downloaded wheel filename). Empty if pytorch
#                             went SOURCE, or when discovering pytorch itself (not applicable).

PACKAGE=$1
PYTORCH_VERSION=$2
ROCM_ARCH=${3:-}
USE_PREBUILT=${4:-0}
ROCM_RELEASE=${5:-}
PYTORCH_RESOLVED_VERSION=${6:-}

PYTHON_VERSION="cp312"
PLATFORM="linux_x86_64"

NIGHTLY_INDEX="https://rocm.nightlies.amd.com/whl-multi-arch/"
# repo.amd.com is AMD's actual stable-release pip index -- confirmed via
# https://github.com/ROCm/TheRock/issues/6931, which names it directly as "the ROCm 7.14
# release pip index". rocm.devreleases.amd.com (below) is NOT this: every build hosted there
# is tagged "rcN", ".dev0+<git-hash>", or "aYYYYMMDD" (nightly) -- confirmed empirically that
# a devreleases build's OWN dependency metadata always declares abstract pins like
# "rocm-sdk-device-<arch>==7.14.0" that devreleases itself never actually has a matching
# build for, regardless of which day's build you pick. Only repo.amd.com carries wheels
# where the version string an install actually asked for is the version string that exists.
STABLE_RELEASE_INDEX="https://repo.amd.com/rocm/whl-multi-arch/"
RELEASE_INDEX="https://rocm.devreleases.amd.com/whl-multi-arch/"

log() {
    echo "[torch-build-decide] $*" >&2
}

determine_rocm_pytorch_branch() {
    local pytorch_version=$1
    local base_branch

    case "$pytorch_version" in
        v2.13.0)  base_branch="release/2.13" ;;
        v2.12.0)  base_branch="release/2.12" ;;
        v2.11.0)  base_branch="release/2.11" ;;
        v2.10.0)  base_branch="release/2.10" ;;
        ""|develop) base_branch="develop" ;;
        *)        log "Unknown PyTorch version: $pytorch_version"; return 1 ;;
    esac

    if [ "$base_branch" != "develop" ]; then
        log "Using PyTorch branch: $base_branch"
    fi

    echo "$base_branch"
}

determine_torchvision_repo_branch() {
    local pytorch_version=$1
    local repo="pytorch/vision"
    local branch="release/0.28"  # Covers 0.28.x versions

    log "Using torchvision branch: $branch"
    echo "$repo;$branch"
}

# Pull the "aYYYYMMDD" nightly date-tag out of a resolved version string, e.g.
# "2.13.0+rocm7.15.0a20260728" -> "20260728". Used to find a torchvision/torchaudio wheel from
# the SAME build day as whatever pytorch itself resolved to, rather than each package guessing
# its own "latest" independently (see the file header comment for why that's unreliable).
extract_rocm_datetag() {
    local version_string=$1
    echo "$version_string" | grep -oE '[0-9]{8}' | tail -1
}

determine_torchaudio_repo_branch() {
    local pytorch_version=$1
    local repo="ROCm/audio"
    local branch="release/2.11.0.2"  # Latest stable patch

    log "Using torchaudio branch: $branch"
    echo "$repo;$branch"
}

# --- Version discovery: find the exact version string to pin, prefer stable rocm major ---

# List candidate wheel filenames for a package dir, return them sorted (oldest -> newest by -V).
list_wheel_versions() {
    local base_url=$1
    local name_pattern=$2  # regex matching the filename (without .whl and directory prefix)

    curl -s "$base_url" 2>/dev/null | \
        grep -oE "href=\"[^\"]*${name_pattern}-${PYTHON_VERSION}-${PYTHON_VERSION}-${PLATFORM}\.whl\"" | \
        sed 's|href="\.\./||; s/"$//; s/%2B/+/g' | \
        sort -V
}

# Extract the version portion (everything between the first '-' after the package name
# and the '-cpXXX' python tag) from a wheel filename.
extract_version() {
    local wheel_filename=$1
    local name_prefix=$2  # e.g. amd_torch_device_gfx900

    echo "$wheel_filename" | sed -E "s/^${name_prefix}-(.+)-${PYTHON_VERSION}-${PYTHON_VERSION}-${PLATFORM}\.whl\$/\1/"
}

# Nightly pytorch tracks a moving target on the ROCm side (always float to latest ROCm nightly
# build/date), but the pytorch version itself CAN be pinned: pass pytorch_ver_num (e.g. "2.13.0")
# to filter to just that pytorch version's nightly wheels; leave empty to float on pytorch
# version too (grab the absolute latest, whatever version it is). No ROCm-major preference is
# applied -- nightly floats to whatever's newest. See README.md.
pytorch_find_nightly_version() {
    local arch=$1
    local pytorch_ver_num=${2:-}
    local base_url="${NIGHTLY_INDEX}amd-torch-device-${arch}/"
    local name_prefix="amd_torch_device_${arch}"

    if [ -n "$pytorch_ver_num" ]; then
        log "Checking nightly pytorch versions: $arch pinned to pytorch=$pytorch_ver_num"
    else
        log "Checking nightly pytorch versions: $arch (any pytorch version)"
    fi

    local matches
    if [ -n "$pytorch_ver_num" ]; then
        matches=$(list_wheel_versions "$base_url" "${name_prefix}-${pytorch_ver_num}%2Brocm[^\"]*")
    else
        matches=$(list_wheel_versions "$base_url" "${name_prefix}-[^\"]*%2Brocm[^\"]*")
    fi
    if [ -z "$matches" ]; then
        return 1
    fi

    extract_version "$(echo "$matches" | tail -1)" "$name_prefix"
}

# Exact match only -- no trailing wildcard, no rc/dev tolerance. repo.amd.com only ever hosts
# the literal "<pytorch_ver>+rocm<release>.0" build (see STABLE_RELEASE_INDEX's own comment
# above), so a match here is a real, fully-resolvable release with no substitution needed.
pytorch_find_stable_release_version() {
    local arch=$1
    local pytorch_version=$2
    local rocm_release=$3
    local pytorch_ver_num=${pytorch_version#v}
    local base_url="${STABLE_RELEASE_INDEX}amd-torch-device-${arch}/"
    local name_prefix="amd_torch_device_${arch}"

    log "Checking stable release (repo.amd.com) pytorch versions: $arch pytorch=$pytorch_ver_num rocm=$rocm_release"

    local matches
    matches=$(list_wheel_versions "$base_url" "${name_prefix}-${pytorch_ver_num}%2Brocm${rocm_release}\.0")
    if [ -z "$matches" ]; then
        return 1
    fi

    extract_version "$(echo "$matches" | head -1)" "$name_prefix"
}

# --- Stable-release (repo.amd.com) discovery for the companion packages ---
#
# A pinned release build must not ship a devreleases nightly when AMD has already
# published a real release for that exact ROCm line. Observed before this existed: a build
# pinned to rocm_version=7.14.0 resolved "torchaudio==2.11.0.2+rocm10.0.0a20260729" -- a
# ROCm 10.0 nightly -- because the companion paths below only ever looked at
# RELEASE_INDEX (which is devreleases, NOT repo.amd.com; see those two variables' comments).
#
# These are only consulted when the resolved torch is a clean "<ver>+rocm<release>.0" build
# for this exact ROCm line -- no aYYYYMMDD / rcN / .dev suffix. That is the same reason
# extract_rocm_datetag finds no date-tag for such a wheel and the date-matching paths below
# fall through with nothing to pair against.
#
# Note this does NOT prove the wheel came from repo.amd.com: the devreleases-snapshot tier
# also produces "2.13.0+rocm7.14.0"-shaped names. It is deliberately the weaker check, because
# the property that actually matters for pairing is "same pytorch version, same ROCm line",
# not which host served it -- and a stable companion is the preferred partner either way. A
# genuinely bad pairing is still caught downstream by the companion ABI gate, which falls back
# to a source build.
stable_torch_version_from_wheel() {
    local resolved=$1
    local rocm_release=$2
    local escaped=${rocm_release//./\\.}

    [ -n "$rocm_release" ] || return 1
    [ -n "$resolved" ] || return 1

    local version
    version=$(echo "$resolved" \
        | sed -nE "s/^torch-([0-9][^+]*)\+rocm${escaped}\.0-.*\.whl$/\1/p")
    [ -n "$version" ] || return 1
    echo "$version"
}

# torchaudio's version tracks torch's 1:1 (2.10.0/2.11.0 on this index sit alongside torch
# 2.10.0/2.11.0; 2.8.0a0+rocm7.13.0 alongside torch 2.8.0). Ask for that exact pairing rather
# than "newest stable" -- taking an arbitrary stable build would reintroduce the same
# mispairing this function exists to prevent, just within one ROCm line instead of across two.
torchaudio_find_stable_release_version() {
    local torch_version=$1
    local rocm_release=$2
    local base_url="${STABLE_RELEASE_INDEX}torchaudio/"

    log "Checking stable release (repo.amd.com) torchaudio: torch=$torch_version rocm=$rocm_release"

    local matches
    matches=$(list_wheel_versions "$base_url" "torchaudio-${torch_version}%2Brocm${rocm_release//./\\.}\.0")
    if [ -z "$matches" ]; then
        return 1
    fi

    extract_version "$(echo "$matches" | tail -1)" "torchaudio"
}

# torchvision does NOT track torch's version; upstream pairs torch 2.N with torchvision
# 0.(N+15) (2.10->0.25, 2.11->0.26, 2.12->0.27, all confirmed present together on this index).
# Derive that exact pairing and ask for it specifically; anything unexpected falls through to
# the existing date-tag path and ultimately to a source build, and the companion ABI gate
# re-checks the result either way.
torchvision_find_stable_release_version() {
    local arch=$1
    local torch_version=$2
    local rocm_release=$3
    local base_url="${STABLE_RELEASE_INDEX}amd-torchvision-device-${arch}/"
    local name_prefix="amd_torchvision_device_${arch}"

    local torch_minor=${torch_version#*.}
    torch_minor=${torch_minor%%.*}
    case "$torch_minor" in
        ''|*[!0-9]*) log "Cannot derive torchvision version from torch=$torch_version"; return 1 ;;
    esac
    local vision_minor=$((torch_minor + 15))

    log "Checking stable release (repo.amd.com) torchvision: $arch torch=$torch_version -> 0.${vision_minor}.x rocm=$rocm_release"

    local matches
    matches=$(list_wheel_versions "$base_url" \
        "${name_prefix}-0\.${vision_minor}\.[0-9]+%2Brocm${rocm_release//./\\.}\.0")
    if [ -z "$matches" ]; then
        return 1
    fi

    extract_version "$(echo "$matches" | tail -1)" "$name_prefix"
}

# torchvision/torchaudio discovery: find a wheel from the SAME build day as pytorch's own
# resolved version (via its date-tag), not an independently-guessed "latest" -- see file header
# and extract_rocm_datetag's comment for why (AMD's torchvision index mixes multiple version
# trains with independent, sometimes stale nightly cadences; a first version of this script
# took the globally-latest wheel regardless of that and silently paired torchvision from over a
# week before pytorch's build -- caught by comparing dates, not by any error at build time).
torchvision_find_matching_version() {
    local arch=$1
    local base_url=$2
    local datetag=$3
    local name_prefix="amd_torchvision_device_${arch}"

    log "Checking torchvision versions: $arch matching date-tag $datetag"

    local matches
    matches=$(list_wheel_versions "$base_url" "${name_prefix}-[^\"]*a${datetag}[^\"]*")
    if [ -z "$matches" ]; then
        return 1
    fi

    extract_version "$(echo "$matches" | tail -1)" "$name_prefix"
}

# torchaudio: no per-arch wheel (same binary for every GPU target, see earlier research),
# so no ROCM_ARCH in the URL/pattern at all -- simplest of the three to discover, and matched
# to pytorch's date-tag the same way as torchvision.
torchaudio_find_matching_version() {
    local base_url=$1
    local datetag=$2

    log "Checking torchaudio versions matching date-tag $datetag"

    local matches
    matches=$(list_wheel_versions "$base_url" "torchaudio-[^\"]*a${datetag}[^\"]*")
    if [ -z "$matches" ]; then
        return 1
    fi

    extract_version "$(echo "$matches" | tail -1)" "torchaudio"
}

# torchaudio's own metadata only pins a bare `torch` (no version constraint -- confirmed by
# inspecting its wheel METADATA), unlike torchvision, so it doesn't need the exact same-day
# match torchvision does: any reasonably fresh torchaudio build works against whatever torch is
# actually installed. Used as a fallback when there's no pytorch date-tag to match against
# (pytorch went SOURCE) -- just grab the latest published torchaudio wheel. No rocm7.x
# preference here either, same reasoning as pytorch_find_nightly_version's comment above.
torchaudio_find_latest_version() {
    local base_url=$1

    log "Checking torchaudio versions: latest available (no pytorch date to match)"

    local matches
    matches=$(list_wheel_versions "$base_url" "torchaudio-[^\"]*")
    if [ -z "$matches" ]; then
        return 1
    fi

    extract_version "$(echo "$matches" | tail -1)" "torchaudio"
}

# --- Main dispatcher ---

case "$PACKAGE" in
    pytorch)
        # Always bundles every tier's info in one line, whether or not the upfront listing
        # checks below found anything -- the caller must retry the NEXT tier if a tier's own
        # actual download fails for any reason THIS script didn't anticipate (network blip, a
        # transient index outage, a dependency gap this listing-only check can't see), not just
        # when nothing was listed at all. A build must never simply die because of that; SOURCE
        # is always present as a guaranteed-workable last resort.
        #
        # PYTORCH_PLAN:<pip_pkg_spec_or_EMPTY>|<pip_index_or_EMPTY>|<try_snapshot 0|1>|<arch>|<pytorch_ver_num>|<rocm_release>|<source_repo>|<source_branch>
        # ("|"-delimited, not ":" -- the index URL field itself contains colons)
        PKG_NAME="amd-torch-device-${ROCM_ARCH}"
        ROCM_PYTORCH_BRANCH=$(determine_rocm_pytorch_branch "$PYTORCH_VERSION") || exit 1
        PIP_SPEC=""
        PIP_INDEX=""
        TRY_SNAPSHOT=0

        if [ "$USE_PREBUILT" = "1" ]; then
            if [ -z "$ROCM_RELEASE" ]; then
                PYTORCH_VER_FILTER=""
                if [ -n "$PYTORCH_VERSION" ] && [ "$PYTORCH_VERSION" != "develop" ]; then
                    PYTORCH_VER_FILTER=${PYTORCH_VERSION#v}
                fi
                if VERSION=$(pytorch_find_nightly_version "$ROCM_ARCH" "$PYTORCH_VER_FILTER"); then
                    PIP_SPEC="${PKG_NAME}==${VERSION}"
                    PIP_INDEX="$NIGHTLY_INDEX"
                else
                    log "No nightly wheel listed -- caller falls straight to source build"
                fi
            else
                # Tier 1: repo.amd.com's real, stable release -- a match here is guaranteed
                # fully resolvable, no further checking needed.
                if VERSION=$(pytorch_find_stable_release_version "$ROCM_ARCH" "$PYTORCH_VERSION" "$ROCM_RELEASE"); then
                    PIP_SPEC="${PKG_NAME}==${VERSION}"
                    PIP_INDEX="$STABLE_RELEASE_INDEX"
                else
                    log "No stable release wheel listed on repo.amd.com for pytorch=${PYTORCH_VERSION#v} rocm=${ROCM_RELEASE}"
                fi
                # Tier 2 (rocm-devrelease-snapshot.py -- picks ONE devreleases nightly date
                # where the whole dependency closure resolves, see its own header) is only
                # meaningful in release mode; always offered as the caller's next fallback,
                # regardless of whether tier 1 was listed above, since the ACTUAL tier 1
                # download can still fail for a reason this listing check didn't catch.
                TRY_SNAPSHOT=1
            fi
        fi

        echo "PYTORCH_PLAN:${PIP_SPEC}|${PIP_INDEX}|${TRY_SNAPSHOT}|${ROCM_ARCH}|${PYTORCH_VERSION#v}|${ROCM_RELEASE}|ROCm/pytorch|${ROCM_PYTORCH_BRANCH}"
        ;;

    torchvision)
        PKG_NAME="amd-torchvision-device-${ROCM_ARCH}"
        # Tier 1: a real repo.amd.com release, when torch itself came from there. Must be
        # tried before the devreleases index below -- see stable_torch_version_from_wheel.
        if [ "$USE_PREBUILT" = "1" ] && [ -n "$PYTORCH_RESOLVED_VERSION" ] \
           && STABLE_TORCH_VER=$(stable_torch_version_from_wheel "$PYTORCH_RESOLVED_VERSION" "$ROCM_RELEASE"); then
            VERSION=$(torchvision_find_stable_release_version "$ROCM_ARCH" "$STABLE_TORCH_VER" "$ROCM_RELEASE") && \
                { echo "PIP:${PKG_NAME}==${VERSION}:${STABLE_RELEASE_INDEX}"; exit 0; }
            log "No stable release torchvision on repo.amd.com for torch=${STABLE_TORCH_VER} rocm=${ROCM_RELEASE}"
        fi
        if [ "$USE_PREBUILT" = "1" ] && [ -n "$PYTORCH_RESOLVED_VERSION" ]; then
            DATETAG=$(extract_rocm_datetag "$PYTORCH_RESOLVED_VERSION")
            if [ -n "$DATETAG" ]; then
                INDEX=$([ -z "$ROCM_RELEASE" ] && echo "$NIGHTLY_INDEX" || echo "$RELEASE_INDEX")
                BASE_URL="${INDEX}amd-torchvision-device-${ROCM_ARCH}/"
                VERSION=$(torchvision_find_matching_version "$ROCM_ARCH" "$BASE_URL" "$DATETAG") && \
                    { echo "PIP:${PKG_NAME}==${VERSION}:${INDEX}"; exit 0; }
            fi
            log "No torchvision wheel matching pytorch's date-tag, falling back to source build"
        elif [ "$USE_PREBUILT" = "1" ]; then
            log "pytorch went SOURCE (no resolved version to match) -- torchvision building from source too"
        fi

        # torchvision: use pytorch/vision (not ROCm/vision) for broader OS/version coverage
        REPO_BRANCH=$(determine_torchvision_repo_branch "$PYTORCH_VERSION") || exit 1
        REPO=$(echo "$REPO_BRANCH" | cut -d';' -f1)
        BRANCH=$(echo "$REPO_BRANCH" | cut -d';' -f2)
        echo "SOURCE:${REPO}:${BRANCH}"
        ;;

    torchaudio)
        # torchaudio has no per-arch wheel (same binary for every GPU target) and its own wheel
        # metadata only pins a bare `torch` (no version constraint) -- unlike torchvision it
        # doesn't need an exact same-day match to be ABI-safe, so it gets an extra fallback
        # torchvision doesn't: prefer matching pytorch's date-tag when there is one (tightest
        # pairing), but if pytorch went SOURCE (no date to match) or no matching build exists,
        # fall back to whatever the latest published torchaudio wheel is -- rather than
        # immediately building from source like torchvision does -- before finally falling back
        # to SOURCE (ROCm/audio, latest 2.11 patch) only if no wheel exists at all.
        if [ "$USE_PREBUILT" = "1" ]; then
            # Tier 1: a real repo.amd.com release, when torch itself came from there. This is
            # the path whose absence let a rocm_version=7.14.0 build resolve a
            # "+rocm10.0.0a<date>" nightly -- see stable_torch_version_from_wheel.
            if [ -n "$PYTORCH_RESOLVED_VERSION" ] \
               && STABLE_TORCH_VER=$(stable_torch_version_from_wheel "$PYTORCH_RESOLVED_VERSION" "$ROCM_RELEASE"); then
                VERSION=$(torchaudio_find_stable_release_version "$STABLE_TORCH_VER" "$ROCM_RELEASE") && \
                    { echo "PIP:torchaudio==${VERSION}:${STABLE_RELEASE_INDEX}"; exit 0; }
                log "No stable release torchaudio on repo.amd.com for torch=${STABLE_TORCH_VER} rocm=${ROCM_RELEASE}"
            fi
            INDEX=$([ -z "$ROCM_RELEASE" ] && echo "$NIGHTLY_INDEX" || echo "$RELEASE_INDEX")
            BASE_URL="${INDEX}torchaudio/"
            if [ -n "$PYTORCH_RESOLVED_VERSION" ]; then
                DATETAG=$(extract_rocm_datetag "$PYTORCH_RESOLVED_VERSION")
                if [ -n "$DATETAG" ]; then
                    VERSION=$(torchaudio_find_matching_version "$BASE_URL" "$DATETAG") && \
                        { echo "PIP:torchaudio==${VERSION}:${INDEX}"; exit 0; }
                fi
                log "No torchaudio wheel matching pytorch's date-tag, trying latest available instead"
            else
                log "pytorch went SOURCE (no date to match) -- torchaudio trying latest available wheel"
            fi
            VERSION=$(torchaudio_find_latest_version "$BASE_URL") && \
                { echo "PIP:torchaudio==${VERSION}:${INDEX}"; exit 0; }
            log "No torchaudio wheel available at all, falling back to source build"
        fi

        REPO_BRANCH=$(determine_torchaudio_repo_branch "$PYTORCH_VERSION") || exit 1
        REPO=$(echo "$REPO_BRANCH" | cut -d';' -f1)
        BRANCH=$(echo "$REPO_BRANCH" | cut -d';' -f2)
        echo "SOURCE:${REPO}:${BRANCH}"
        ;;

    *)
        log "Unknown package: $PACKAGE (pytorch|torchvision|torchaudio)"
        exit 1
        ;;
esac
