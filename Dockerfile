# syntax=docker/dockerfile:1
# MIGraphX + ONNX Runtime, built from source against a ROCm release that has
# no matching prebuilt onnxruntime-rocm/MIGraphX package yet (e.g. no
# `rocm/onnxruntime:rocm7.14*` image exists, and MIGraphX isn't part of
# TheRock's package set). Produces a drop-in BASE_IMAGE for downstream
# Dockerfiles that expect the `rocm/onnxruntime` convention: onnxruntime
# (built with MIGraphXExecutionProvider) preinstalled into a venv at
# /opt/venv, with /opt/rocm containing the from-source MIGraphX + its ROCm
# runtime deps (HIP/rocBLAS/MIOpen from the base image, composable_kernel +
# rocMLIR built alongside MIGraphX).
#
# PyTorch is built from source too, in its own stage below: AMD's rocm/pytorch
# image looked like a shortcut, but its current tag ships TheRock's
# pip-packaged ROCm SDK (no /opt/rocm, no /opt/rocm/llvm/bin/clang) instead of
# the classic layout MIGraphX's build expects -- reusing it would mean two
# independent ROCm runtime stacks in one image (TheRock's for torch, classic
# for MIGraphX/ORT), risking conflicting libamdhip64.so builds loaded into the
# same process when the app touches both (it does, see tasks/memory_utils.py).
# One consistent stack is worth the extra build time.
#
# Build: docker build -t <tag> .
#
# Defaults to our own nightly-built ROCm base (see the rocm-builder stage below) rather than
# AMD's own rocm/dev-ubuntu-26.04 image, since AMD publishes no rolling/nightly tag for that
# image at all -- see README.md's "Nightly ROCm versioning" section. The manual release
# workflow explicitly overrides this to the AMD-pinned tag instead (it already overrides every
# other version-related ARG by hand, so one more is no added burden); nightly leaves it at this
# default and gets our own self-built, TheRock-nightly-tracking base.
ARG BASE_IMAGE=ghcr.io/schaka/rocm-builder:latest

# Component-image references. Downstream stages COPY each component's artifacts
# from these, so in CI every component can be built as its own job/image (see
# .github/workflows) and a later stage just pulls the prebuilt result instead
# of recompiling it. Defaults point at the local stage names, so a plain
# `docker build .` on one machine still resolves them to the in-tree stages and
# builds everything in a single shot -- BuildKit treats `COPY --from=<name>` (or
# `FROM <name>`, which is how ROCBLAS_IMAGE is consumed) as a stage reference
# when <name> matches a stage, else as an image to pull.
ARG ROCBLAS_IMAGE=rocblas-builder
ARG MIGRAPHX_IMAGE=migraphx-builder
ARG PYTORCH_IMAGE=pytorch-builder
ARG TORCHVISION_IMAGE=torchvision-builder
ARG TORCHAUDIO_IMAGE=torchaudio-builder
ARG ORT_IMAGE=ort-builder

# Git ref to build MIGraphX from. Defaults to the moving `develop` branch;
# override to pin a stable release branch (e.g.
# release/rocm-rel-7.13) when develop regresses on a given GPU target and you
# need a known-good build instead. Reflected in the CI image tag (see
# nightly.yml) so a pinned build doesn't collide with/get overwritten by the
# develop-tracking one.
ARG MIGRAPHX_REF=develop

# rocBLAS release tag to rebuild from for gfx900/gfx906/gfx90c (see
# rocblas-builder below) -- must match BASE_IMAGE's ROCm release, since rocBLAS
# built against a different release than the hipBLAS/rocSOLVER/etc it links
# against is not a supported combination. AMD's prebuilt rocBLAS package for
# this ROCm line ships no gfx900/gfx906/gfx90c code objects at all (confirmed:
# no amdrocm-blas*-gfx900/-gfx906/-gfx90c apt package exists, and the base
# image's Tensile library on disk has no gfx900/gfx906/gfx90c folder) -- but
# the *source* still carries them (rocBLAS's own TARGET_LIST_ROCM_7.1 in
# CMakeLists.txt still lists gfx900;gfx906:xnack-, and its Tensile Logic tree
# still has vega10/vega20 folders), so rebuilding covers them without a
# ROCm-major downgrade, unlike gfx803 (which needs one for CDNA... err
# Polaris's actual ROCR enumeration break on ROCm 7).
#
# Same ARG pytorch/torchvision/torchaudio's own wheel-discovery below reads
# (major.minor, e.g. "7.14"; empty = nightly float). rocblas-builder builds
# from rocm-libraries' own `develop` branch when this is empty, matching
# MIGraphX's own default, instead of a pinned `therock-<release>` tag.
ARG ROCM_RELEASE=

# Single source of truth for which arches need the rocBLAS-from-source /
# composable_kernel-off / hipBLASLt-off special-casing scattered through
# rocblas-builder and migraphx-builder below (AMD's prebuilt packages for
# this ROCm line ship no gfx900/gfx906/gfx90c code objects at all, in rocBLAS,
# hipBLASLt, or composable_kernel). A shell `case ... in ${LEGACY_GCN_ARCHES})`
# pattern, not a real list -- Docker build stages don't share shell state
# across RUN layers, so this can't eliminate the repeated `case` scaffolding
# itself, only keep the actual arch set in one place. Extend here if another
# arch needs the same treatment.
ARG LEGACY_GCN_ARCHES="gfx900|gfx906|gfx90c"

# Runtime preference passed straight through to the final image's ENV.
# Meaningless everywhere except gfx900/gfx906/gfx90c: PyTorch links hipBLASLt
# unconditionally regardless of arch (cmake/Dependencies.cmake), but
# hipBLASLt's own Tensile Logic tree has never had gfx900/gfx906/gfx90c kernels
# (oldest arch present is arcturus/gfx908) -- 0 forces GEMMs onto the
# rocBLAS rebuilt below instead of a library with nothing to dispatch to.
# 1 (the default) matches upstream's own default preference and is a
# no-op on every other arch, which does have hipBLASLt kernels.
ARG TORCH_BLAS_PREFER_HIPBLASLT=1

# Cache-bust token for the develop-tracking MIGraphX clone below. A moving
# branch's `git clone` layer has a cache key that never changes, so it would
# cache-hit forever and keep rebuilding the same stale commit every night --
# CI passes the build date here so each nightly run re-clones develop. Any
# changing value works; left constant locally (pinned deps like PyTorch/ORT
# don't need it -- their tags already change the cache key when bumped). Not
# needed when MIGRAPHX_REF pins a fixed branch/tag, since that ref's own
# commits already change the cache key when it moves.
ARG SOURCE_DATE=unknown

# Self-built ROCm base from TheRock's nightly .deb feed -- what BASE_IMAGE defaults to above.
# CI builds and publishes this as its own component (ghcr.io/<owner>/rocm-builder:latest and
# :<date>, see nightly.yml), same pattern as rocblas-builder/migraphx-builder/etc below, just
# arch-independent (amdrocm-hpc-sdk pulls in every gfx target's runtime bits in one package, so
# there's exactly one of these, not one per arch).
#
# Validated end-to-end: a real gfx1201 build ran clean through migraphx-builder and
# pytorch-builder against this stage. Package layout under "amdrocm-*" does differ from
# rocm/dev-ubuntu-26.04 (see the two fixes below -- top-level symlinks and the -dev packages).
#
# No GPG verification ([trusted=yes]) -- this is an internal nightly-only path, not the release
# path (which explicitly overrides BASE_IMAGE to AMD's own signed rocm/dev-ubuntu image instead);
# acceptable for the same reason nightly already accepts cross-component date drift (README.md).
FROM ubuntu:26.04 AS rocm-builder
ARG THEROCK_DEB_INDEX=https://rocm.nightlies.amd.com/packages-multi-arch/deb/
# Referenced (as a no-op echo) purely to bust this RUN's layer cache -- same reasoning as
# MIGraphX's SOURCE_DATE usage above: the RUN command's own text never changes, so without this,
# `cache-from` would replay whatever dated snapshot was resolved the very first time this stage
# was ever built, forever. CI passes today's date here on every nightly run.
ARG SOURCE_DATE
RUN set -eux; \
    echo "rocm-builder snapshot: ${SOURCE_DATE}"; \
    apt-get update && apt-get install -y --no-install-recommends ca-certificates curl gnupg2 \
    && rm -rf /var/lib/apt/lists/*; \
    # TheRock's deb feed has no "latest" alias, only dated YYYYMMDD-<run-id> directories -- same
    # "resolve newest yourself" problem the wheel-index discovery functions above solve, just for
    # a directory listing instead of wheel filenames. The bare index URL serves a stale cached
    # snapshot (observed capped at a months-old page); a throwaway query string busts that cache
    # and gets the real, current listing back.
    latest=$(curl -s "${THEROCK_DEB_INDEX}?_=$(date +%s)" | grep -oE 'href="[0-9]{8}-[0-9]+/index.html"' \
        | sed 's/href="//; s|/index.html"||' | sort -V | tail -1); \
    if [ -z "$latest" ]; then echo "FATAL: no dated build dir found under ${THEROCK_DEB_INDEX}" >&2; exit 1; fi; \
    echo "Using TheRock nightly deb snapshot: ${latest}"; \
    echo "deb [trusted=yes] ${THEROCK_DEB_INDEX}${latest}/ stable main" > /etc/apt/sources.list.d/therock-nightly.list; \
    # amdrocm-hpc-sdk pulls "-dev" packages for peripheral libs (rocALUTION, hipTensor) but NOT
    # for the core HIP runtime itself -- confirmed by `dpkg -l` after install: amdrocm-runtime10.1
    # (the runtime .so) was present, amdrocm-runtime-dev/amdrocm-core-dev (headers + HIP's own
    # hip-config.cmake) were not, which is exactly why rocMLIR's cmake configure step below fails
    # with "Could not find a package configuration file provided by hip". Install them explicitly.
    apt-get update && apt-get install -y --no-install-recommends \
        amdrocm-hpc-sdk amdrocm-core-dev amdrocm-runtime-dev \
    && rm -rf /var/lib/apt/lists/*; \
    # amdrocm-hpc-sdk lands everything under /opt/rocm/core-<major.minor>/ (bin, include, lib,
    # libexec, llvm, share, amdgcn) with NO top-level convenience symlinks -- unlike
    # rocm/dev-ubuntu-26.04, which symlinks include/lib/share/etc at /opt/rocm/ straight into its
    # own versioned core dir (see rocblas-builder's comment below for that same layout on the
    # official image). Every other stage in this Dockerfile hardcodes paths like
    # /opt/rocm/llvm/bin/clang++ and /opt/rocm/lib -- recreate those top-level symlinks here so
    # this base is a drop-in for what they expect, instead of patching every consumer.
    core_dir=$(find /opt/rocm -mindepth 1 -maxdepth 1 -type d -name 'core-*' | head -1); \
    if [ -z "$core_dir" ]; then echo "FATAL: no /opt/rocm/core-* dir found after install" >&2; exit 1; fi; \
    for d in "$core_dir"/*; do \
        ln -s "$(basename "$core_dir")/$(basename "$d")" "/opt/rocm/$(basename "$d")"; \
    done

# Shared ancestor for every stage below: this base image's native python3 is
# 3.14, but AudioMuse-AI (and other downstream consumers) pin numpy in a way
# that only resolves against onnx's deps under 3.12 -- so every wheel built
# below, and the final venv, all target uv-managed Python 3.12 instead.
# Installing uv and downloading that interpreter here, once, and having every
# other stage FROM this one (rather than FROM ${BASE_IMAGE} directly) avoids
# re-fetching the same ~30MB standalone interpreter in 4 independent stages.
FROM ${BASE_IMAGE} AS python-base
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh \
    && uv python install 3.12

# Rebuilds rocBLAS for gfx900/gfx906/gfx90c only; every other arch passes
# through unchanged (this stage is a no-op for them -- /opt/rocm keeps the base
# image's own prebuilt rocBLAS, which does have their kernels). Both
# migraphx-builder and pytorch-builder build on ${ROCBLAS_IMAGE} instead of
# python-base directly, so both see the rebuilt rocBLAS when it applies.
# CI overrides ROCBLAS_IMAGE with the published rocm-rocblas-builder:<arch> on
# those three arches, so the rebuild runs once in its own job instead of once
# per dependent job; every other arch leaves it at the stage name above.
#
# Only fires for a single-arch build (ROCM_ARCH exactly "gfx900", "gfx906",
# or "gfx90c"), which is what CI always passes (see build-component.yml). A
# local one-shot `docker build .` using the default multi-arch ROCM_ARCH
# list won't match the case below and won't rebuild -- that default is a
# convenience fallback for arches that don't need this fix, not how
# gfx900/gfx906/gfx90c are actually meant to be built.
FROM python-base AS rocblas-builder

ARG ROCM_ARCH
ARG ROCM_RELEASE
ARG BUILD_PARALLEL_LEVEL=auto
ARG LEGACY_GCN_ARCHES

# rocBLAS's standalone repo (ROCm/rocBLAS) stopped at v14.3.0 and is
# deprecated -- current development moved into the ROCm/rocm-libraries
# monorepo, tagged "therock-<major>.<minor>" (no patch component -- exactly
# what ROCM_RELEASE already is). Empty ROCM_RELEASE (nightly, fully floating)
# builds from the monorepo's own `develop` branch instead of a pinned tag,
# matching MIGraphX's own default below. Confirmed at the therock-7.14 tag
# specifically (not just develop): projects/rocblas/
# CMakeLists.txt's TARGET_LIST_ROCM_7.13 still lists gfx900;gfx906:xnack-,
# and projects/rocblas/library/src/blas3/Tensile/Logic/asm_full/ still has
# vega10 (gfx900) and vega20 (gfx906) folders. Tensile is no longer a git
# submodule of rocblas -- CMakeLists.txt resolves it from
# ${CMAKE_CURRENT_SOURCE_DIR}/../../shared/tensile, i.e. the monorepo's
# shared/tensile at the same tag, which is why the sparse-checkout below
# pulls that alongside projects/rocblas rather than rocblas alone.
RUN --mount=type=cache,target=/root/.ccache,id=rocblas-legacy-ccache \
    set -eux; \
    # Matches LEGACY_GCN_ARCHES (a "|"-joined list, see its ARG comment near
    # the top of this file) against ROCM_ARCH -- not a `case` pattern, since
    # a variable's expanded value can't supply case's own `|` alternation
    # syntax (that's parsed as literal grammar, not built from a runtime
    # string). Padding both sides with "|" avoids "gfx90" false-matching
    # "gfx900".
    if ! echo "|${LEGACY_GCN_ARCHES}|" | grep -q "|${ROCM_ARCH}|"; then \
        echo "rocblas-builder: ${ROCM_ARCH} uses the base image's prebuilt rocBLAS, no rebuild needed"; \
        exit 0; \
    fi; \
    apt-get update && apt-get install -y --no-install-recommends \
        git cmake ninja-build build-essential pkg-config gfortran ccache \
        libmsgpack-cxx-dev wget python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*; \
    # Tensile's find_package(msgpackc-cxx CONFIG) wants
    # msgpackc-cxx(-config|Config).cmake -- Debian's package installs the
    # same content one letter off, as msgpack-cxx-config.cmake (project
    # name "msgpack-cxx", not "msgpackc-cxx"; libmsgpack-dev itself is just
    # a transitional dummy pulling in the plain-C libmsgpack-c-dev, which
    # has no CMake config at all). Symlink under a name CMake's default
    # PREFIX/lib/cmake/<pkg> search convention actually matches, rather
    # than patching Tensile's CMakeLists for a packaging-naming quirk.
    mkdir -p /usr/local/lib/cmake/msgpackc-cxx; \
    for f in /usr/lib/x86_64-linux-gnu/cmake/msgpack-cxx/msgpack-cxx-*.cmake; do \
        ln -s "$f" "/usr/local/lib/cmake/msgpackc-cxx/$(basename "$f" | sed 's/^msgpack-cxx/msgpackc-cxx/')"; \
    done; \
    pip3 install --break-system-packages --no-cache-dir pyyaml joblib; \
    if [ -z "${ROCM_RELEASE}" ]; then monorepo_ref="develop"; else monorepo_ref="therock-${ROCM_RELEASE}"; fi; \
    echo "rocBLAS source: ROCm/rocm-libraries @ ${monorepo_ref} (projects/rocblas)"; \
    git clone --filter=blob:none --depth 1 --no-checkout \
        --branch "${monorepo_ref}" \
        https://github.com/ROCm/rocm-libraries.git /rocm-libraries-src; \
    cd /rocm-libraries-src; \
    git sparse-checkout init --cone; \
    git sparse-checkout set cmake shared/tensile projects/rocblas; \
    git checkout "${monorepo_ref}"; \
    cd /rocm-libraries-src/projects/rocblas; \
    jobs="${BUILD_PARALLEL_LEVEL}"; \
    if [ "$jobs" = "auto" ]; then \
        jobs=$(awk '/MemAvailable/{printf "%d", $2/1024/1024/4}' /proc/meminfo); \
        cpu=$(nproc); [ "$jobs" -gt "$cpu" ] && jobs=$cpu; \
        [ "$jobs" -lt 1 ] && jobs=1; \
    fi; \
    echo "rocBLAS build: arch ${ROCM_ARCH}, $jobs parallel jobs"; \
    python3 ./rmake.py -i -a "${ROCM_ARCH}" -j "$jobs" --no_hipblaslt; \
    # rmake's -i install target lands under build/release/rocblas-install,
    # not /opt/rocm -- same install-path quirk as gfx803/Dockerfile.gfx803,
    # see the comments there for the full story (relative CMAKE_INSTALL_PREFIX,
    # and why --cleanup is never passed).
    echo "Copying rocBLAS ${ROCM_ARCH} install output into /opt/rocm..."; \
    # This base image lays /opt/rocm out as versioned component dirs
    # (core, core-7, core-7.14) with include/lib/share at the top level as
    # convenience symlinks into core-7.14/ -- unlike gfx803's older, flatter
    # base image, where a plain `cp -a src/. /opt/rocm/` (the same command
    # used here originally) is enough. Here it fails outright: cp won't
    # overwrite an existing symlink-named entry with a real directory
    # ("cannot overwrite non-directory ... with directory"). Resolve each
    # of include/lib/share to what it actually points at first, so content
    # lands in core-7.14/ and the top-level symlinks keep working, instead
    # of cp clobbering them.
    src="/rocm-libraries-src/projects/rocblas/build/release/rocblas-install"; \
    for d in include lib share; do \
        [ -e "$src/$d" ] || continue; \
        real_dest="$(readlink -f "/opt/rocm/$d" 2>/dev/null || echo "/opt/rocm/$d")"; \
        mkdir -p "$real_dest"; \
        cp -a "$src/$d/." "$real_dest/"; \
    done; \
    find "$src" -mindepth 1 -maxdepth 1 ! -name include ! -name lib ! -name share \
        -exec cp -a {} /opt/rocm/ \; ; \
    # Captured before the rm -rf below deletes $src, so there's still
    # something to compare librocblas.so's resolved target against afterward.
    built_real="$(find "$src/lib" -maxdepth 1 -name 'librocblas.so.*' ! -type l | head -1)"; \
    built_size="$(stat -c%s "$built_real" 2>/dev/null || echo 0)"; \
    rm -rf /rocm-libraries-src; \
    echo "Verifying ${ROCM_ARCH} Tensile library is present in /opt/rocm..."; \
    if ! find -L /opt/rocm -iname "*TensileLibrary*${ROCM_ARCH}*" | grep -q .; then \
        echo "FATAL: /opt/rocm has no ${ROCM_ARCH} Tensile library after the copy." >&2; \
        exit 1; \
    fi; \
    echo "OK: ${ROCM_ARCH} Tensile library confirmed present in /opt/rocm."; \
    echo "Verifying librocblas.so resolves to the ${ROCM_ARCH} build, not the stock one..."; \
    resolved="$(readlink -f /opt/rocm/lib/librocblas.so)"; \
    if [ "$built_size" = "0" ] || [ "$(stat -c%s "$resolved")" != "$built_size" ]; then \
        echo "FATAL: /opt/rocm/lib/librocblas.so resolves to ${resolved}, which is not our freshly-built rocBLAS (size mismatch)." >&2; \
        echo "The stock base-image rocBLAS is still what actually gets loaded at runtime." >&2; \
        exit 1; \
    fi; \
    echo "Verifying librocblas.so embeds real ${ROCM_ARCH} device code..."; \
    objcopy -O binary --only-section=.hip_fatbin "$resolved" /tmp/rocblas_fatbin_check.bin; \
    fatbin_size="$(stat -c%s /tmp/rocblas_fatbin_check.bin)"; \
    rm -f /tmp/rocblas_fatbin_check.bin; \
    if [ "$fatbin_size" -lt 1000000 ]; then \
        echo "FATAL: librocblas.so's .hip_fatbin is only ${fatbin_size} bytes -- too small to contain real ${ROCM_ARCH} device code (expect several MB)." >&2; \
        exit 1; \
    fi; \
    echo "OK: librocblas.so resolves to a ${ROCM_ARCH} build with a ${fatbin_size}-byte .hip_fatbin."

FROM ${ROCBLAS_IMAGE} AS migraphx-builder

# Semicolon-separated GPU_TARGETS list, matching the breadth AMD's own
# published images build for (CDNA1-3, RDNA2-4), not just this host's GPU.
# Narrow it via --build-arg if you only need one target and want a faster
# build.
ARG ROCM_ARCH="gfx900;gfx90c;gfx906;gfx908;gfx90a;gfx942;gfx950;gfx1010;gfx1011;gfx1012;gfx1030;gfx1031;gfx1032;gfx1034;gfx1035;gfx1036;gfx1100;gfx1101;gfx1102;gfx1103;gfx1150;gfx1151;gfx1152;gfx1153;gfx1200;gfx1201"
ARG LEGACY_GCN_ARCHES

RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake ninja-build build-essential pkg-config ccache \
        python3 python3-dev python3-pybind11 \
    && rm -rf /var/lib/apt/lists/*

# TheRock's bundled rocm-cmake modules under /opt/rocm are stale/incomplete
# for building components outside TheRock itself (e.g. missing the
# rocm_add_version_resource macro AMDMIGraphX's CMakeLists.txt needs) --
# install a fresh copy over the same prefix.
RUN git clone --branch develop --depth 1 \
        https://github.com/ROCm/rocm-cmake.git /rocm-cmake-src \
    && cmake -S /rocm-cmake-src -B /rocm-cmake-src/build -DCMAKE_INSTALL_PREFIX=/opt/rocm \
    && cmake --build /rocm-cmake-src/build --target install \
    && rm -rf /rocm-cmake-src

# Use MIGraphX's own documented from-source build path (rbuild, per
# docs/install/install-migraphx.rst on develop), instead of hand-rolling
# each third-party dependency: rbuild reads requirements.txt and builds
# every dependency itself at the exact pinned commit MIGraphX's CMakeLists.txt
# expects -- notably ROCm/composable_kernel's `codegen` subdir specifically,
# which is what actually produces the composable_kernel_host CMake package
# (not the same artifact as the plain "composable_kernel" apt package), and
# ROCm/rocMLIR built from source, since neither has a matching apt package
# for this ROCm release.
#
# rbuild's dependency `cget` subclasses urllib.request.FancyURLopener, which
# Python 3.14 (this image's only python3) removed outright -- unrelated to
# MIGraphX itself, just this base image shipping a python newer than that
# old tool supports. Give rbuild/cget an isolated Python 3.12 via uv instead
# of touching the system interpreter or MIGraphX's own build.
#
# This same venv doubles as the interpreter MIGraphX's python module is built
# against (passed explicitly as Python3_EXECUTABLE below -- PATH order alone
# doesn't decide what CMake's FindPython3 picks, and the apt python3-dev 3.14
# would win otherwise), so the module's ABI tag lines up with the Python 3.12
# that pytorch-builder/ort-builder/the final stage all use -- downstream apps
# (e.g. AudioMuse-AI) pin numpy in a way that only resolves against onnx's
# deps under 3.12, not this base image's native 3.14.
RUN uv venv /rbuild-venv --python 3.12 \
    && uv pip install --python /rbuild-venv/bin/python3 \
        https://github.com/RadeonOpenCompute/rbuild/archive/master.tar.gz

# SOURCE_DATE (see top of file) is referenced here purely to bust this layer's
# cache when it changes -- develop is a moving branch, so without it the clone
# would reuse a stale commit on every nightly rebuild. Its downstream compile
# layer busts with it, which is intended: fresh develop each night.
ARG SOURCE_DATE
ARG MIGRAPHX_REF
RUN echo "MIGraphX ${MIGRAPHX_REF} snapshot: ${SOURCE_DATE}" \
    && git clone --branch "${MIGRAPHX_REF}" --depth 1 \
        https://github.com/ROCm/AMDMIGraphX.git /migraphx-src

WORKDIR /migraphx-src

# composable_kernel explicitly denylists gfx900/gfx906 (its CMakeLists.txt:
# CK_UNSUPPORTED_GPU_TARGETS="gfx900;gfx906;gfx90c" -- a single-target
# request against that list generates a dummy target and returns, per CK's
# own CMakeLists, rather than building anything real). Turning it off via
# -DMIGRAPHX_USE_COMPOSABLEKERNEL=Off alone is not sufficient: rbuild builds
# everything requirements.txt lists regardless of MIGraphX's own cmake
# options (same gotcha gfx803/Dockerfile.gfx803 documents for its own
# rocMLIR/CK exclusion), so it would still try to build CK's dummy target
# for these two arches. Strip the line so rbuild never attempts it.
RUN if echo "|${LEGACY_GCN_ARCHES}|" | grep -q "|${ROCM_ARCH}|"; then \
        sed -i '/composable_kernel/d' requirements.txt \
        && ! grep -q 'composable_kernel' requirements.txt; \
    fi

# Upstream bug (filed: see gfx900-906-gfx-default-rocblas-bug-report.md),
# reproduced independently of any of our own build flags/caching:
# device_name.hpp guards gfx_default_rocblas()'s DECLARATION behind
# `#if MIGRAPHX_USE_HIPBLASLT`, but lowering.cpp's one call site has no
# matching guard, so -DMIGRAPHX_USE_HIPBLASLT=Off (required on gfx900/gfx906/gfx90c
# -- hipBLASLt has no kernels for any of them) fails to compile outright:
# "no member named 'gfx_default_rocblas' in namespace 'migraphx::gpu'".
# hipblaslt_supported() itself already returns a hardcoded false with the
# flag off, which alone makes the enclosing `or` chain unconditionally true
# at runtime regardless of gfx_default_rocblas() -- so replacing the call
# with a literal `true` under the same guard is a semantics-preserving fix,
# not a behavior change. Confirmed this is the only unguarded call site in
# actually-compiled code (the other two references are in test/, excluded by
# -DBUILD_TESTING=Off below).
RUN if echo "|${LEGACY_GCN_ARCHES}|" | grep -q "|${ROCM_ARCH}|"; then \
        sed -i \
            's/not hipblaslt_supported() or gpu::gfx_default_rocblas()/not hipblaslt_supported()/' \
            src/targets/gpu/lowering.cpp \
        && grep -q 'not hipblaslt_supported()' src/targets/gpu/lowering.cpp \
        && ! grep -q 'gfx_default_rocblas' src/targets/gpu/lowering.cpp; \
    fi

# rbuild shells out to a bare `cget` (relies on PATH, not sys.executable), so
# the venv's bin dir must be on PATH, not just invoked via absolute path.
#
# rocMLIR's LLVM build: each -O3 clang job doing IPO/codegen on LLVM's own
# sources needs several GB RSS. Default ninja parallelism (= nproc) can
# exceed available RAM on the build host, which surfaces as clang segfaulting
# on a different random file each time (or the OOM killer taking out
# unrelated host processes) rather than a clean failure -- cap the job count
# so peak memory stays bounded. Default "auto" sizes it from MemAvailable at
# ~4GB/job, capped at nproc; pass --build-arg BUILD_PARALLEL_LEVEL=<n> for an
# explicit count instead (e.g. on a CI runner with known, dedicated RAM).
ARG BUILD_PARALLEL_LEVEL=auto
# Cache mount (not an image layer) persists ccache's compiled-object cache
# across a *failed* RUN retry -- a plain layer is only committed on success,
# so without this a segfault at job 359/658 (like the one that motivated the
# MemAvailable-based job cap above) would otherwise force recompiling
# everything from zero on the next attempt. Can't also cache-mount the `build`
# dir itself: rbuild renames/recreates it between dependency phases (LLVM,
# then MIGraphX proper), which fails against a bind-mounted mountpoint.
RUN --mount=type=cache,target=/root/.ccache,id=migraphx-ccache \
    ulimit -s unlimited && \
    jobs="${BUILD_PARALLEL_LEVEL}"; \
    if [ "$jobs" = "auto" ]; then \
        jobs=$(awk '/MemAvailable/{printf "%d", $2/1024/1024/4}' /proc/meminfo); \
        cpu=$(nproc); [ "$jobs" -gt "$cpu" ] && jobs=$cpu; \
        [ "$jobs" -lt 1 ] && jobs=1; \
    fi; \
    echo "rocMLIR/LLVM build: using $jobs parallel jobs"; \
    # hipBLASLt's Tensile Logic tree has never had gfx900/gfx906/gfx90c kernels
    # (oldest arch present is arcturus/gfx908, same fact that motivates
    # rocblas-builder above) -- link against the rocBLAS rebuilt there
    # instead of a hipBLASLt with nothing to dispatch to on this hardware.
    # composable_kernel is denylisted for all three by CK's own CMakeLists (see
    # the requirements.txt strip above) -- MIGRAPHX_USE_COMPOSABLEKERNEL=Off
    # keeps MIGraphX's own cmake from expecting the CK package that was
    # never built.
    extra_cmake_args=""; \
    if echo "|${LEGACY_GCN_ARCHES}|" | grep -q "|${ROCM_ARCH}|"; then \
        extra_cmake_args="-DMIGRAPHX_USE_HIPBLASLT=Off -DMIGRAPHX_USE_COMPOSABLEKERNEL=Off"; \
    fi; \
    CMAKE_BUILD_PARALLEL_LEVEL=$jobs \
    PATH=/rbuild-venv/bin:$PATH /rbuild-venv/bin/rbuild build -d /migraphx-deps -B build -G Ninja \
        --cxx=/opt/rocm/llvm/bin/clang++ --cc=/opt/rocm/llvm/bin/clang \
        "-DGPU_TARGETS=${ROCM_ARCH}" \
        -DCMAKE_INSTALL_PREFIX=/opt/rocm \
        -DCMAKE_BUILD_TYPE=Release \
        -DMIGRAPHX_ENABLE_PYTHON=On \
        -DPython3_EXECUTABLE=/rbuild-venv/bin/python3 \
        -DBUILD_TESTING=Off \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        ${extra_cmake_args} \
        -T install

# Stamp the built ref + resolved commit into the image so downstream
# consumers (e.g. AudioMuse-AI's entrypoint) can detect a MIGraphX change
# across BASE_IMAGE bumps and invalidate their compiled-model cache instead
# of silently recompiling against a stale cache forever.
RUN echo "${MIGRAPHX_REF} $(git -C /migraphx-src rev-parse HEAD)" > /opt/rocm/migraphx-version.txt

FROM ${ROCBLAS_IMAGE} AS pytorch-builder

# torch-package-build-decide.sh (see its own invocation below) checks AMD's
# per-arch nightly/release wheel indices first and falls back to a from-source
# build only when no matching wheel exists there -- doesn't need MIGraphX,
# only the HIP/rocBLAS/MIOpen/rocRAND stack already in this base image (or the
# gfx900/gfx906/gfx90c rebuild from rocblas-builder above, on those arches).
#
# Source-build fallback uses ROCm/pytorch fork, not upstream pytorch/pytorch.
# AMD's fork carries ROCm fixes and a newer composable_kernel submodule that
# upstream lags on.
ARG ROCM_ARCH="gfx900;gfx90c;gfx906;gfx908;gfx90a;gfx942;gfx950;gfx1010;gfx1011;gfx1012;gfx1030;gfx1031;gfx1032;gfx1034;gfx1035;gfx1036;gfx1100;gfx1101;gfx1102;gfx1103;gfx1150;gfx1151;gfx1152;gfx1153;gfx1200;gfx1201"
ARG PYTORCH_VERSION=v2.13.0
ARG BUILD_PARALLEL_LEVEL=auto
ARG USE_PREBUILT_PYTORCH=1
ARG ROCM_RELEASE=
ARG LEGACY_GCN_ARCHES

RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake ninja-build build-essential \
        libopenblas-dev pkg-config libdrm-dev ccache \
    && rm -rf /var/lib/apt/lists/*

# Same reasoning as migraphx-builder's /rbuild-venv: this base image's native
# python3 is 3.14, but the wheel this stage produces has to load into the
# final stage's venv, which is pinned to 3.12 for AudioMuse-AI's numpy/onnx
# pin compatibility -- so build against that same 3.12, not the system one.
RUN uv venv /build-venv --python 3.12 --seed \
    && uv pip install --python /build-venv/bin/python3 \
        numpy pyyaml typing_extensions requests setuptools wheel six
ENV PATH=/build-venv/bin:$PATH

# torch-package-build-decide.sh outputs, for pytorch:
#   PYTORCH_PLAN:<pip_pkg_spec_or_EMPTY>|<pip_index_or_EMPTY>|<try_snapshot 0|1>|<arch>|<pytorch_ver_num>|<rocm_release>|<source_repo>|<source_branch>
# ("|"-delimited, not ":" -- the index URL field itself contains colons)
# ALL fields are always present, regardless of whether the upfront listing checks in that
# script found anything -- the RUN step below actually attempts each tier in order and moves
# on if a tier's real command fails, for whatever reason (not just "nothing was listed"): a
# transient index outage or an unanticipated dependency gap must not just kill the build when
# a working fallback (down to SOURCE, always present) exists.
# mkdir -p /pytorch/dist keeps every path's output in the same place so downstream
# COPY --from=pytorch-builder /pytorch/dist/*.whl keeps working regardless of which tier fired.
COPY scripts/torch-package-build-decide.sh scripts/rocm-devrelease-snapshot.py /tmp/
RUN chmod +x /tmp/torch-package-build-decide.sh /tmp/rocm-devrelease-snapshot.py && \
    /tmp/torch-package-build-decide.sh pytorch "${PYTORCH_VERSION}" "${ROCM_ARCH}" "${USE_PREBUILT_PYTORCH}" "${ROCM_RELEASE}" > /tmp/pytorch-decision.txt && \
    cat /tmp/pytorch-decision.txt

RUN --mount=type=cache,target=/root/.ccache,id=pytorch-ccache \
    mkdir -p /pytorch/dist && \
    clear_dist() { rm -f /pytorch/dist/*.whl /pytorch/dist/*.tar.gz; }; \
    build_pytorch_from_source() { \
        REPO=$1; BRANCH=$2; \
        echo "Building PyTorch from source (repo: $REPO, branch: $BRANCH)" && \
        git clone --recursive --branch "${BRANCH}" --depth 1 --shallow-submodules \
            "https://github.com/${REPO}.git" /pytorch-src && \
        cd /pytorch-src && \
        python3 tools/amd_build/build_amd.py && \
        ulimit -s unlimited && \
        jobs="${BUILD_PARALLEL_LEVEL}"; \
        if [ "$jobs" = "auto" ]; then \
            jobs=$(awk '/MemAvailable/{printf "%d", $2/1024/1024/4}' /proc/meminfo); \
            cpu=$(nproc); [ "$jobs" -gt "$cpu" ] && jobs=$cpu; \
            [ "$jobs" -lt 1 ] && jobs=1; \
        fi; \
        echo "PyTorch build: using $jobs parallel jobs"; \
        ck_gemm=1; \
        if echo "|${LEGACY_GCN_ARCHES}|" | grep -q "|${ROCM_ARCH}|"; then \
            echo "Legacy GCN arch (${ROCM_ARCH}): composable_kernel denylists" \
                 "gfx900/gfx906/gfx90c outright (undeclared CK_BUFFER_RESOURCE_3RD_DWORD)" \
                 "-- disabling PyTorch's own CK-based bgemm kernels (USE_ROCM_CK_GEMM)."; \
            ck_gemm=0; \
        fi; \
        env USE_ROCM=1 ROCM_HOME=/opt/rocm "PYTORCH_ROCM_ARCH=${ROCM_ARCH}" \
            MAX_JOBS=$jobs USE_MKLDNN=0 USE_CCACHE=1 \
            USE_FLASH_ATTENTION=0 USE_MEM_EFF_ATTENTION=0 \
            USE_DISTRIBUTED=0 USE_ROCM_CK_GEMM=$ck_gemm \
            python3 setup.py bdist_wheel && \
        cp dist/*.whl /pytorch/dist/; \
    }; \
    DECISION=$(cat /tmp/pytorch-decision.txt) && \
    REST="${DECISION#PYTORCH_PLAN:}" && \
    PIP_SPEC=$(echo "$REST" | cut -d'|' -f1) && \
    PIP_INDEX=$(echo "$REST" | cut -d'|' -f2) && \
    TRY_SNAPSHOT=$(echo "$REST" | cut -d'|' -f3) && \
    PLAN_ARCH=$(echo "$REST" | cut -d'|' -f4) && \
    PLAN_PYTORCH_VER=$(echo "$REST" | cut -d'|' -f5) && \
    PLAN_ROCM_RELEASE=$(echo "$REST" | cut -d'|' -f6) && \
    PLAN_SOURCE_REPO=$(echo "$REST" | cut -d'|' -f7) && \
    PLAN_SOURCE_BRANCH=$(echo "$REST" | cut -d'|' -f8) && \
    got_it=0 && \
    if [ -n "$PIP_SPEC" ]; then \
        echo "Downloading prebuilt PyTorch: $PIP_SPEC (+ deps) from $PIP_INDEX" && \
        if /build-venv/bin/pip download "$PIP_SPEC" --index-url "$PIP_INDEX" -d /pytorch/dist/; then \
            got_it=1; \
        else \
            echo "Prebuilt download failed -- trying the next fallback tier instead of failing the build" && \
            clear_dist; \
        fi; \
    fi; \
    if [ "$got_it" = "0" ] && [ "$TRY_SNAPSHOT" = "1" ]; then \
        if /build-venv/bin/python3 /tmp/rocm-devrelease-snapshot.py \
                /pytorch/dist cp312 linux_x86_64 "$PLAN_ARCH" "$PLAN_PYTORCH_VER" "$PLAN_ROCM_RELEASE"; then \
            got_it=1; \
        else \
            echo "No usable devreleases snapshot either -- trying the next fallback tier instead of failing the build" && \
            clear_dist; \
        fi; \
    fi; \
    if [ "$got_it" = "0" ]; then \
        build_pytorch_from_source "$PLAN_SOURCE_REPO" "$PLAN_SOURCE_BRANCH"; \
    fi

# Alias declared here (not down by migraphx-export/ort-export where the other
# *-export aliases live) so torchvision-builder/torchaudio-builder below can
# reference it too. PYTORCH_IMAGE defaults to the local stage name
# "pytorch-builder" (see the ARG's own declaration near the top of this file),
# so a plain local `docker build .` still resolves this to that in-tree stage
# with zero behavior change.
#
# torchvision-builder/torchaudio-builder used to COPY --from=pytorch-builder
# directly -- the LOCAL stage, always rebuilt from scratch, ignoring
# PYTORCH_IMAGE entirely. That meant CI's "final" job (which always sets
# PYTORCH_IMAGE to the already-published pytorch component image) redundantly
# rebuilt pytorch from source/PIP a SECOND time just to feed torchvision/
# torchaudio -- wasted build time, and worse, no guarantee the two builds
# agree: torch-package-build-decide.sh's discovery can resolve a different
# wheel on the second run (different moment, same floating nightly index),
# producing two non-identical torch wheels under the same version number.
# uv then refuses to install both ("conflicting URLs for package `torch`")
# once torchvision's own dist dir (which re-bundles the torch wheel it built
# against) and pytorch-export's dist land in the same directory below.
# Aliasing through pytorch-export instead guarantees torchvision/torchaudio
# build against the EXACT SAME torch artifact the final stage installs -- and
# skips rebuilding it entirely when PYTORCH_IMAGE points at a prebuilt image.
FROM ${PYTORCH_IMAGE} AS pytorch-export

FROM ${ROCBLAS_IMAGE} AS torchvision-builder

ARG ROCM_ARCH="gfx900;gfx90c;gfx906;gfx908;gfx90a;gfx942;gfx950;gfx1010;gfx1011;gfx1012;gfx1030;gfx1031;gfx1032;gfx1034;gfx1035;gfx1036;gfx1100;gfx1101;gfx1102;gfx1103;gfx1150;gfx1151;gfx1152;gfx1153;gfx1200;gfx1201"
ARG PYTORCH_VERSION=v2.13.0
ARG BUILD_PARALLEL_LEVEL=auto
ARG USE_PREBUILT_TORCHVISION=1
ARG ROCM_RELEASE=

RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake ninja-build build-essential pkg-config ccache \
        libjpeg-dev libpng-dev libfreetype6-dev libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

RUN uv venv /build-venv --python 3.12 --seed \
    && uv pip install --python /build-venv/bin/python3 \
        numpy pyyaml typing_extensions requests setuptools wheel
ENV PATH=/build-venv/bin:$PATH

# torchvision: try prebuilt per-arch wheel first, fallback to source build from pytorch/vision.
# Requires pytorch installed for setup.py (source path) -- install the pytorch wheel first.
# See the pytorch-builder stage above for the PIP:/SOURCE: decision format. The dist dir may
# also contain a non-.whl sdist (e.g. the `rocm` metapackage, when USE_PREBUILT_PYTORCH=1
# pulled in TheRock's rocm-sdk-* deps) -- copy the whole dir, not just *.whl, and pass
# --find-links so pip/uv can resolve that dependency locally instead of hitting PyPI for it.
# --from=pytorch-export (not pytorch-builder directly) -- see that alias's own comment above.
COPY --from=pytorch-export /pytorch/dist/* /tmp/torch/
COPY scripts/torch-package-build-decide.sh scripts/generate-torch-constraints.sh /tmp/

# --no-deps + the explicit package list: pytorch-builder's devreleases-snapshot fallback (see
# rocm-devrelease-snapshot.py) downloads a self-consistent set of wheels whose OWN declared
# Requires-Dist pins (e.g. "rocm-sdk-device-<arch>==7.14.0") don't match what was actually
# downloaded (AMD's devreleases metadata always declares that abstract pin, regardless of
# which nightly snapshot the file actually is) -- letting pip re-resolve dependencies here
# would re-check that pin and fail again. The named packages are torch's own ordinary,
# non-ROCm-pinned runtime deps (harmless to (re-)request explicitly on every path: satisfied
# instantly from the local wheels /tmp/torch already has when the PIP or SNAPSHOT tier
# fetched them, or pulled fresh from PyPI when SOURCE build didn't).
# The rocm-wheel-fetch/snapshot tiers can also produce a non-.whl sdist (the `rocm`
# metapackage itself, when devreleases had no clean release for it) -- torch's own
# _rocm_init.py imports the `rocm_sdk` module that sdist provides, so it must be installed
# too, not just the *.whl glob. --no-build-isolation: its build backend is already present
# (setuptools, installed just above) -- no need to let pip fetch one from the network.
RUN LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH uv pip install --python /build-venv/bin/python3 --no-deps \
        --find-links /tmp/torch \
        filelock typing_extensions sympy networkx jinja2 fsspec mpmath MarkupSafe setuptools \
        /tmp/torch/*.whl && \
    if ls /tmp/torch/*.tar.gz >/dev/null 2>&1; then \
        LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH uv pip install --python /build-venv/bin/python3 --no-deps \
            --no-build-isolation --find-links /tmp/torch /tmp/torch/*.tar.gz; \
    fi && \
    chmod +x /tmp/torch-package-build-decide.sh /tmp/generate-torch-constraints.sh && \
    mkdir -p /torchvision/dist && \
    PYTORCH_RESOLVED=$(basename $(ls /tmp/torch/torch-*.whl | head -1)) && \
    /tmp/generate-torch-constraints.sh /tmp/torch > /tmp/torch-constraints.txt && \
    DECISION=$(/tmp/torch-package-build-decide.sh torchvision "${PYTORCH_VERSION}" "${ROCM_ARCH}" "${USE_PREBUILT_TORCHVISION}" "${ROCM_RELEASE}" "${PYTORCH_RESOLVED}") && \
    if echo "$DECISION" | grep -q '^PIP:'; then \
        REST="${DECISION#PIP:}" && \
        PKG_SPEC="${REST%%:*}" && \
        INDEX_URL="${REST#*:}" && \
        echo "Downloading prebuilt torchvision: $PKG_SPEC from $INDEX_URL" && \
        /build-venv/bin/pip download "$PKG_SPEC" --constraint /tmp/torch-constraints.txt \
            --find-links /tmp/torch --index-url "$INDEX_URL" -d /torchvision/dist/; \
    else \
        REST="${DECISION#SOURCE:}" && \
        REPO="${REST%%:*}" && \
        BRANCH="${REST#*:}" && \
        echo "Building torchvision from source (repo: $REPO, branch: $BRANCH)" && \
        git clone --recursive --branch "${BRANCH}" --depth 1 --shallow-submodules \
            "https://github.com/${REPO}.git" /torchvision-src && \
        cd /torchvision-src && \
        LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH python3 setup.py bdist_wheel && \
        cp dist/*.whl /torchvision/dist/; \
    fi

# Build-time ABI check: the PIP path resolves torch and torchvision independently (separate
# `pip download` calls, not one joint solve), so a version mismatch between them (torchvision's
# wheel pins an exact matching torch build; if pytorch-builder resolved a different one) would
# otherwise only surface as a runtime ImportError in whatever downstream app imports torchvision
# first. Fail the build here instead, where it's obvious which stage/arch is at fault.
RUN LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH uv pip install --python /build-venv/bin/python3 \
        --find-links /tmp/torch --find-links /torchvision/dist /torchvision/dist/*.whl && \
    LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH /build-venv/bin/python3 -c \
        "import torch; import torchvision; print('torchvision', torchvision.__version__, 'OK against torch', torch.__version__)"

FROM ${ROCBLAS_IMAGE} AS torchaudio-builder

ARG PYTORCH_VERSION=v2.13.0
ARG BUILD_PARALLEL_LEVEL=auto
ARG USE_PREBUILT_TORCHAUDIO=1
ARG ROCM_RELEASE=

RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake ninja-build build-essential pkg-config ccache \
        libsndfile1-dev libsndfile1 sox libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

RUN uv venv /build-venv --python 3.12 --seed \
    && uv pip install --python /build-venv/bin/python3 \
        numpy pyyaml typing_extensions requests setuptools wheel
ENV PATH=/build-venv/bin:$PATH

# torchaudio: non-device-specific (same wheel works for every GPU target), builds once (not
# per-gfx) -- try prebuilt first same as pytorch/torchvision, source build (ROCm/audio) only
# when no matching wheel exists.
# Requires pytorch installed first (torchaudio setup.py imports torch). See torchvision-builder
# above for why the dist dir is copied whole and --find-links is passed (non-.whl sdist dep),
# and for --no-deps + the explicit package list (pytorch-builder's devreleases-snapshot
# fallback's declared pins don't match what was actually downloaded).
# --from=pytorch-export (not pytorch-builder directly) -- see that alias's own comment above.
COPY --from=pytorch-export /pytorch/dist/* /tmp/torch/
COPY scripts/torch-package-build-decide.sh scripts/generate-torch-constraints.sh /tmp/

RUN LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH uv pip install --python /build-venv/bin/python3 --no-deps \
        --find-links /tmp/torch \
        filelock typing_extensions sympy networkx jinja2 fsspec mpmath MarkupSafe setuptools \
        /tmp/torch/*.whl && \
    if ls /tmp/torch/*.tar.gz >/dev/null 2>&1; then \
        LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH uv pip install --python /build-venv/bin/python3 --no-deps \
            --no-build-isolation --find-links /tmp/torch /tmp/torch/*.tar.gz; \
    fi && \
    chmod +x /tmp/torch-package-build-decide.sh /tmp/generate-torch-constraints.sh && \
    mkdir -p /torchaudio/dist && \
    PYTORCH_RESOLVED=$(basename $(ls /tmp/torch/torch-*.whl | head -1)) && \
    /tmp/generate-torch-constraints.sh /tmp/torch > /tmp/torch-constraints.txt && \
    DECISION=$(/tmp/torch-package-build-decide.sh torchaudio "${PYTORCH_VERSION}" "" "${USE_PREBUILT_TORCHAUDIO}" "${ROCM_RELEASE}" "${PYTORCH_RESOLVED}") && \
    if echo "$DECISION" | grep -q '^PIP:'; then \
        REST="${DECISION#PIP:}" && \
        PKG_SPEC="${REST%%:*}" && \
        INDEX_URL="${REST#*:}" && \
        echo "Downloading prebuilt torchaudio: $PKG_SPEC from $INDEX_URL" && \
        /build-venv/bin/pip download "$PKG_SPEC" --constraint /tmp/torch-constraints.txt \
            --find-links /tmp/torch --index-url "$INDEX_URL" -d /torchaudio/dist/; \
    else \
        REST="${DECISION#SOURCE:}" && \
        REPO="${REST%%:*}" && \
        BRANCH="${REST#*:}" && \
        echo "Building torchaudio from source (repo: $REPO, branch: $BRANCH)" && \
        git clone --recursive --branch "${BRANCH}" --depth 1 --shallow-submodules \
            "https://github.com/${REPO}.git" /torchaudio-src && \
        cd /torchaudio-src && \
        LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH python3 setup.py bdist_wheel && \
        cp dist/*.whl /torchaudio/dist/; \
    fi

# Build-time ABI check, same reasoning as torchvision-builder's: torch and torchaudio are
# resolved independently, verify they actually load together before shipping.
RUN LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH uv pip install --python /build-venv/bin/python3 \
        --find-links /tmp/torch --find-links /torchaudio/dist /torchaudio/dist/*.whl && \
    LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH /build-venv/bin/python3 -c \
        "import torch; import torchaudio; print('torchaudio', torchaudio.__version__, 'OK against torch', torch.__version__)"

# Indirection stages: `COPY --from=$VAR` isn't allowed (BuildKit rejects a
# variable in --from), so resolve each component-image ARG through a FROM with
# a static alias, then COPY from the alias. With the defaults these aliases are
# just the local builder stages (one-shot `docker build .` compiles everything
# and BuildKit dedups the shared migraphx-builder); in CI the ARGs are passed
# as prebuilt image refs, so these become plain image pulls and the local
# builder stages drop out of the target's graph entirely -- no recompilation.
FROM ${MIGRAPHX_IMAGE} AS migraphx-export
# pytorch-export declared earlier (right before torchvision-builder) so that
# stage can also alias through it -- see its own comment there.
FROM ${TORCHVISION_IMAGE} AS torchvision-export
FROM ${TORCHAUDIO_IMAGE} AS torchaudio-export

FROM python-base AS ort-builder

ARG ROCM_ARCH="gfx900;gfx90c;gfx906;gfx908;gfx90a;gfx942;gfx950;gfx1010;gfx1011;gfx1012;gfx1030;gfx1031;gfx1032;gfx1034;gfx1035;gfx1036;gfx1100;gfx1101;gfx1102;gfx1103;gfx1150;gfx1151;gfx1152;gfx1153;gfx1200;gfx1201"
ARG ORT_VERSION=v1.28.0

COPY --from=migraphx-export /opt/rocm /opt/rocm

RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake ninja-build build-essential pkg-config ccache \
    && rm -rf /var/lib/apt/lists/*

# Same reasoning as pytorch-builder above: build against 3.12, not this base
# image's native 3.14, so the wheel's ABI tag matches the final stage's venv.
RUN uv venv /build-venv --python 3.12 \
    && uv pip install --python /build-venv/bin/python3 \
        numpy packaging wheel setuptools cmake
ENV PATH=/build-venv/bin:$PATH

RUN git clone --recursive --branch ${ORT_VERSION} --depth 1 \
        https://github.com/microsoft/onnxruntime.git /onnxruntime

WORKDIR /onnxruntime

# Standalone --use_rocm/--rocm_home are gone as of this ORT version (ROCm EP
# folded away, see https://github.com/microsoft/onnxruntime/issues/26801) --
# --rocm_home only survives as a deprecated no-op under the MIGraphX group.
# Only --use_migraphx/--migraphx_home is needed; that's all this project ever
# wanted anyway.
#
# Same ccache-mount reasoning as the earlier builder stages -- and same
# reason it doesn't also cache-mount `build` itself (stale CMakeCache.txt
# across attempts with different flags/mounts). Wheel is copied to
# /onnxruntime/dist (a real layer path, not cache-mounted) below.
RUN --mount=type=cache,target=/root/.ccache,id=ort-ccache \
    python3 tools/ci_build/build.py \
        --config Release \
        --build_dir /onnxruntime/build \
        --parallel \
        --build_wheel \
        --skip_tests \
        --allow_running_as_root \
        --compile_no_warning_as_error \
        --use_migraphx --migraphx_home /opt/rocm \
        --cmake_extra_defines "CMAKE_HIP_ARCHITECTURES=${ROCM_ARCH}" \
        --cmake_extra_defines "CMAKE_C_COMPILER_LAUNCHER=ccache" \
        --cmake_extra_defines "CMAKE_CXX_COMPILER_LAUNCHER=ccache" \
    && mkdir -p /onnxruntime/dist && cp /onnxruntime/build/Release/dist/*.whl /onnxruntime/dist/

# ort-export alias: same indirection as migraphx-export/pytorch-export above,
# declared here (after ort-builder) so its default resolves to that stage.
FROM ${ORT_IMAGE} AS ort-export

FROM python-base

ARG TORCH_BLAS_PREFER_HIPBLASLT
# See the ARG's own comment near the top of this file. 0 only on
# gfx900/gfx906/gfx90c; a no-op 1 everywhere else.
ENV TORCH_BLAS_PREFER_HIPBLASLT=${TORCH_BLAS_PREFER_HIPBLASLT}

COPY --from=migraphx-export /opt/rocm /opt/rocm

# Unlike rocm/onnxruntime (which ships an /etc/ld.so.conf.d entry for its
# ROCm lib dir), this base image (rocm/dev-ubuntu-*) doesn't register
# /opt/rocm/lib with the dynamic linker at all -- libs like libhiprand.so.1
# exist on disk but aren't found at runtime (e.g. by ctranslate2) without
# this, failing with "cannot open shared object file" despite the file being
# right there.
RUN echo "/opt/rocm/lib" > /etc/ld.so.conf.d/rocm.conf && ldconfig

# libprotobuf is a runtime dependency of libonnxruntime_providers_migraphx.so
# (MIGraphX links against it), libopenblas of torch's linear-algebra ops,
# and torchvision/torchaudio link against various image/audio codecs.
# All were apt-installed in builder stages but not carried into this stage's rootfs.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libprotobuf-dev libopenblas0 \
        libjpeg-turbo-progs libpng-dev libfreetype-dev libtiff-dev libwebp-dev \
        libsndfile1 libflac-dev libvorbis-dev libopus-dev \
    && rm -rf /var/lib/apt/lists/*

# This image's own python3 is 3.14 (native to this base), but the wheels
# built above target 3.12 (see migraphx-builder/pytorch-builder/ort-builder),
# for AudioMuse-AI's numpy/onnx pin compatibility -- so /opt/venv also has to
# be 3.12, not this stage's native interpreter, via the same uv-managed
# standalone Python as the builder stages (uv itself + the 3.12 interpreter
# both inherited from python-base, no re-download needed here).
ENV PYTHONPATH=/opt/rocm/lib
ENV VIRTUAL_ENV=/opt/venv
# --seed: uv venvs ship no pip binary by default, but this image is a drop-in
# for rocm/onnxruntime:* bases whose venvs do -- downstream Dockerfiles may
# call "$VIRTUAL_ENV/bin/pip" directly.
RUN uv venv $VIRTUAL_ENV --python 3.12 --seed

# onnxruntime + migraphx only -- the "classic" /opt/rocm stack's own Python
# surface. Kept in its own install step (not merged with torch's wheels
# below) because torch's prebuilt wheel bundles TheRock's pip-packaged ROCm
# SDK (its own libamd_comgr.so/libLLVM.so under
# site-packages/_rocm_sdk_core/lib), a second copy of comgr/LLVM independent
# of this stage's classic /opt/rocm one. Any process that ends up loading
# both (e.g. ctranslate2 -- built against classic /opt/rocm -- opportunistically
# does `import torch` in its specs/model_spec module if torch happens to be
# importable) aborts at startup: LLVM's global CommandLine option registry
# rejects the second registration of the same option
# ("CommandLine Error: Option 'spirv-expand-step' registered more than once!").
# Keeping torch out of $VIRTUAL_ENV entirely means classic-stack consumers
# never see it importable, so that opportunistic import just ImportErrors
# harmlessly instead of loading a second comgr/LLVM.
COPY --from=ort-export /onnxruntime/dist/*.whl /tmp/wheels-ort/
RUN uv pip install --python "$VIRTUAL_ENV/bin/python3" --no-cache --no-deps --find-links /tmp/wheels-ort \
        numpy flatbuffers packaging protobuf \
        /tmp/wheels-ort/*.whl \
    && rm -rf /tmp/wheels-ort \
    && "$VIRTUAL_ENV/bin/python3" -c "import onnxruntime as ort; p=ort.get_available_providers(); print('ORT providers:', p); assert 'MIGraphXExecutionProvider' in p" \
    && "$VIRTUAL_ENV/bin/python3" -c "import migraphx; print('migraphx python module OK')"

# torch/torchvision/torchaudio: deliberately a SEPARATE venv, not merged into
# $VIRTUAL_ENV above -- see the comment on the onnxruntime install. Not added
# to PATH either: torch is opt-in via this venv's own interpreter path, never
# the default `python3` a downstream Dockerfile gets from this image.
ENV VIRTUAL_ENV_TORCH=/opt/venv-torch
RUN uv venv $VIRTUAL_ENV_TORCH --python 3.12 --seed

COPY --from=pytorch-export /pytorch/dist/* /tmp/wheels-torch/
COPY --from=torchvision-export /torchvision/dist/* /tmp/wheels-torch/
COPY --from=torchaudio-export /torchaudio/dist/* /tmp/wheels-torch/
# All three dist dirs land in ONE shared directory (not three separate ones): pytorch,
# torchvision, and torchaudio's independent `pip download` calls each pull their own copy of
# shared transitive deps (filelock, jinja2, sympy, etc.) from the same index -- byte-identical
# content, but if left in separate directories, uv's resolver sees the same package name at two
# different local file:// paths and errors with "conflicting URLs" instead of just picking one.
#
# --no-deps + the explicit package list, same reasoning as torchvision-builder/torchaudio-
# builder's own installs above: pytorch-builder's devreleases-snapshot fallback can produce
# wheels whose declared Requires-Dist pins don't match what was actually downloaded, so pip's
# normal resolver can't be trusted to re-verify them here. The named packages cover every
# wheel's own ordinary, non-ROCm-pinned runtime deps (torch's numpy + torchvision's pillow) --
# satisfied instantly from /tmp/wheels-torch when already present, pulled fresh from PyPI
# otherwise.
RUN uv pip install --python "$VIRTUAL_ENV_TORCH/bin/python3" --no-cache --no-deps --find-links /tmp/wheels-torch \
        numpy filelock typing_extensions sympy networkx jinja2 fsspec mpmath MarkupSafe \
        setuptools pillow \
        /tmp/wheels-torch/*.whl \
    && if ls /tmp/wheels-torch/*.tar.gz >/dev/null 2>&1; then \
        uv pip install --python "$VIRTUAL_ENV_TORCH/bin/python3" --no-cache --no-deps \
            --no-build-isolation --find-links /tmp/wheels-torch /tmp/wheels-torch/*.tar.gz; \
    fi \
    && rm -rf /tmp/wheels-torch \
    && "$VIRTUAL_ENV_TORCH/bin/python3" -c "import torch; print('torch', torch.__version__, 'HIP built:', torch.version.hip)" \
    && "$VIRTUAL_ENV_TORCH/bin/python3" -c "import torch, torchvision; print('torchvision', torchvision.__version__)" \
    && "$VIRTUAL_ENV_TORCH/bin/python3" -c "import torch, torchaudio; print('torchaudio', torchaudio.__version__)"

ENV PATH="$VIRTUAL_ENV/bin:${PATH}"
