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
ARG BASE_IMAGE=rocm/dev-ubuntu-26.04:7.14.0-full

# Component-image references. Downstream stages COPY each component's artifacts
# from these, so in CI every component can be built as its own job/image (see
# .github/workflows) and a later stage just pulls the prebuilt result instead
# of recompiling it. Defaults point at the local stage names, so a plain
# `docker build .` on one machine still resolves them to the in-tree stages and
# builds everything in a single shot -- BuildKit treats `COPY --from=<name>` as
# a stage reference when <name> matches a stage, else as an image to pull.
ARG MIGRAPHX_IMAGE=migraphx-builder
ARG PYTORCH_IMAGE=pytorch-builder
ARG ORT_IMAGE=ort-builder

# Cache-bust token for the develop-tracking MIGraphX clone below. A moving
# branch's `git clone` layer has a cache key that never changes, so it would
# cache-hit forever and keep rebuilding the same stale commit every night --
# CI passes the build date here so each nightly run re-clones develop. Any
# changing value works; left constant locally (pinned deps like PyTorch/ORT
# don't need it -- their tags already change the cache key when bumped).
ARG SOURCE_DATE=unknown

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

FROM python-base AS migraphx-builder

# Semicolon-separated GPU_TARGETS list, matching the breadth AMD's own
# published images build for (CDNA1-3, RDNA2-4), not just this host's GPU.
# Narrow it via --build-arg if you only need one target and want a faster
# build.
ARG ROCM_ARCH="gfx900;gfx906;gfx908;gfx90a;gfx942;gfx1030;gfx1100;gfx1101;gfx1102;gfx1150;gfx1151;gfx1200;gfx1201"

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
RUN echo "MIGraphX develop snapshot: ${SOURCE_DATE}" \
    && git clone --branch develop --depth 1 \
        https://github.com/ROCm/AMDMIGraphX.git /migraphx-src

WORKDIR /migraphx-src
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
        -T install

FROM python-base AS pytorch-builder

# No prebuilt wheel exists for this ROCm release yet (AMD's nightly index at
# download.pytorch.org/whl/nightly/ only goes up to rocm7.2, this base is
# 7.14) -- build from source. Doesn't need MIGraphX, only the HIP/rocBLAS/
# MIOpen/rocRAND stack already in this base image.
ARG ROCM_ARCH="gfx900;gfx906;gfx908;gfx90a;gfx942;gfx1030;gfx1100;gfx1101;gfx1102;gfx1150;gfx1151;gfx1200;gfx1201"
ARG PYTORCH_VERSION=v2.12.0
ARG BUILD_PARALLEL_LEVEL=auto

RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake ninja-build build-essential \
        libopenblas-dev pkg-config libdrm-dev ccache \
    && rm -rf /var/lib/apt/lists/*

# Same reasoning as migraphx-builder's /rbuild-venv: this base image's native
# python3 is 3.14, but the wheel this stage produces has to load into the
# final stage's venv, which is pinned to 3.12 for AudioMuse-AI's numpy/onnx
# pin compatibility -- so build against that same 3.12, not the system one.
RUN uv venv /build-venv --python 3.12 \
    && uv pip install --python /build-venv/bin/python3 \
        numpy pyyaml typing_extensions requests setuptools wheel six
ENV PATH=/build-venv/bin:$PATH

RUN git clone --recursive --branch ${PYTORCH_VERSION} --depth 1 --shallow-submodules \
        https://github.com/pytorch/pytorch.git /pytorch

WORKDIR /pytorch
# setup.py does NOT hipify CUDA sources itself -- tools/amd_build/build_amd.py
# (translates c10/cuda -> c10/hip, aten/src/ATen/cuda -> aten/src/ATen/hip,
# etc.) has to run as an explicit step first, or CMake fails looking for
# .hip sources/templates that don't exist yet. PYTORCH_ROCM_ARCH takes the
# same semicolon-separated list as GPU_TARGETS above.
RUN python3 tools/amd_build/build_amd.py

# USE_FLASH_ATTENTION/USE_MEM_EFF_ATTENTION default on for ROCm and pull in
# aotriton, which configures its own isolated venv that doesn't inherit this
# stage's numpy (--break-system-packages install), failing the subbuild.
# AudioMuse-AI doesn't need flash attention, so disable it instead of feeding
# that nested venv its own numpy.
#
# USE_DISTRIBUTED defaults on and pulls in USE_NCCL/USE_NVSHMEM, whose symm_mem
# device-communicator code (nccl_extension.cu, ncclDevComm/NCCLDevCommManager/
# ncclCoopCta) is NVIDIA-NCCL-only -- RCCL doesn't implement it, so it fails to
# compile under ROCm at this PyTorch tag. AudioMuse-AI is single-GPU/single-
# node, doesn't need torch.distributed, so disable the whole subsystem instead
# of patching upstream's NCCL-only code.
#
# MAX_JOBS sizing: same MemAvailable/4GB-per-job "auto" logic as the
# migraphx-builder stage above -- PyTorch's own C++ TUs are cheaper than
# LLVM's, but running as many of them as nproc on a RAM-constrained host
# still risks OOM.
# Same ccache-mount reasoning as migraphx-builder above; same reason it can't
# also cache-mount `build` itself -- a cached CMakeCache.txt from a prior
# attempt with different USE_* flags left a stale/incomplete configure state
# (ATen/Config.h missing) rather than reconfiguring cleanly.
RUN --mount=type=cache,target=/root/.ccache,id=pytorch-ccache \
    ulimit -s unlimited && \
    jobs="${BUILD_PARALLEL_LEVEL}"; \
    if [ "$jobs" = "auto" ]; then \
        jobs=$(awk '/MemAvailable/{printf "%d", $2/1024/1024/4}' /proc/meminfo); \
        cpu=$(nproc); [ "$jobs" -gt "$cpu" ] && jobs=$cpu; \
        [ "$jobs" -lt 1 ] && jobs=1; \
    fi; \
    echo "PyTorch build: using $jobs parallel jobs"; \
    env USE_ROCM=1 ROCM_HOME=/opt/rocm "PYTORCH_ROCM_ARCH=${ROCM_ARCH}" \
        MAX_JOBS=$jobs USE_MKLDNN=0 USE_CCACHE=1 \
        USE_FLASH_ATTENTION=0 USE_MEM_EFF_ATTENTION=0 \
        USE_DISTRIBUTED=0 \
        python3 setup.py bdist_wheel

# Indirection stages: `COPY --from=$VAR` isn't allowed (BuildKit rejects a
# variable in --from), so resolve each component-image ARG through a FROM with
# a static alias, then COPY from the alias. With the defaults these aliases are
# just the local builder stages (one-shot `docker build .` compiles everything
# and BuildKit dedups the shared migraphx-builder); in CI the ARGs are passed
# as prebuilt image refs, so these become plain image pulls and the local
# builder stages drop out of the target's graph entirely -- no recompilation.
FROM ${MIGRAPHX_IMAGE} AS migraphx-export
FROM ${PYTORCH_IMAGE} AS pytorch-export

FROM python-base AS ort-builder

ARG ROCM_ARCH="gfx900;gfx906;gfx908;gfx90a;gfx942;gfx1030;gfx1100;gfx1101;gfx1102;gfx1150;gfx1151;gfx1200;gfx1201"
ARG ORT_VERSION=v1.23.2

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

# v1.23.2's semver.h relies on <cstdint> being pulled in transitively, which
# newer GCC (as shipped in this base image) no longer does.
RUN sed -i '/#include <optional>/a #include <cstdint>' \
        /onnxruntime/onnxruntime/core/common/semver.h

# build.py's build_python_wheel() passes --use_rocm to setup.py bdist_wheel
# when --use_rocm is set, but v1.23.2's setup.py has no handler for that flag
# (only --use_migraphx), so bdist_wheel rejects it as an unrecognized option.
RUN sed -i \
        's/elif parse_arg_remove_boolean(sys.argv, "--use_migraphx"):/elif parse_arg_remove_boolean(sys.argv, "--use_rocm"):\n    parse_arg_remove_boolean(sys.argv, "--use_migraphx")\n    is_migraphx = True\n    package_name = "onnxruntime-rocm"\nelif parse_arg_remove_boolean(sys.argv, "--use_migraphx"):/' \
        /onnxruntime/setup.py

WORKDIR /onnxruntime

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
        --use_rocm --rocm_home /opt/rocm \
        --use_migraphx --migraphx_home /opt/rocm \
        --cmake_extra_defines "CMAKE_HIP_ARCHITECTURES=${ROCM_ARCH}" \
        --cmake_extra_defines "CMAKE_C_COMPILER_LAUNCHER=ccache" \
        --cmake_extra_defines "CMAKE_CXX_COMPILER_LAUNCHER=ccache" \
    && mkdir -p /onnxruntime/dist && cp /onnxruntime/build/Release/dist/*.whl /onnxruntime/dist/

# ort-export alias: same indirection as migraphx-export/pytorch-export above,
# declared here (after ort-builder) so its default resolves to that stage.
FROM ${ORT_IMAGE} AS ort-export

FROM python-base

COPY --from=migraphx-export /opt/rocm /opt/rocm

# Unlike rocm/onnxruntime (which ships an /etc/ld.so.conf.d entry for its
# ROCm lib dir), this base image (rocm/dev-ubuntu-*) doesn't register
# /opt/rocm/lib with the dynamic linker at all -- libs like libhiprand.so.1
# exist on disk but aren't found at runtime (e.g. by ctranslate2) without
# this, failing with "cannot open shared object file" despite the file being
# right there.
RUN echo "/opt/rocm/lib" > /etc/ld.so.conf.d/rocm.conf && ldconfig

# libprotobuf is a runtime dependency of libonnxruntime_providers_migraphx.so
# (MIGraphX links against it), and libopenblas of torch's linear-algebra ops
# -- both were apt-installed in migraphx-builder/pytorch-builder but not
# carried into this stage's rootfs.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libprotobuf-dev libopenblas0 \
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

COPY --from=ort-export /onnxruntime/dist/*.whl /tmp/ort/
COPY --from=pytorch-export /pytorch/dist/*.whl /tmp/torch/
RUN uv pip install --python "$VIRTUAL_ENV/bin/python3" --no-cache numpy /tmp/ort/*.whl /tmp/torch/*.whl \
    && rm -rf /tmp/ort /tmp/torch \
    && "$VIRTUAL_ENV/bin/python3" -c "import onnxruntime as ort; p=ort.get_available_providers(); print('ORT providers:', p); assert 'MIGraphXExecutionProvider' in p" \
    && "$VIRTUAL_ENV/bin/python3" -c "import torch; print('torch', torch.__version__, 'HIP built:', torch.version.hip)" \
    && "$VIRTUAL_ENV/bin/python3" -c "import migraphx; print('migraphx python module OK')"

ENV PATH="$VIRTUAL_ENV/bin:${PATH}"
