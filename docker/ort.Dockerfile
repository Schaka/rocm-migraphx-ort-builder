# syntax=docker/dockerfile:1
#
# ONNX Runtime built from source with the MIGraphX execution provider, against
# the MIGraphX-carrying /opt/rocm the migraphx target produced. Publishes the
# wheel in /onnxruntime/dist.
FROM python-base

ARG ROCM_ARCH
ARG ORT_VERSION

COPY --from=migraphx /opt/rocm /opt/rocm

# ROCm 10.0 ships its own flatbuffers in /opt/rocm, and ORT's MIGraphX
# provider sets CMAKE_PREFIX_PATH=/opt/rocm -- so ORT's flatbuffers
# FetchContent declaration (a minimum-version FIND_PACKAGE_ARGS) find_package()s
# the ROCm copy instead of downloading the version it pins, and the mismatched
# headers fail ORT's own generated-schema static_assert
# (FLATBUFFERS_VERSION_MAJOR mismatch). Removing the ROCm flatbuffers footprint
# makes find_package fail so FetchContent downloads and builds the pinned
# version instead. Nothing in ORT or the MIGraphX provider uses the ROCm copy.
RUN rm -rf /opt/rocm/include/flatbuffers \
        /opt/rocm/lib/cmake/flatbuffers \
        /opt/rocm/lib/libflatbuffers.a \
        /opt/rocm/lib/pkgconfig/flatbuffers.pc

RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake ninja-build build-essential pkg-config ccache \
    && rm -rf /var/lib/apt/lists/*

# Build against 3.12, not the base image's native python3, so the wheel's ABI tag
# matches the final image's venv.
RUN uv venv /build-venv --python 3.12 \
    && uv pip install --python /build-venv/bin/python3 \
        numpy packaging wheel setuptools cmake
ENV PATH=/build-venv/bin:$PATH

RUN git clone --recursive --branch ${ORT_VERSION} --depth 1 \
        https://github.com/microsoft/onnxruntime.git /onnxruntime

WORKDIR /onnxruntime

# Same ccache-mount reasoning as the other builder targets -- and the same reason
# it doesn't also cache-mount `build` itself (a stale CMakeCache.txt across
# attempts with different flags/mounts).
RUN --mount=type=cache,target=/root/.ccache,id=ort-ccache \
    --mount=type=bind,source=scripts/build/ort.sh,target=/scripts/build/ort.sh \
    /scripts/build/ort.sh
