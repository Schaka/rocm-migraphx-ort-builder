# syntax=docker/dockerfile:1
#
# ONNX Runtime built from source with the MIGraphX execution provider, against
# the MIGraphX-carrying /opt/rocm the migraphx target produced. Publishes the
# wheel in /onnxruntime/dist.
FROM python-base

ARG ROCM_ARCH
ARG ORT_VERSION

COPY --from=migraphx /opt/rocm /opt/rocm

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
