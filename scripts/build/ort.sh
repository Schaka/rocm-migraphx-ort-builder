#!/bin/sh
# Builds the ONNX Runtime python wheel with the MIGraphX execution provider.
# Run from /onnxruntime.
#
# Standalone --use_rocm/--rocm_home are gone as of this ORT version (ROCm EP
# folded away, see https://github.com/microsoft/onnxruntime/issues/26801) --
# --rocm_home only survives as a deprecated no-op under the MIGraphX group. Only
# --use_migraphx/--migraphx_home is needed; that's all this project ever wanted
# anyway.
# Inputs (build-args): ROCM_ARCH.
set -eu

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
    --cmake_extra_defines "CMAKE_CXX_COMPILER_LAUNCHER=ccache"

# /onnxruntime/dist is a real layer path, not a cache mount -- this is what the
# final stage bind-mounts to install from.
mkdir -p /onnxruntime/dist
cp /onnxruntime/build/Release/dist/*.whl /onnxruntime/dist/
