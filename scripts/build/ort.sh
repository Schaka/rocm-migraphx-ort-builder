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

# setup.py reads this file's contents verbatim as the wheel's version, no
# format validation -- appending a PEP 440 local segment here lands straight
# in the wheel's METADATA. PyPI's own onnxruntime can otherwise report the
# exact same version this build does, which would make an exact-version pin
# in the final image's constraints file unenforceable (PyPI could satisfy it
# too); a suffix no PyPI release will ever carry closes that gap. Tagged by
# arch, not just a fixed marker, because this repo publishes one wheel per
# ROCM_ARCH -- a plain "+migraphx" would make gfx900's and gfx1100's wheels
# version-identical despite being different, arch-specific builds. ROCM_ARCH
# is always plain alphanumeric (gfx900, gfx1100, ...), a valid PEP 440 local
# segment as-is, no sanitizing needed.
printf '%s+%s' "$(cat VERSION_NUMBER)" "${ROCM_ARCH}" > VERSION_NUMBER

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
