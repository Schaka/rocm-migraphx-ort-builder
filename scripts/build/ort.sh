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

# Tagging the wheel's version happens here, post-build, not by editing
# VERSION_NUMBER before the C++ build: onnxruntime_c_api.cc has a compile-time
# static_assert comparing ORT_VERSION against a hardcoded literal, which a PEP
# 440 local segment in VERSION_NUMBER trips. wheel unpack/pack is the only
# safe way to add one -- PyPI's own onnxruntime can otherwise report the exact
# same version this build does, which would make an exact-version pin in the
# final image's constraints file unenforceable (PyPI could satisfy it too); a
# suffix no PyPI release will ever carry closes that gap. Tagged by arch, not
# just a fixed marker, because this repo publishes one wheel per ROCM_ARCH --
# a plain "+migraphx" would make gfx900's and gfx1100's wheels
# version-identical despite being different, arch-specific builds. ROCM_ARCH
# is always plain alphanumeric (gfx900, gfx1100, ...), a valid PEP 440 local
# segment as-is, no sanitizing needed.
built_whl="$(ls /onnxruntime/build/Release/dist/*.whl)"
unpack_dir="/onnxruntime/build/Release/dist/unpacked"
python3 -m wheel unpack "$built_whl" -d "$unpack_dir"
old_dir="$(find "$unpack_dir" -mindepth 1 -maxdepth 1 -type d)"
old_name="$(basename "$old_dir")"
new_name="${old_name}+${ROCM_ARCH}"
mv "$old_dir" "$unpack_dir/$new_name"
old_dist_info="$(find "$unpack_dir/$new_name" -maxdepth 1 -name '*.dist-info')"
new_dist_info="$unpack_dir/$new_name/${new_name}.dist-info"
mv "$old_dist_info" "$new_dist_info"
sed -i "s/^Version: .*/Version: ${old_name#*-}+${ROCM_ARCH}/" "$new_dist_info/METADATA"

mkdir -p /onnxruntime/dist
python3 -m wheel pack "$unpack_dir/$new_name" -d /onnxruntime/dist
