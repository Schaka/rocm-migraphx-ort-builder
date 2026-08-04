# shellcheck shell=sh
# Sourced, never executed.

# Makes /opt/rocm's shared libraries loadable for the current script. Needed by
# every `uv pip install`/`python -c` that touches a torch wheel: none of the
# builder stages register /opt/rocm/lib with the dynamic linker (only the final
# image does, via /etc/ld.so.conf.d).
#
# ${LD_LIBRARY_PATH:-} rather than $LD_LIBRARY_PATH: it is unset in these
# images, and callers run under `set -u` in places.
use_rocm_runtime_path() {
    LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
    export LD_LIBRARY_PATH
}

# Build environment for torchvision/torchaudio's HIP extensions.
#
# ROCM_HOME/ROCM_PATH are the load-bearing part, not the -I flags. torchaudio and
# torchvision build their extensions through torch's cpp_extension, whose
# include_paths() appends _join_rocm_home('include') for a HIP extension -- and
# _find_rocm_home() prefers $ROCM_HOME, then `which hipcc`, and only then
# /opt/rocm. The PIP/snapshot-tier torch wheels install TheRock's rocm_sdk_core
# into the SAME venv, which puts a hipcc shim on PATH, so that second guess wins
# and ROCM_HOME resolves to the venv root -- yielding a bogus <venv>/include that
# has no hip/ at all, and a compile that dies on `hip/hip_runtime.h: No such file
# or directory`. Verified against the published gfx90c torch image: unset,
# include_paths('cuda') ends in <venv>/include and the build fails; pinned to
# /opt/rocm it ends in /opt/rocm/include and the wheel builds and imports.
#
# CPPFLAGS/CXXFLAGS/LDFLAGS are belt-and-braces for anything that bypasses
# include_paths().
#
# FORCE_CUDA is not optional here, despite the CUDA-sounding name (torchvision
# uses the same switch for HIP once its sources are hipified). torchvision's
# setup.py only compiles GPU kernels when `torch.cuda.is_available()` is true OR
# FORCE_CUDA=1 -- and inside a build container there is no GPU, so the check is
# always false. Without it the build hipifies every source, reports
#     FORCE_CUDA = False / BUILD_CUDA_SOURCES = False
# and then silently emits a CPU-only wheel: it imports fine, so nothing downstream
# notices until an actual GPU op is called and dies with
#     NotImplementedError: Could not run 'torchvision::nms' with arguments from
#     the 'CUDA' backend
# (observed on a real gfx1201 image before this was set). companion_verify's
# dispatch check is the guard that makes this impossible to reintroduce quietly.
#
# PYTORCH_ROCM_ARCH has to come with it: with no GPU to autodetect, the HIP
# extension build otherwise has no target list to compile for.
use_rocm_ext_build_env() {
    use_rocm_runtime_path
    ROCM_HOME=/opt/rocm
    ROCM_PATH=/opt/rocm
    PATH="$PATH:/opt/rocm/bin"
    CPPFLAGS="-I/opt/rocm/include ${CPPFLAGS:-}"
    CXXFLAGS="-I/opt/rocm/include ${CXXFLAGS:-}"
    LDFLAGS="-L/opt/rocm/lib ${LDFLAGS:-}"
    export ROCM_HOME ROCM_PATH PATH CPPFLAGS CXXFLAGS LDFLAGS

    if [ -n "${ROCM_ARCH:-}" ]; then
        FORCE_CUDA=1
        PYTORCH_ROCM_ARCH="${ROCM_ARCH}"
        export FORCE_CUDA PYTORCH_ROCM_ARCH
    else
        echo "WARNING: ROCM_ARCH unset -- building ${TORCH_COMPANION:-extension} without GPU kernels" >&2
    fi
}
