# shellcheck shell=sh
# Sourced, never executed. Shared by the torchvision and torchaudio stages,
# which differ only in package name, apt dependencies, and whether their wheel
# is device-specific -- everything else (tier order, ABI gate, source build) is
# identical, so it lives here once.
#
# Requires rocm-env.sh and torch-wheels.sh to have been sourced.
# Inputs (build-args) read from the environment: TORCH_COMPANION,
# PYTORCH_VERSION, ROCM_RELEASE, ROCM_ARCH (torchvision only), USE_PREBUILT.

# Where this companion's own wheel is collected. Kept per-package rather than a
# generic /dist so already-published component images stay readable by the final
# stage's COPY paths.
companion_dist() {
    echo "/${TORCH_COMPANION}/dist"
}

# torchaudio is non-device-specific: the same wheel works for every GPU target,
# so it is built once rather than per-gfx and its decision must not be keyed on
# an arch. torchvision's is.
companion_decide_arch() {
    case "${TORCH_COMPANION}" in
        torchaudio) echo "" ;;
        *)          echo "${ROCM_ARCH:-}" ;;
    esac
}

# companion_decide <use_prebuilt> <resolved_torch_wheel> -> "PIP:<spec>:<index>"
# or "SOURCE:<repo>:<branch>". Pass 0 for <use_prebuilt> to force SOURCE.
companion_decide() {
    /scripts/torch-package-build-decide.sh \
        "${TORCH_COMPANION}" "${PYTORCH_VERSION}" "$(companion_decide_arch)" \
        "$1" "${ROCM_RELEASE}" "$2"
}

# companion_build_from_source <decision> -- clones and builds the wheel into
# companion_dist. Leaves the source tree removed and cwd at /, because the ABI
# gate runs afterwards and a process whose cwd has been unlinked makes every
# later command warn on getcwd.
companion_build_from_source() {
    _rest="${1#SOURCE:}"
    _repo="${_rest%%:*}"
    _branch="${_rest#*:}"
    _src="/${TORCH_COMPANION}-src"

    echo "Building ${TORCH_COMPANION} from source (repo: ${_repo}, branch: ${_branch})"
    git clone --recursive --branch "${_branch}" --depth 1 --shallow-submodules \
        "https://github.com/${_repo}.git" "${_src}"
    cd "${_src}" || exit 1
    (
        use_rocm_ext_build_env
        python3 setup.py bdist_wheel
    )
    cp dist/*.whl "$(companion_dist)/"
    cd /
    rm -rf "${_src}"
}

# companion_download_prebuilt <decision> -- pulls the prebuilt wheel (plus
# whatever deps pip decides it needs) into companion_dist, constrained to the
# torch that is actually installed.
companion_download_prebuilt() {
    _rest="${1#PIP:}"
    _spec="${_rest%%:*}"
    _index="${_rest#*:}"
    echo "Downloading prebuilt ${TORCH_COMPANION}: ${_spec} from ${_index}"
    /build-venv/bin/pip download "${_spec}" \
        --constraint /tmp/torch-constraints.txt \
        --find-links /wheels-torch --index-url "${_index}" \
        -d "$(companion_dist)/"
}

# companion_verify -- installs the collected wheel and imports it against the
# resolved torch. Non-fatal by design: the caller decides what a failure means.
#
# --no-deps is load-bearing here, not an optimisation. Without it this gate does
# the opposite of its job: torchvision's metadata needs `pillow`, which is not in
# /wheels-torch, so uv falls through to PyPI -- and once it is resolving there it
# takes PyPI's `torch` as well, pulling ~2.6GB of nvidia-cu13 wheels, replacing
# the ROCm torch in this venv, and then "verifying" the wheel against a CUDA
# build. Observed before this flag was added:
#   "torchvision 0.28.0+8fb8771 OK against torch 2.13.0+cu130"
# i.e. the exact torch/torchvision mismatch this gate exists to catch is the one
# it would let through. torchaudio escaped only because its metadata pins a bare
# `torch` with no version, so uv left the installed ROCm one alone.
#
# Everything torch itself needs is already installed by install_torch_wheels;
# pillow is the one genuine extra, and naming it explicitly keeps resolution
# local instead of reopening the door to PyPI.
companion_verify() {
    uv pip install --python /build-venv/bin/python3 --no-deps \
        --find-links /wheels-torch --find-links "$(companion_dist)" \
        pillow \
        "$(companion_dist)"/*.whl \
    && /build-venv/bin/python3 -c \
        "import torch, ${TORCH_COMPANION} as pkg; print('${TORCH_COMPANION}', pkg.__version__, 'OK against torch', torch.__version__)" \
    && companion_verify_gpu_kernels
}

# An `import` alone is not evidence the wheel is usable: a torchvision built
# without FORCE_CUDA imports perfectly and only fails later, on real hardware,
# with "Could not run 'torchvision::nms' with arguments from the 'CUDA' backend".
# That shipped undetected until a GPU smoke test caught it.
#
# torch's dispatcher can be interrogated without a GPU, so the build container can
# check it: _dispatch_dump lists every backend an operator has a kernel
# registered for. A line beginning "CUDA:" is a real GPU kernel -- substring
# matching is not enough, since the dump always mentions AutogradCUDA and
# AutocastCUDA even for a CPU-only build.
#
# torchvision only: `torchvision::nms` is unconditionally present and GPU-backed
# in every supported version. torchaudio has no single operator that is
# guaranteed to be GPU-backed across versions, so asserting one there would risk
# failing a perfectly good wheel; it is left unchecked rather than guessed at.
companion_verify_gpu_kernels() {
    [ "${TORCH_COMPANION}" = "torchvision" ] || return 0

    /build-venv/bin/python3 -c "
import sys, torch, torchvision
try:
    dump = torch._C._dispatch_dump('torchvision::nms')
except Exception as exc:
    print('WARNING: cannot introspect dispatcher (%s) -- skipping GPU-kernel check' % exc)
    sys.exit(0)
backends = [l.split(':', 1)[0].strip() for l in dump.splitlines() if ':' in l]
if 'CUDA' not in backends:
    print('FATAL: torchvision has no CUDA/HIP kernel for nms -- built without GPU support.')
    print('registered backends:', sorted(set(backends)))
    sys.exit(1)
print('torchvision GPU kernels present (nms registered for CUDA/HIP)')
"
}
