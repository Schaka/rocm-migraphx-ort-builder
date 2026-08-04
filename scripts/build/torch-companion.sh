#!/bin/sh
# Materialises the torchvision or torchaudio wheel into /<package>/dist: prebuilt
# per-arch wheel first, from-source build (pytorch/vision, ROCm/audio) when no
# matching wheel exists.
#
# torch itself has to be installed first -- both packages' setup.py imports it --
# and it arrives on a read-only bind mount at /wheels-torch rather than a COPY:
# torch's wheel alone is ~1.7GB and a COPY would commit it as a layer of this
# stage's published image, which the final stage then has to pull for nothing.
# Inputs (build-args): TORCH_COMPANION, PYTORCH_VERSION, ROCM_RELEASE,
# ROCM_ARCH, USE_PREBUILT.
set -eu

. /scripts/lib/rocm-env.sh
. /scripts/lib/torch-wheels.sh
. /scripts/lib/torch-companion.sh

use_rocm_runtime_path

install_torch_wheels /build-venv/bin/python3 /wheels-torch

mkdir -p "$(companion_dist)"

# The constraints file pins the companion's `pip download` to exactly the torch
# that just got installed, so pip can't quietly resolve a different one.
torch_resolved="$(resolved_torch_wheel /wheels-torch)"
/scripts/generate-torch-constraints.sh /wheels-torch > /tmp/torch-constraints.txt

decision="$(companion_decide "${USE_PREBUILT}" "${torch_resolved}")"
case "${decision}" in
    PIP:*) companion_download_prebuilt "${decision}" ;;
    *)     companion_build_from_source "${decision}" ;;
esac
