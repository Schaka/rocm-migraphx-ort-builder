#!/bin/sh
# ABI gate for the torchvision/torchaudio wheel, and the tier fallback it feeds.
#
# torch and the companion are resolved by independent `pip download` calls, not
# one joint solve, and the companion's wheel metadata pins a bare `torch` with no
# version constraint (see torch-package-build-decide.sh's own comment on that),
# so pip cannot catch a mismatch and the date-tag match the decision script does
# is not an ABI guarantee: a wheel built against a different libtorch installs
# cleanly and then dies on `import` with an undefined symbol (e.g.
# torch_exception_get_what_without_backtrace). It happens whenever the resolved
# torch did NOT come from the same nightly index the companion wheel did --
# notably on the arches where the pytorch stage falls through to its
# devreleases-snapshot tier.
#
# So an unusable prebuilt wheel drops to the source build rather than failing the
# image, which is the same tier philosophy the pytorch stage applies to its own
# download failures. The second verify is deliberately NOT tolerant -- if a
# from-source build can't import either, that is a real defect, not a bad wheel
# pick.
# Inputs (build-args): TORCH_COMPANION, PYTORCH_VERSION, ROCM_RELEASE, ROCM_ARCH.
set -eu

. /scripts/lib/rocm-env.sh
. /scripts/lib/torch-wheels.sh
. /scripts/lib/torch-companion.sh

use_rocm_runtime_path

if companion_verify; then
    exit 0
fi

echo "Prebuilt ${TORCH_COMPANION} does not load against the resolved torch -- rebuilding from source"
uv pip uninstall --python /build-venv/bin/python3 "${TORCH_COMPANION}" || true
rm -rf "$(companion_dist)"
mkdir -p "$(companion_dist)"

# 0 for use_prebuilt forces the decision script to emit SOURCE.
torch_resolved="$(resolved_torch_wheel /wheels-torch)"
companion_build_from_source "$(companion_decide 0 "${torch_resolved}")"

companion_verify
