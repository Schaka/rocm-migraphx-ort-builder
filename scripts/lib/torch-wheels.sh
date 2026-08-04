# shellcheck shell=sh
# Sourced, never executed.
#
# In the builder stages, source rocm-env.sh and call use_rocm_runtime_path()
# first -- they do not register /opt/rocm/lib with the dynamic linker, and torch
# wheels fail to import without it. The final image does register it, so it
# needs nothing.

# install_torch_wheels <venv-python> <wheel-dir> [extra packages...]
#
# Installs every wheel in <wheel-dir> into <venv-python>'s environment, plus the
# named runtime dependencies.
#
# --no-deps + the explicit package list: pytorch's devreleases-snapshot fallback
# tier (see rocm-devrelease-snapshot.py) downloads a self-consistent set of
# wheels whose OWN declared Requires-Dist pins (e.g.
# "rocm-sdk-device-<arch>==7.14.0") don't match what was actually downloaded --
# AMD's devreleases metadata always declares that abstract pin, regardless of
# which nightly snapshot the file actually is -- so letting pip re-resolve
# dependencies here would re-check that pin and fail. The named packages are
# torch's own ordinary, non-ROCm-pinned runtime deps, harmless to (re-)request
# explicitly on every path: satisfied instantly from <wheel-dir> when the PIP or
# SNAPSHOT tier already fetched them, pulled fresh from PyPI when a SOURCE build
# didn't.
#
# The tar.gz pass: the rocm-wheel-fetch/snapshot tiers can also produce a
# non-.whl sdist (the `rocm` metapackage itself, when devreleases had no clean
# release for it). torch's own _rocm_init.py imports the `rocm_sdk` module that
# sdist provides, so it must be installed too, not just the *.whl glob.
# --no-build-isolation: its build backend (setuptools) is already present -- no
# need to let pip fetch one from the network.
install_torch_wheels() {
    _py="$1"
    _dir="$2"
    shift 2

    uv pip install --python "$_py" --no-deps --find-links "$_dir" \
        filelock typing_extensions sympy networkx jinja2 fsspec mpmath MarkupSafe setuptools \
        "$@" \
        "$_dir"/*.whl

    if ls "$_dir"/*.tar.gz >/dev/null 2>&1; then
        uv pip install --python "$_py" --no-deps --no-build-isolation \
            --find-links "$_dir" "$_dir"/*.tar.gz
    fi
}

# resolved_torch_wheel <wheel-dir> -- basename of the torch wheel that actually
# landed there, which torch-package-build-decide.sh needs to match a companion
# package's build against.
resolved_torch_wheel() {
    basename "$(ls "$1"/torch-*.whl | head -1)"
}
