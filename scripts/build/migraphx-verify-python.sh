#!/bin/sh
# Confirms MIGraphX's python module was built against the uv-managed 3.12 and
# not the base image's native 3.14. Fails the build here, loudly, rather than
# shipping a migraphx.so downstream consumers' 3.12 venvs can't import
# (silently, until someone's `import migraphx` breaks).
# See the python3.12-config discussion in migraphx.sh for why this can go wrong.
set -eu

if ! find /opt/rocm -iname "migraphx.cpython-312-*.so" | grep -q .; then
    echo "FATAL: no migraphx.cpython-312-*.so under /opt/rocm -- python module built for the wrong interpreter (see migraphx.sh)." >&2
    find /opt/rocm -iname "migraphx.cpython-*.so" >&2
    exit 1
fi
echo "OK: migraphx python module built for cpython-312."
