#!/bin/sh
# Builds rocBLAS from source for the legacy GCN arches (gfx900/gfx906/gfx90c)
# wherever the base image's prebuilt rocBLAS can't be trusted to serve them:
#   - release builds (ROCM_RELEASE set) always rebuild. AMD's pinned stable
#     bases ship no kernels for these arches at all (7.14 and 10.0 stable have
#     no gfx900/gfx906/gfx90c blas packages), and upstream marks them
#     build-passing but not sanity-tested (TheRock SUPPORTED_GPUS.md), so a
#     version-targeted build never trusts the prebuilt.
#   - nightly builds use the base's prebuilt kernels when they are actually
#     present in this exact base image, and rebuild only when absent -- the
#     fallback is a from-source build, never a substitution of another
#     version's kernels. The presence check runs against the /opt/rocm in this
#     container, so it always answers for the right version.
# A no-op passthrough on every other arch.
# Inputs (build-args): ROCM_ARCH, ROCM_RELEASE, BUILD_PARALLEL_LEVEL,
# LEGACY_GCN_ARCHES.
set -eux

. /scripts/lib/legacy-arch.sh
. /scripts/lib/build-jobs.sh

if ! is_legacy_gcn_arch "${ROCM_ARCH}"; then
    echo "rocblas: ${ROCM_ARCH} uses the base image's prebuilt rocBLAS, no rebuild needed"
    exit 0
fi

if [ -z "${ROCM_RELEASE}" ]; then
    if find -L /opt/rocm -iname "*TensileLibrary*${ROCM_ARCH}*" | grep -q .; then
        echo "rocblas: nightly base ships ${ROCM_ARCH} rocBLAS kernels, using prebuilt -- no rebuild"
        exit 0
    fi
    echo "rocblas: nightly base has no ${ROCM_ARCH} rocBLAS kernels -- building from source"
else
    echo "rocblas: release build (ROCM_RELEASE=${ROCM_RELEASE}) -- rebuilding from source, never trusting prebuilt"
fi

apt-get update
apt-get install -y --no-install-recommends \
    git cmake ninja-build build-essential pkg-config gfortran ccache \
    libmsgpack-cxx-dev wget python3-pip python3-venv
rm -rf /var/lib/apt/lists/*

# Tensile's find_package(msgpackc-cxx CONFIG) wants msgpackc-cxx(-config|Config).cmake
# -- Debian's package installs the same content one letter off, as
# msgpack-cxx-config.cmake (project name "msgpack-cxx", not "msgpackc-cxx";
# libmsgpack-dev itself is just a transitional dummy pulling in the plain-C
# libmsgpack-c-dev, which has no CMake config at all). Symlink under a name
# CMake's default PREFIX/lib/cmake/<pkg> search convention actually matches,
# rather than patching Tensile's CMakeLists for a packaging-naming quirk.
mkdir -p /usr/local/lib/cmake/msgpackc-cxx
for f in /usr/lib/x86_64-linux-gnu/cmake/msgpack-cxx/msgpack-cxx-*.cmake; do
    ln -s "$f" "/usr/local/lib/cmake/msgpackc-cxx/$(basename "$f" | sed 's/^msgpack-cxx/msgpackc-cxx/')"
done

pip3 install --break-system-packages --no-cache-dir pyyaml joblib

# rocBLAS's standalone repo (ROCm/rocBLAS) stopped at v14.3.0 and is deprecated
# -- current development moved into the ROCm/rocm-libraries monorepo, tagged
# "therock-<major>.<minor>" (no patch component -- exactly what ROCM_RELEASE
# already is). Empty ROCM_RELEASE (nightly, fully floating) builds from the
# monorepo's own `develop` branch instead of a pinned tag, matching MIGraphX's
# own default.
#
# Tensile is no longer a git submodule of rocblas -- CMakeLists.txt resolves it
# from ${CMAKE_CURRENT_SOURCE_DIR}/../../shared/tensile, i.e. the monorepo's
# shared/tensile at the same tag, which is why the sparse-checkout below pulls
# that alongside projects/rocblas rather than rocblas alone.
if [ -z "${ROCM_RELEASE}" ]; then
    monorepo_ref="develop"
else
    monorepo_ref="therock-${ROCM_RELEASE}"
fi
echo "rocBLAS source: ROCm/rocm-libraries @ ${monorepo_ref} (projects/rocblas)"

git clone --filter=blob:none --depth 1 --no-checkout \
    --branch "${monorepo_ref}" \
    https://github.com/ROCm/rocm-libraries.git /rocm-libraries-src
cd /rocm-libraries-src
git sparse-checkout init --cone
git sparse-checkout set cmake shared/tensile projects/rocblas
git checkout "${monorepo_ref}"

cd /rocm-libraries-src/projects/rocblas
jobs="$(resolve_build_jobs)"
echo "rocBLAS build: arch ${ROCM_ARCH}, $jobs parallel jobs"
python3 ./rmake.py -i -a "${ROCM_ARCH}" -j "$jobs" --no_hipblaslt

# rmake's -i install target lands under build/release/rocblas-install, not
# /opt/rocm (relative CMAKE_INSTALL_PREFIX) -- the copy below moves it into
# place.
echo "Copying rocBLAS ${ROCM_ARCH} install output into /opt/rocm..."

# This base image lays /opt/rocm out as versioned component dirs (core, core-7,
# core-7.14) with include/lib/share at the top level as convenience symlinks
# into core-7.14/. A plain `cp -a src/. /opt/rocm/` fails outright here: cp won't
# overwrite an existing symlink-named entry with a real directory ("cannot
# overwrite non-directory ... with directory"). Resolve each of
# include/lib/share to what it actually points at first, so content lands in
# core-7.14/ and the top-level symlinks keep working, instead of cp clobbering
# them.
src="/rocm-libraries-src/projects/rocblas/build/release/rocblas-install"
for d in include lib share; do
    [ -e "$src/$d" ] || continue
    real_dest="$(readlink -f "/opt/rocm/$d" 2>/dev/null || echo "/opt/rocm/$d")"
    mkdir -p "$real_dest"
    cp -a "$src/$d/." "$real_dest/"
done
find "$src" -mindepth 1 -maxdepth 1 ! -name include ! -name lib ! -name share \
    -exec cp -a {} /opt/rocm/ \;

# Captured before the rm -rf below deletes $src, so there's still something to
# compare librocblas.so's resolved target against afterward.
built_real="$(find "$src/lib" -maxdepth 1 -name 'librocblas.so.*' ! -type l | head -1)"
built_size="$(stat -c%s "$built_real" 2>/dev/null || echo 0)"
rm -rf /rocm-libraries-src

echo "Verifying ${ROCM_ARCH} Tensile library is present in /opt/rocm..."
if ! find -L /opt/rocm -iname "*TensileLibrary*${ROCM_ARCH}*" | grep -q .; then
    echo "FATAL: /opt/rocm has no ${ROCM_ARCH} Tensile library after the copy." >&2
    exit 1
fi
echo "OK: ${ROCM_ARCH} Tensile library confirmed present in /opt/rocm."

echo "Verifying librocblas.so resolves to the ${ROCM_ARCH} build, not the stock one..."
resolved="$(readlink -f /opt/rocm/lib/librocblas.so)"
if [ "$built_size" = "0" ] || [ "$(stat -c%s "$resolved")" != "$built_size" ]; then
    echo "FATAL: /opt/rocm/lib/librocblas.so resolves to ${resolved}, which is not our freshly-built rocBLAS (size mismatch)." >&2
    echo "The stock base-image rocBLAS is still what actually gets loaded at runtime." >&2
    exit 1
fi

echo "Verifying librocblas.so embeds real ${ROCM_ARCH} device code..."
objcopy -O binary --only-section=.hip_fatbin "$resolved" /tmp/rocblas_fatbin_check.bin
fatbin_size="$(stat -c%s /tmp/rocblas_fatbin_check.bin)"
rm -f /tmp/rocblas_fatbin_check.bin
if [ "$fatbin_size" -lt 1000000 ]; then
    echo "FATAL: librocblas.so's .hip_fatbin is only ${fatbin_size} bytes -- too small to contain real ${ROCM_ARCH} device code (expect several MB)." >&2
    exit 1
fi
echo "OK: librocblas.so resolves to a ${ROCM_ARCH} build with a ${fatbin_size}-byte .hip_fatbin."
