#!/bin/sh
# Installs TheRock's nightly ROCm .deb snapshot into a plain ubuntu image and
# reshapes /opt/rocm into the classic layout every builder stage expects.
# Inputs (build-args): THEROCK_DEB_INDEX, SOURCE_DATE.
set -eux

# SOURCE_DATE is referenced (as a no-op echo) purely to bust this layer's cache:
# the command text never changes, so without it `cache-from` would replay
# whatever dated snapshot was resolved the very first time this stage was ever
# built, forever. CI passes today's date here on every nightly run.
echo "rocm-builder snapshot: ${SOURCE_DATE}"

apt-get update
apt-get install -y --no-install-recommends ca-certificates curl gnupg2
rm -rf /var/lib/apt/lists/*

# TheRock's deb feed has no "latest" alias, only dated YYYYMMDD-<run-id>
# directories -- the same "resolve newest yourself" problem the wheel-index
# discovery in torch-package-build-decide.sh solves, just for a directory
# listing instead of wheel filenames. The bare index URL serves a stale cached
# snapshot (observed capped at a months-old page); a throwaway query string
# busts that cache and gets the real, current listing back.
latest=$(curl -s "${THEROCK_DEB_INDEX}?_=$(date +%s)" \
    | grep -oE 'href="[0-9]{8}-[0-9]+/index.html"' \
    | sed 's/href="//; s|/index.html"||' | sort -V | tail -1)
if [ -z "$latest" ]; then
    echo "FATAL: no dated build dir found under ${THEROCK_DEB_INDEX}" >&2
    exit 1
fi
echo "Using TheRock nightly deb snapshot: ${latest}"

# No GPG verification ([trusted=yes]) -- this is an internal nightly-only path,
# not the release path (which overrides BASE_IMAGE to AMD's own signed
# rocm/dev-ubuntu image instead); acceptable for the same reason nightly already
# accepts cross-component date drift (README.md).
echo "deb [trusted=yes] ${THEROCK_DEB_INDEX}${latest}/ stable main" \
    > /etc/apt/sources.list.d/therock-nightly.list

# amdrocm-hpc-sdk pulls "-dev" packages for peripheral libs (rocALUTION,
# hipTensor) but NOT for the core HIP runtime itself -- confirmed by `dpkg -l`
# after install: amdrocm-runtime10.1 (the runtime .so) was present,
# amdrocm-runtime-dev/amdrocm-core-dev (headers + HIP's own hip-config.cmake)
# were not, which is exactly why rocMLIR's cmake configure step fails with
# "Could not find a package configuration file provided by hip". Install them
# explicitly.
#
# amdrocm-llvm/-llvm-dev carry the ROCm device compiler (clang/clang++/lld and
# the amdgcn device bitcode), and amdrocm-hpc-sdk does NOT depend on either: its
# Depends line lists only its own per-gfx amdrocm-hpc-sdk10.1-gfx<arch>
# siblings. Any clang that shows up without naming these explicitly arrived
# through some peripheral library's transitive dependency, which is not a
# contract -- name them, so the compiler every builder stage invokes is actually
# a declared input of this base.
apt-get update
apt-get install -y --no-install-recommends \
    amdrocm-hpc-sdk amdrocm-core-dev amdrocm-runtime-dev \
    amdrocm-llvm amdrocm-llvm-dev
rm -rf /var/lib/apt/lists/*

# amdrocm-hpc-sdk lands everything under /opt/rocm/core-<major.minor>/ (bin,
# include, lib, libexec, llvm, share, amdgcn) with NO top-level convenience
# symlinks -- unlike rocm/dev-ubuntu-26.04, which symlinks include/lib/share/etc
# at /opt/rocm/ straight into its own versioned core dir. Every other stage
# hardcodes paths like /opt/rocm/llvm/bin/clang++ and /opt/rocm/lib -- recreate
# those top-level symlinks here so this base is a drop-in for what they expect,
# instead of patching every consumer.
core_dir=$(find /opt/rocm -mindepth 1 -maxdepth 1 -type d -name 'core-*' | head -1)
if [ -z "$core_dir" ]; then
    echo "FATAL: no /opt/rocm/core-* dir found after install" >&2
    exit 1
fi
for d in "$core_dir"/*; do
    ln -s "$(basename "$core_dir")/$(basename "$d")" "/opt/rocm/$(basename "$d")"
done

# The llvm child the loop just linked is a decoy. amdrocm-llvm ships TWO llvm
# trees: core-<ver>/llvm holds nothing but
# bin/{clang,clang++,clang-cl,clang-cpp,clang-23,flang,rocm}.cfg -- seven config
# files and not one executable -- while the actual toolchain (clang, clang++,
# lld, amdclang, llvm-*, the amdgcn device bitcode) sits one level deeper at
# core-<ver>/lib/llvm, whose own bin/ carries those same seven .cfg files. The
# shallow tree is therefore a strict subset, and /opt/rocm/llvm ->
# core-<ver>/llvm is a directory that exists, satisfies any [ -e ] test, and
# still has no compiler in it -- which is exactly how a base image with no
# usable clang gets published.
#
# Repoint it at whichever tree actually contains bin/clang++ (searched for, not
# hardcoded, since which of the two upstream populates has changed before).
# AMD's rocm/dev-ubuntu-26.04 exposes the compiler at /opt/rocm/llvm/bin/clang++,
# that is the path every stage hardcodes, and a pinned release build against
# AMD's own image gets it for free -- so this base has to present it under the
# same name.
clangxx=$(find "$core_dir" -maxdepth 4 -path '*/bin/clang++' | head -1)
if [ -z "$clangxx" ]; then
    echo "FATAL: no bin/clang++ anywhere under ${core_dir} -- amdrocm-llvm did not install a compiler." >&2
    find "$core_dir" -maxdepth 3 -type d -name llvm >&2
    exit 1
fi
llvm_root=$(dirname "$(dirname "$clangxx")")
# rm targets the loop's symlink itself (no trailing slash), never the tree it points at.
rm -f /opt/rocm/llvm
ln -s "${llvm_root#/opt/rocm/}" /opt/rocm/llvm

# Hard gate on the result. Without it a compiler-less base still builds and
# publishes happily, and the breakage only surfaces hours later in the migraphx
# stage as cmake's "is not a full path to an existing compiler tool" -- or,
# worse, as that stage's unrelated-looking python-ABI guard. Fail here, where
# the cause is one line up.
/opt/rocm/llvm/bin/clang++ --version
echo "OK: /opt/rocm/llvm/bin/clang++ present (-> $(readlink -f /opt/rocm/llvm))"
