#!/bin/bash
set -eu

# Emit a pip constraints file (one "name==version" per line) from the wheels/sdists already
# downloaded for pytorch, so a LATER independent `pip download` (for torchvision/torchaudio)
# is forced to reuse those exact versions for shared transitive deps (torch itself, triton,
# rocm-sdk-core, rocm-sdk-device-<arch>, rocm-sdk-libraries, the `rocm` metapackage) instead of
# re-resolving them fresh and potentially landing on a different date-tagged nightly build --
# torchvision/torchaudio's own metadata only pins a bare, unversioned `torch` requirement, so
# without this constraint pip is free to pick whatever's newest at the moment of ITS OWN
# resolution, which can silently diverge from what pytorch-builder already chose and installed.
#
# Usage: generate-torch-constraints.sh <wheel_dir> > constraints.txt

WHEEL_DIR=$1

for f in "$WHEEL_DIR"/*.whl; do
    [ -e "$f" ] || continue
    base=$(basename "$f" .whl)
    # Wheel filename spec: {name}-{version}(-{build})?-{pytag}-{abitag}-{platform}.whl -- name
    # and version themselves never contain a literal hyphen (normalized to underscores), so the
    # last 3 dash-fields are always pytag-abitag-platform; everything before that is name-version.
    # (avoid `rev` -- not guaranteed present in minimal build images; awk handles this natively)
    name_ver=$(echo "$base" | awk -F'-' '{out=$1; for(i=2;i<=NF-3;i++) out=out"-"$i; print out}')
    name=$(echo "$name_ver" | cut -d'-' -f1)
    ver=$(echo "$name_ver" | cut -d'-' -f2-)
    echo "${name}==${ver}"
done

for f in "$WHEEL_DIR"/*.tar.gz; do
    [ -e "$f" ] || continue
    base=$(basename "$f" .tar.gz)
    name=$(echo "$base" | awk -F'-' '{out=$1; for(i=2;i<NF;i++) out=out"-"$i; print out}')
    ver=$(echo "$base" | awk -F'-' '{print $NF}')
    echo "${name}==${ver}"
done
