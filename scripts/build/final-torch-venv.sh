#!/bin/sh
# Installs torch + torchvision + torchaudio into $VIRTUAL_ENV_TORCH (and, when
# it is safe to, also into $VIRTUAL_ENV).
#
# The three dist dirs are bind-mounted read-only (same layer-size reasoning as
# final-ort-venv.sh), then merged into ONE shared directory before installing --
# not passed as three separate --find-links: pytorch, torchvision and
# torchaudio's independent `pip download` calls each pull their own copy of
# shared transitive deps (filelock, jinja2, sympy, ...) from the same index --
# byte-identical content, but seen at two different local file:// paths uv's
# resolver errors with "conflicting URLs" instead of just picking one.
#
# One cp PER source dir, not a single cp with three globs: the duplicates above
# mean the same basename is supplied twice in one argument list, and GNU cp
# refuses that with "will not overwrite just-created" AND exits non-zero -- it
# only permits clobbering a file an EARLIER invocation wrote. Sequential
# invocations therefore give last-source-wins, and the order below (pytorch,
# then torchvision, then torchaudio) is what decides the winner.
#
# The merge target is a cache mount, so the merged copy is likewise absent from
# the image. It is emptied on entry and on exit: a cache mount persists between
# builds on the same builder, and stale wheels left in it would be picked up by
# the globs below on a later, differently resolved build.
set -eu

. /scripts/lib/torch-wheels.sh

# No LD_LIBRARY_PATH juggling here, unlike the builder stages: this image
# registers /opt/rocm/lib with the dynamic linker via /etc/ld.so.conf.d.
merged=/tmp/wheels-torch
rm -rf "${merged:?}"/*
for d in pytorch torchvision torchaudio; do
    cp -f "/wheels-src/$d"/* "$merged/"
done

install_torch_wheels "$VIRTUAL_ENV_TORCH/bin/python3" "$merged" numpy pillow

"$VIRTUAL_ENV_TORCH/bin/python3" -c "import torch; print('torch', torch.__version__, 'HIP built:', torch.version.hip)"
"$VIRTUAL_ENV_TORCH/bin/python3" -c "import torch, torchvision; print('torchvision', torchvision.__version__)"
"$VIRTUAL_ENV_TORCH/bin/python3" -c "import torch, torchaudio; print('torchaudio', torchaudio.__version__)"

# torchvision must actually carry device code, not merely import. Two distinct
# ways it can fail to, both of which shipped silently before this check existed:
#
#   1. A source build without FORCE_CUDA compiles CPU kernels only -- nms then
#      has no CUDA registration at all (see rocm-env.sh).
#   2. A prebuilt build that ships the arch-agnostic torchvision wheel WITHOUT
#      its amd_torchvision_device_<arch> sibling -- nms registers a CUDA kernel,
#      but _C.so's .hip_fatbin is NOBITS and the first GPU call dies with
#      "CUDA error: invalid kernel file" (see torchvision.Dockerfile).
#
# Neither needs a GPU to detect, which matters: this stage never has one. Case 1
# is a missing dispatcher entry; case 2 is a fatbin section with no file content.
"$VIRTUAL_ENV_TORCH/bin/python3" - <<'PYCHECK'
import sys, torch, torchvision, pathlib

dump = torch._C._dispatch_dump("torchvision::nms")
backends = {l.split(":", 1)[0].strip() for l in dump.splitlines() if ":" in l}
if "CUDA" not in backends:
    sys.exit("FATAL: torchvision has no CUDA/HIP kernel for nms -- built without GPU support. "
             "Registered backends: %s" % sorted(backends))

# Section headers are parsed here rather than shelled out to readelf/objcopy:
# this image ships no binutils, and adding them just for a build-time assertion
# would put them in the published rootfs forever.
def hip_fatbin_bytes(path):
    """Bytes of .hip_fatbin actually present in the file (0 if NOBITS/absent)."""
    import struct
    with open(path, "rb") as fh:
        data = fh.read()
    if data[:4] != b"\x7fELF" or data[4] != 2:      # ELF64 only
        return None
    e_shoff, = struct.unpack_from("<Q", data, 0x28)
    e_shentsize, e_shnum, e_shstrndx = struct.unpack_from("<HHH", data, 0x3A)
    if not e_shoff or not e_shnum:
        return None

    def sh(i):
        off = e_shoff + i * e_shentsize
        name, typ = struct.unpack_from("<II", data, off)
        size, = struct.unpack_from("<Q", data, off + 0x20)
        sh_off, = struct.unpack_from("<Q", data, off + 0x18)
        return name, typ, sh_off, size

    _, _, str_off, _ = sh(e_shstrndx)
    for i in range(e_shnum):
        name, typ, _, size = sh(i)
        end = data.index(b"\0", str_off + name)
        if data[str_off + name:end] == b".hip_fatbin":
            return 0 if typ == 8 else size   # SHT_NOBITS == 8
    return None

# A NOBITS .hip_fatbin is only a fault on the SOURCE path. TheRock's prebuilt
# wheels are split arch-agnostic/per-arch: the generic torchvision wheel ships a
# _C.so whose .hip_fatbin is deliberately NOBITS, and the real code objects come
# from its amd_torchvision_device_<arch> sibling as separate files. So decide by
# whether that sibling is installed, not by the fatbin alone -- checking the
# fatbin unconditionally rejects a perfectly good prebuilt install.
import importlib.metadata as md

device_pkgs = sorted(
    name for d in md.distributions()
    for name in [(d.metadata["Name"] or "").replace("_", "-")]
    if name.startswith("amd-torchvision-device-")
)

so = pathlib.Path(torchvision.__file__).parent / "_C.so"
n = hip_fatbin_bytes(so) if so.exists() else None

if device_pkgs:
    # Prebuilt split layout: the device package is what must carry real content.
    print("torchvision device package(s):", ", ".join(device_pkgs))
    # Size of the largest non-metadata file it ships, whatever the container
    # format happens to be. Deliberately NOT an extension whitelist: TheRock
    # currently packs the code objects as torchvision/.kpack/*.kpack, and a
    # whitelist of .so/.hsaco/.co rejected a perfectly good wheel. The only
    # thing worth asserting is that real bytes of device code are present.
    objs = [p for d in md.distributions()
            if (d.metadata["Name"] or "").replace("_", "-") in device_pkgs
            for p in (d.files or [])
            if ".dist-info/" not in str(p)]
    sizes = [(p.locate().stat().st_size, str(p)) for p in objs if p.locate().exists()]
    biggest, biggest_name = max(sizes, default=(0, "<none>"))
    if biggest < 4096:
        sys.exit("FATAL: %s is installed but its largest payload file is %s at %d bytes -- the "
                 "per-arch wheel is a stub, and every GPU op will die with "
                 "'CUDA error: invalid kernel file'." % (", ".join(device_pkgs), biggest_name, biggest))
    print("torchvision device code present: %s (%d bytes); generic _C.so .hip_fatbin %s"
          % (biggest_name, biggest, n))
elif n == 0:
    sys.exit("FATAL: torchvision _C.so has an empty/NOBITS .hip_fatbin and no "
             "amd_torchvision_device_<arch> package is installed -- the wheel carries no device "
             "code at all, and the first GPU op will die with 'CUDA error: invalid kernel file'.")
else:
    print("torchvision GPU device code present (source build, .hip_fatbin bytes: %s)" % n)
PYCHECK

# rocm_sdk_core's presence in the collected wheels is the tell for which tier
# actually fired: only the PIP tier pulls in TheRock's rocm_sdk_* packages (the
# second comgr/LLVM copy this venv split exists to keep out of $VIRTUAL_ENV --
# see final-ort-venv.sh). The SOURCE fallback links directly against this
# image's own classic /opt/rocm, no rocm_sdk involved, so it carries none of that
# risk -- safe to ALSO install into $VIRTUAL_ENV, restoring the zero-friction
# "torch just works from the default python3" behaviour for whichever nightly
# runs happen to fall back to it, without reintroducing the crash on runs that
# resolve the PIP wheel.
if ls "$merged"/rocm_sdk_core-*.whl >/dev/null 2>&1; then
    echo "PIP-tier torch (rocm_sdk_core present) -- staying isolated to $VIRTUAL_ENV_TORCH"
else
    echo "SOURCE-tier torch (no rocm_sdk_core) -- also merging into $VIRTUAL_ENV"
    install_torch_wheels "$VIRTUAL_ENV/bin/python3" "$merged" numpy pillow
    "$VIRTUAL_ENV/bin/python3" -c \
        "import torch; print('torch', torch.__version__, 'HIP built:', torch.version.hip, '(merged into main venv)')"
    "$VIRTUAL_ENV/bin/python3" -c \
        "import onnxruntime as ort; p=ort.get_available_providers(); assert 'MIGraphXExecutionProvider' in p, 'MIGraphX EP lost after merging torch into main venv'; print('MIGraphX EP intact after merge:', p)"
fi

rm -rf "${merged:?}"/*
