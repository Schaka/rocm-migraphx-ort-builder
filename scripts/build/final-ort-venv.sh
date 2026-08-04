#!/bin/sh
# Installs onnxruntime + migraphx into $VIRTUAL_ENV -- the "classic" /opt/rocm
# stack's own Python surface -- and gates on both actually importing.
#
# Kept in its own install step (not merged with torch's wheels) because torch's
# prebuilt wheel bundles TheRock's pip-packaged ROCm SDK (its own
# libamd_comgr.so/libLLVM.so under site-packages/_rocm_sdk_core/lib), a second
# copy of comgr/LLVM independent of this image's classic /opt/rocm one. Any
# process that ends up loading both (e.g. ctranslate2 -- built against classic
# /opt/rocm -- opportunistically does `import torch` in its specs/model_spec
# module if torch happens to be importable) aborts at startup: LLVM's global
# CommandLine option registry rejects the second registration of the same option
# ("CommandLine Error: Option 'spirv-expand-step' registered more than once!").
# Keeping torch out of $VIRTUAL_ENV entirely means classic-stack consumers never
# see it importable, so that opportunistic import just ImportErrors harmlessly
# instead of loading a second comgr/LLVM.
#
# Wheels arrive on a bind mount rather than via COPY: a COPY writes them into
# this image's own filesystem and commits a layer, and no later `rm -rf` can
# un-commit it, so the wheels would ship inside the published image (several GB
# of them, between this install and the torch one). A bind mount exposes the
# source stage's directory read-only for the duration of the RUN and contributes
# nothing to the layer -- only what pip actually installs is kept.
set -eu

uv pip install --python "$VIRTUAL_ENV/bin/python3" --no-deps --find-links /wheels-ort \
    numpy flatbuffers packaging protobuf \
    /wheels-ort/*.whl

"$VIRTUAL_ENV/bin/python3" -c \
    "import onnxruntime as ort; p=ort.get_available_providers(); print('ORT providers:', p); assert 'MIGraphXExecutionProvider' in p"
"$VIRTUAL_ENV/bin/python3" -c "import migraphx; print('migraphx python module OK')"
