# syntax=docker/dockerfile:1
#
# The published product: a drop-in BASE_IMAGE for downstream Dockerfiles that
# expect the `rocm/onnxruntime` convention -- onnxruntime (built with
# MIGraphXExecutionProvider) preinstalled into a venv at /opt/venv, with
# /opt/rocm carrying the from-source MIGraphX and its ROCm runtime deps, plus
# torch/torchvision/torchaudio in a second venv at /opt/venv-torch.
#
# This target compiles nothing. Every component arrives as a prebuilt image
# through a named context, and all this does is pull them and install wheels --
# which is load-bearing, not incidental: a hosted runner does not have the disk
# for it to build any of these itself on top of the ~10GB ROCm base it already
# has to pull.
FROM python-base

# See docker-bake.hcl: 0 only on gfx900/gfx906/gfx90c, a no-op 1 everywhere else.
# PyTorch links hipBLASLt unconditionally regardless of arch
# (cmake/Dependencies.cmake), but hipBLASLt's own Tensile Logic tree has never
# had gfx900/gfx906/gfx90c kernels (oldest arch present is arcturus/gfx908) -- 0
# forces GEMMs onto the rocBLAS rebuilt in the rocblas target instead of a
# library with nothing to dispatch to.
ARG TORCH_BLAS_PREFER_HIPBLASLT
ENV TORCH_BLAS_PREFER_HIPBLASLT=${TORCH_BLAS_PREFER_HIPBLASLT}

COPY --from=migraphx /opt/rocm /opt/rocm

# Unlike rocm/onnxruntime (which ships an /etc/ld.so.conf.d entry for its ROCm
# lib dir), these base images don't register /opt/rocm/lib with the dynamic
# linker at all -- libs like libhiprand.so.1 exist on disk but aren't found at
# runtime (e.g. by ctranslate2) without this, failing with "cannot open shared
# object file" despite the file being right there.
RUN echo "/opt/rocm/lib" > /etc/ld.so.conf.d/rocm.conf && ldconfig

# libprotobuf is a runtime dependency of libonnxruntime_providers_migraphx.so
# (MIGraphX links against it), libopenblas of torch's linear-algebra ops, and
# torchvision/torchaudio link against various image/audio codecs. All were
# apt-installed in builder stages but not carried into this image's rootfs.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libprotobuf-dev libopenblas0 \
        libjpeg-turbo-progs libpng-dev libfreetype-dev libtiff-dev libwebp-dev \
        libsndfile1 libflac-dev libvorbis-dev libopus-dev \
    && rm -rf /var/lib/apt/lists/*

# uv's own download cache would otherwise be committed into this image's layers;
# nothing here benefits from it surviving the build.
ENV UV_NO_CACHE=1

# This image's own python3 is 3.14 (native to the base), but every wheel
# installed below targets 3.12 -- so the venvs have to be 3.12 too, via the same
# uv-managed standalone Python the builder targets used (uv itself and the 3.12
# interpreter are both inherited from python-base, no re-download needed).
ENV PYTHONPATH=/opt/rocm/lib
ENV VIRTUAL_ENV=/opt/venv
# --seed: uv venvs ship no pip binary by default, but this image is a drop-in for
# rocm/onnxruntime:* bases whose venvs do -- downstream Dockerfiles may call
# "$VIRTUAL_ENV/bin/pip" directly.
RUN uv venv $VIRTUAL_ENV --python 3.12 --seed

RUN --mount=type=bind,from=ort,source=/onnxruntime/dist,target=/wheels-ort \
    --mount=type=bind,source=scripts/build/final-ort-venv.sh,target=/scripts/build/final-ort-venv.sh \
    /scripts/build/final-ort-venv.sh

# torch/torchvision/torchaudio always get their own venv -- see
# scripts/build/final-ort-venv.sh for why. Not added to PATH: torch is opt-in via
# this venv's own interpreter path, never the default `python3` a downstream
# Dockerfile gets from this image.
ENV VIRTUAL_ENV_TORCH=/opt/venv-torch
RUN uv venv $VIRTUAL_ENV_TORCH --python 3.12 --seed

RUN --mount=type=bind,from=pytorch,source=/pytorch/dist,target=/wheels-src/pytorch \
    --mount=type=bind,from=torchvision,source=/torchvision/dist,target=/wheels-src/torchvision \
    --mount=type=bind,from=torchaudio,source=/torchaudio/dist,target=/wheels-src/torchaudio \
    --mount=type=cache,target=/tmp/wheels-torch,sharing=locked,id=final-wheels-torch \
    --mount=type=bind,source=scripts/lib,target=/scripts/lib \
    --mount=type=bind,source=scripts/build/final-torch-venv.sh,target=/scripts/build/final-torch-venv.sh \
    /scripts/build/final-torch-venv.sh

# This image is a drop-in BASE_IMAGE for downstream Dockerfiles -- any of them
# can `pip install`/`uv pip install` something that pulls in torch,
# torchvision, torchaudio or onnxruntime as a transitive dependency (exactly
# what happened installing faster-whisper: ctranslate2 depends on
# onnxruntime, and pip silently swapped this image's MIGraphX-EP build for a
# generic CPU-only PyPI wheel, no error, no warning). Pinning the exact
# already-installed version via a constraints file turns that from a silent
# swap into a loud resolution failure -- pip/uv can't satisfy the pin from
# any index, they can only reuse what's already installed. Both env vars are
# set because pip reads PIP_CONSTRAINT and uv reads UV_CONSTRAINT; neither
# tool honors the other's. onnxruntime's build tags itself with a PEP 440
# local version (+<ROCM_ARCH>, see scripts/build/ort.sh's VERSION_NUMBER
# edit) specifically so this pin is enforceable -- an exact-version pin PyPI
# could also satisfy would defeat the whole point.
RUN { "$VIRTUAL_ENV/bin/pip" freeze --local | grep -E '^onnxruntime==' || true; \
      "$VIRTUAL_ENV_TORCH/bin/pip" freeze --local | grep -E '^(torch|torchvision|torchaudio)==' || true; \
    } > /opt/pip-constraints.txt \
    && cat /opt/pip-constraints.txt
ENV PIP_CONSTRAINT=/opt/pip-constraints.txt
ENV UV_CONSTRAINT=/opt/pip-constraints.txt

ENV PATH="$VIRTUAL_ENV/bin:${PATH}"
