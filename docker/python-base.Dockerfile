# syntax=docker/dockerfile:1
#
# Shared ancestor for every builder target and for the final image.
#
# The ROCm base image's native python3 is 3.14, but AudioMuse-AI (and other
# downstream consumers) pin numpy in a way that only resolves against onnx's deps
# under 3.12 -- so every wheel this repo builds, and the final venv, all target
# uv-managed Python 3.12 instead. Installing uv and downloading that interpreter
# here, once, means the other targets don't each re-fetch the same ~30MB
# standalone interpreter.
#
# BASE_IMAGE has no default here on purpose: docker-bake.hcl owns it (and every
# other version-shaped variable in this repo), so there is one place to change
# it. Build through `docker buildx bake`, not `docker build`.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh \
    && uv python install 3.12
