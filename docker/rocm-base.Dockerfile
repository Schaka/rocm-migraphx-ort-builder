# syntax=docker/dockerfile:1
#
# Self-built ROCm base from TheRock's nightly .deb feed -- what BASE_IMAGE
# defaults to for every other target. AMD publishes no rolling/nightly tag for
# rocm/dev-ubuntu-26.04 at all (see README.md's "Nightly ROCm versioning"), so
# nightly builds this instead. The manual release workflow overrides BASE_IMAGE
# to AMD's own pinned, signed image and never touches this target.
#
# Arch-independent: amdrocm-hpc-sdk pulls in every gfx target's runtime bits in
# one package, so there is exactly one of these, not one per arch. CI builds it
# once, before the arch matrix.
#
# Validated end-to-end: a real gfx1201 build ran clean through the migraphx and
# pytorch targets against this image. The package layout under "amdrocm-*" does
# differ from rocm/dev-ubuntu-26.04 -- see rocm-base.sh for the two fixes that
# papers over.
FROM ubuntu:26.04

ARG THEROCK_DEB_INDEX
# Cache-bust token: see rocm-base.sh.
ARG SOURCE_DATE

RUN --mount=type=bind,source=scripts/build/rocm-base.sh,target=/scripts/build/rocm-base.sh \
    /scripts/build/rocm-base.sh
