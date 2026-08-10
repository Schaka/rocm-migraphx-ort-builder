# Build graph for the ROCm 7 pipeline. One target per component, one Dockerfile
# per target under docker/.
#
#   python-base ──┬─ rocblas ──┬─ migraphx ── ort ──┐
#                 │            ├─ pytorch ──┬───────┼── final
#                 │            ├─ torchvision ──────┤
#                 │            └─ torchaudio ───────┘
#                 └─ (ort and final also start from python-base directly)
#
#   rocm-base is standalone: the self-built ROCm base image BASE_IMAGE defaults
#   to. It is arch-independent and built once, ahead of the arch matrix.
#
# This file is the single source of truth for every version-shaped knob, for the
# image/tag naming scheme, and for the registry cache refs. The Dockerfiles
# declare bare ARGs and get their values from here; the workflows set variables
# and name a target, and carry no build logic of their own.
#
# Local use:
#   docker buildx bake final                     # build everything, one arch list
#   docker buildx bake migraphx --set '*.args.ROCM_ARCH=gfx1201'
#   docker buildx bake --print final             # resolved graph, no build
#
# CI use: set the variables below as environment variables and name one target.
# Every variable is overridable from the environment -- that is how the workflows
# pass a single arch, a pinned ROCm release, or a prebuilt component image.

# ---------------------------------------------------------------------------
# Registry / naming
# ---------------------------------------------------------------------------

variable "REGISTRY" { default = "ghcr.io" }

# Registry build cache is CI-only: exporting it needs write access to the
# packages, which a local `docker buildx bake` has no reason to have (and would
# fail on). Set REGISTRY_CACHE=true in CI; leave it alone locally.
variable "REGISTRY_CACHE" { default = "false" }

# The component this run is actually building. Only that target exports cache:
# when a job builds e.g. `migraphx`, bake also builds the `rocblas` target it
# depends on, and letting a dependency export its own cache would write a
# cache-<arch> tag under a package that has no image for that arch, on every
# single job. Reading a dependency's cache is free and useful, so cache_from is
# not gated -- only cache_to is. Exactly one cache export per job, as before.
variable "CACHE_TARGET" { default = "" }

function "cache_from" {
  params = [component]
  result = REGISTRY_CACHE == "true" ? ["type=registry,ref=${cache_ref(component)}"] : []
}

function "cache_to" {
  params = [component]
  result = REGISTRY_CACHE == "true" && CACHE_TARGET == component ? ["type=registry,ref=${cache_ref(component)},mode=max"] : []
}

# Lowercased repository owner. CI passes ${GITHUB_REPOSITORY_OWNER,,}.
variable "OWNER" { default = "schaka" }

# Each component publishes to its OWN package rather than sharing one package
# under many component-prefixed tags, so a package's tags are only that
# component's arches -- no incomplete-image tags cluttering the published
# product. Downstream consumers want `final`; the rest is build plumbing.
variable "PACKAGES" {
  default = {
    rocm-base   = "rocm-builder"
    rocblas     = "rocm-rocblas-builder"
    migraphx    = "rocm-migraphx-builder"
    pytorch     = "rocm-migraphx-torch-builder"
    torchvision = "rocm-torchvision-builder"
    torchaudio  = "rocm-torchaudio-builder"
    ort         = "rocm-migraphx-ort-builder"
    final       = "rocm-migraphx-ort-torch-builder"
  }
}

# YYYYMMDD, for the dated nightly tags. Empty (the local default) publishes no
# dated tag at all rather than an invalid ":-gfx1201".
variable "DATE" { default = "" }

# ---------------------------------------------------------------------------
# What to build
# ---------------------------------------------------------------------------

# Semicolon-separated GPU_TARGETS list. The default matches the breadth AMD's own
# published images build for (CDNA1-3, RDNA2-4); CI always passes exactly one
# arch, which is what keeps a single job's disk footprint to one arch's worth of
# kernel binaries. Note the legacy-GCN special-casing only fires for a
# single-arch build.
variable "ROCM_ARCH" {
  default = "gfx900;gfx90c;gfx906;gfx908;gfx90a;gfx942;gfx950;gfx1010;gfx1011;gfx1012;gfx1030;gfx1031;gfx1032;gfx1034;gfx1035;gfx1036;gfx1100;gfx1101;gfx1102;gfx1103;gfx1150;gfx1151;gfx1152;gfx1153;gfx1200;gfx1201"
}

# Arches whose prebuilt ROCm packages carry no code objects at all -- not in
# rocBLAS, not in hipBLASLt, not in composable_kernel. Drives the rocBLAS
# from-source rebuild, the composable_kernel/hipBLASLt opt-outs, and the final
# image's TORCH_BLAS_PREFER_HIPBLASLT. Single source of truth: the build scripts
# read this same value (see scripts/lib/legacy-arch.sh). Extend it here.
variable "LEGACY_GCN_ARCHES" { default = "gfx900 gfx906 gfx90c" }

# Base image every stage starts from. Defaults to our own nightly-built ROCm base
# (the rocm-base target) because AMD publishes no rolling/nightly tag for
# rocm/dev-ubuntu-26.04 at all -- see README.md's "Nightly ROCm versioning". The
# release workflow overrides this to AMD's own pinned tag.
variable "BASE_IMAGE" { default = "ghcr.io/schaka/rocm-builder:latest" }

variable "THEROCK_DEB_INDEX" { default = "https://rocm.nightlies.amd.com/packages-multi-arch/deb/" }

# Git ref to build MIGraphX from. The moving `develop` branch by default;
# override to pin a stable release branch (e.g. release/rocm-rel-7.13) when
# develop regresses on a given GPU target. Reflected in the migraphx/ort/final
# image tags, so a pinned build doesn't collide with the develop-tracking one.
variable "MIGRAPHX_REF" { default = "develop" }

# Set for a pinned release build (e.g. "rocm7.14"). Replaces the
# MIGRAPHX_REF-derived tag suffix with this exact string on EVERY component, and
# makes `final` publish a single immutable "<release>-<arch>" tag instead of the
# develop-tracking :latest-<arch>/:<date>-<arch> pair.
variable "RELEASE_TAG" { default = "" }

# major.minor (e.g. "7.14"). Empty = nightly-mode wheel discovery, and rocBLAS
# builds from rocm-libraries' own develop branch instead of a pinned
# therock-<release> tag.
variable "ROCM_RELEASE" { default = "" }

# Nightly CI passes an explicit empty string here so wheel discovery floats on
# pytorch's version too; a release build pins it. The non-empty default is what a
# local build gets, matching the behaviour before this file existed.
variable "PYTORCH_VERSION" { default = "v2.13.0" }

variable "ORT_VERSION" { default = "v1.28.0" }

# "1" tries AMD's prebuilt wheels for pytorch/torchvision/torchaudio first,
# falling back to source per package; "0" forces a full from-source build.
variable "USE_PREBUILT" { default = "1" }

# "auto" sizes the compile job count from MemAvailable; see
# scripts/lib/build-jobs.sh.
variable "BUILD_PARALLEL_LEVEL" { default = "auto" }

# Cache-bust token for the moving-branch clones (MIGraphX's develop, TheRock's
# dated deb snapshot). Their RUN text never changes, so without this a registry
# cache hit would replay the same stale commit/snapshot forever. CI passes the
# build date. Only varied for develop -- a pinned ref's own commits already
# change the cache key when it moves.
variable "SOURCE_DATE" { default = "unknown" }

# ---------------------------------------------------------------------------
# Component wiring
#
# Each of these selects where a target gets a dependency from: the in-tree target
# (built in the same run) or an already-published component image. CI sets the
# ones whose component ran as its own job, so nothing gets recompiled; a local
# build leaves them all at "target:" and builds the whole graph in one shot.
#
# The published-image refs are computed below from the same naming scheme the
# tags use, so a workflow never has to spell an image ref out.
# ---------------------------------------------------------------------------

variable "WITH_ROCBLAS_IMAGE"     { default = "false" }
variable "WITH_MIGRAPHX_IMAGE"    { default = "false" }
variable "WITH_PYTORCH_IMAGE"     { default = "false" }
variable "WITH_TORCHVISION_IMAGE" { default = "false" }
variable "WITH_TORCHAUDIO_IMAGE"  { default = "false" }
variable "WITH_ORT_IMAGE"         { default = "false" }

# ---------------------------------------------------------------------------
# Derived values
# ---------------------------------------------------------------------------

function "slug" {
  params = [s]
  result = trim(regex_replace(s, "[^A-Za-z0-9]+", "-"), "-")
}

function "pkg" {
  params = [component]
  result = "${REGISTRY}/${OWNER}/${PACKAGES[component]}"
}

# MIGRAPHX_REF only changes what the migraphx component actually builds, but ort
# and final both COPY its /opt/rocm output, so they (and their published tags)
# key off the same ref too -- otherwise a pinned-ref rebuild would silently
# overwrite the develop-tracking ort/final tags, or reuse a stale develop-built
# migraphx layer via cache. rocblas, pytorch, torchvision and torchaudio never
# touch MIGraphX, so they keep the plain arch tag regardless of the ref and a
# pinned-ref run reuses the same images.
#
# A release build is a different case entirely: RELEASE_TAG applies to EVERY
# component, including pytorch and rocblas -- both actually do vary by ROCm
# release. Exempting them would tag a release build's images with the same plain
# "<arch>" tag nightly uses, silently overwriting whichever build ran last.
function "suffix" {
  params = [component]
  result = RELEASE_TAG != "" ? "-${RELEASE_TAG}" : (
    contains(["migraphx", "ort", "final"], component) && MIGRAPHX_REF != "develop"
      ? "-${slug(MIGRAPHX_REF)}"
      : ""
  )
}

function "tag_arch" {
  params = [component]
  result = "${slug(ROCM_ARCH)}${suffix(component)}"
}

function "image" {
  params = [component]
  result = "${pkg(component)}:${tag_arch(component)}"
}

function "cache_ref" {
  params = [component]
  result = "${pkg(component)}:cache-${tag_arch(component)}"
}

# "target:<name>" builds the component here; "docker-image://<ref>" pulls the
# already-published one and drops that target out of the graph entirely.
function "ctx" {
  params = [component, use_image]
  result = use_image == "true" ? "docker-image://${image(component)}" : "target:${component}"
}

function "is_legacy_arch" {
  params = [arch]
  result = contains(split(" ", LEGACY_GCN_ARCHES), arch)
}

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

group "default" {
  targets = ["final"]
}

# Shared defaults. Nothing consumes the provenance attestation, and producing it
# turns every push into an index with a second, attestation-only manifest.
#
# Args are deliberately NOT set here: buildx warns about a build-arg no ARG in
# the Dockerfile consumes, and spreading every arg over every target would make
# that warning constant background noise. Each target below passes exactly what
# its own Dockerfile declares, which doubles as a readable summary of what that
# stage actually depends on.
target "_common" {
  context = "."
  attest  = ["type=provenance,disabled=true"]
}

# Deliberately no registry cache: it is two cheap layers (uv plus a ~30MB
# standalone interpreter) on top of BASE_IMAGE, and giving it its own cache ref
# would mean a package tag whose contents depend on which BASE_IMAGE a given
# workflow passed.
target "python-base" {
  inherits   = ["_common"]
  dockerfile = "docker/python-base.Dockerfile"
  args = {
    BASE_IMAGE = BASE_IMAGE
  }
}

target "rocm-base" {
  inherits   = ["_common"]
  dockerfile = "docker/rocm-base.Dockerfile"
  args = {
    THEROCK_DEB_INDEX = THEROCK_DEB_INDEX
    SOURCE_DATE       = SOURCE_DATE
  }
  tags = concat(
    ["${pkg("rocm-base")}:latest"],
    DATE != "" ? ["${pkg("rocm-base")}:${DATE}"] : [],
  )
  cache-from = REGISTRY_CACHE == "true" ? ["type=registry,ref=${pkg("rocm-base")}:cache"] : []
  cache-to   = REGISTRY_CACHE == "true" && CACHE_TARGET == "rocm-base" ? ["type=registry,ref=${pkg("rocm-base")}:cache,mode=max"] : []
}

target "rocblas" {
  inherits   = ["_common"]
  dockerfile = "docker/rocblas.Dockerfile"
  contexts   = { python-base = "target:python-base" }
  args = {
    ROCM_ARCH            = ROCM_ARCH
    ROCM_RELEASE         = ROCM_RELEASE
    LEGACY_GCN_ARCHES    = LEGACY_GCN_ARCHES
    BUILD_PARALLEL_LEVEL = BUILD_PARALLEL_LEVEL
  }
  tags       = [image("rocblas")]
  cache-from = cache_from("rocblas")
  cache-to   = cache_to("rocblas")
}

target "migraphx" {
  inherits   = ["_common"]
  dockerfile = "docker/migraphx.Dockerfile"
  contexts   = { rocblas = ctx("rocblas", WITH_ROCBLAS_IMAGE) }
  args = {
    ROCM_ARCH            = ROCM_ARCH
    LEGACY_GCN_ARCHES    = LEGACY_GCN_ARCHES
    BUILD_PARALLEL_LEVEL = BUILD_PARALLEL_LEVEL
    MIGRAPHX_REF         = MIGRAPHX_REF
    # Only bust the develop clone; a pinned ref's own commits do that already.
    SOURCE_DATE = MIGRAPHX_REF == "develop" ? SOURCE_DATE : "pinned"
  }
  tags       = [image("migraphx")]
  cache-from = cache_from("migraphx")
  cache-to   = cache_to("migraphx")
}

target "pytorch" {
  inherits   = ["_common"]
  dockerfile = "docker/pytorch.Dockerfile"
  contexts   = { rocblas = ctx("rocblas", WITH_ROCBLAS_IMAGE) }
  args = {
    ROCM_ARCH            = ROCM_ARCH
    LEGACY_GCN_ARCHES    = LEGACY_GCN_ARCHES
    BUILD_PARALLEL_LEVEL = BUILD_PARALLEL_LEVEL
    PYTORCH_VERSION      = PYTORCH_VERSION
    ROCM_RELEASE         = ROCM_RELEASE
    USE_PREBUILT         = USE_PREBUILT
  }
  tags       = [image("pytorch")]
  cache-from = cache_from("pytorch")
  cache-to   = cache_to("pytorch")
}

target "torchvision" {
  inherits   = ["_common"]
  dockerfile = "docker/torchvision.Dockerfile"
  contexts = {
    rocblas = ctx("rocblas", WITH_ROCBLAS_IMAGE)
    pytorch = ctx("pytorch", WITH_PYTORCH_IMAGE)
  }
  args = {
    ROCM_ARCH            = ROCM_ARCH
    BUILD_PARALLEL_LEVEL = BUILD_PARALLEL_LEVEL
    PYTORCH_VERSION      = PYTORCH_VERSION
    ROCM_RELEASE         = ROCM_RELEASE
    USE_PREBUILT         = USE_PREBUILT
  }
  tags       = [image("torchvision")]
  cache-from = cache_from("torchvision")
  cache-to   = cache_to("torchvision")
}

target "torchaudio" {
  inherits   = ["_common"]
  dockerfile = "docker/torchaudio.Dockerfile"
  contexts = {
    rocblas = ctx("rocblas", WITH_ROCBLAS_IMAGE)
    pytorch = ctx("pytorch", WITH_PYTORCH_IMAGE)
  }
  args = {
    ROCM_ARCH            = ROCM_ARCH
    BUILD_PARALLEL_LEVEL = BUILD_PARALLEL_LEVEL
    PYTORCH_VERSION      = PYTORCH_VERSION
    ROCM_RELEASE         = ROCM_RELEASE
    USE_PREBUILT         = USE_PREBUILT
  }
  tags       = [image("torchaudio")]
  cache-from = cache_from("torchaudio")
  cache-to   = cache_to("torchaudio")
}

target "ort" {
  inherits   = ["_common"]
  dockerfile = "docker/ort.Dockerfile"
  contexts = {
    python-base = "target:python-base"
    migraphx    = ctx("migraphx", WITH_MIGRAPHX_IMAGE)
  }
  args = {
    ROCM_ARCH   = ROCM_ARCH
    ORT_VERSION = ORT_VERSION
  }
  tags       = [image("ort")]
  cache-from = cache_from("ort")
  cache-to   = cache_to("ort")
}

# No cache at all, in either direction. Nothing builds FROM the final image, so
# its cache has no consumer; and every expensive component it needs arrives as a
# prebuilt image, leaving it nothing but image pulls and a few wheel installs --
# cheap to redo on a retry. Exporting anyway costs a second compressed copy of
# every layer unique to the build, written to the same nearly-full runner disk.
# cache-from without a cache-to would only ever 404, hence neither.
target "final" {
  inherits   = ["_common"]
  dockerfile = "docker/final.Dockerfile"
  contexts = {
    python-base = "target:python-base"
    migraphx    = ctx("migraphx", WITH_MIGRAPHX_IMAGE)
    pytorch     = ctx("pytorch", WITH_PYTORCH_IMAGE)
    torchvision = ctx("torchvision", WITH_TORCHVISION_IMAGE)
    torchaudio  = ctx("torchaudio", WITH_TORCHAUDIO_IMAGE)
    ort         = ctx("ort", WITH_ORT_IMAGE)
  }
  args = {
    TORCH_BLAS_PREFER_HIPBLASLT = is_legacy_arch(ROCM_ARCH) ? "0" : "1"
  }
  # A release build gets one immutable "<release>-<arch>" tag -- :latest-<arch>
  # and :<date>-<arch> are meant to move nightly, a release tag is meant to never
  # change under the same name again.
  tags = RELEASE_TAG != "" ? ["${pkg("final")}:${RELEASE_TAG}-${slug(ROCM_ARCH)}"] : concat(
    ["${pkg("final")}:latest-${tag_arch("final")}"],
    DATE != "" ? ["${pkg("final")}:${DATE}-${tag_arch("final")}"] : [],
  )
}
