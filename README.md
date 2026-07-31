# rocm-migraphx-ort-builder

MIGraphX + ONNX Runtime + PyTorch, combined into one image. Started as a way to
get MIGraphX fixes that hadn't shipped yet against a stable ROCm 7.14, paired
with otherwise-stable Torch/ORT. Grew into two separate things:

- A **true nightly** build: every version (ROCm itself, rocBLAS, MIGraphX,
  PyTorch, TheRock's whole stack) fully parameterized and left to float --
  nobody, including the maintainer, has to hand-pin anything for it to keep
  building. Only ONNX Runtime's version stays fixed for now.
- A **manual release** build, independent of AMD's own release cadence (which
  tends to lag): a `workflow_dispatch`-triggered pipeline with sane defaults
  (ROCm 7.14, PyTorch 2.13) that pins every moving part explicitly, so a
  reproducible build never depends on AMD shipping their own combined image on
  any particular schedule.

  This same pinning also doubles as an escape hatch if an arch ever falls out
  of upstream support the way gfx803 already has (see below): trigger one last
  manual release build pinned to the last ROCm/PyTorch/MIGraphX combination
  that arch still works on, and keep re-publishing that exact combination on
  demand -- no separate Dockerfile fork needed, unlike gfx803, which predates
  this workflow and needed one because ROCm 7 dropped it outright.

Both published to `ghcr.io/<owner>/rocm-migraphx-ort-torch-builder`.

Baking every arch in `ROCM_ARCH` into one image isn't feasible on GitHub-hosted
runners -- AMD's own `rocm/pytorch` image, which only bakes in 5 archs, is
~54GB uncompressed, well past what a hosted runner's disk holds. CI instead
builds and pushes one tag per arch (a matrix over `ROCM_ARCH`'s values), rather
than one fat multi-arch image.

The build is also split by component across separate CI jobs: MIGraphX, PyTorch
and ONNX Runtime each compile from source on their own runner (each with its
own ~6h budget -- one monolithic build of all three overran the hosted-runner
ceiling), and each publishes an intermediate image the next job pulls from
instead of recompiling. MIGraphX and PyTorch are independent and run in
parallel; ONNX Runtime builds against MIGraphX's `/opt/rocm`; a final job
assembles all three. Each component lands in its own package:

- `rocm-builder:latest` / `:<YYYYMMDD>` -- the self-built nightly ROCm base
  (see "Nightly ROCm base image" below). Arch-independent, built once before
  the arch matrix, not part of the per-arch tag scheme below.
- `rocm-rocblas-builder:<arch>` -- ROCm with rocBLAS rebuilt from source. Only
  exists for the arches that need it (`gfx900`, `gfx906`, `gfx90c`, `gfx803`);
  MIGraphX and PyTorch both start from it there instead of each rebuilding
  rocBLAS themselves
- `rocm-migraphx-builder:<arch>` -- from-source MIGraphX + ROCm deps in `/opt/rocm`
- `rocm-migraphx-torch-builder:<arch>` -- PyTorch wheel
- `rocm-migraphx-ort-builder:<arch>` -- ONNX Runtime wheel (built against MIGraphX)
- `rocm-migraphx-ort-torch-builder:latest-<arch>` / `:<YYYYMMDD>-<arch>` -- the
  combined image, all three installed

All but the last two are build plumbing (each an incomplete slice of the
stack); downstream consumers want the combined `rocm-migraphx-ort-torch-builder`.

A **manual release build** (see "Manual release build" below) tags every
component differently: `rocblas`/`migraphx`/`pytorch`/`ort` each publish as
`<arch>-rocm<version>` (e.g. `gfx1201-rocm7.14`) instead of the nightly
scheme's plain `<arch>`, and the combined image publishes as
`rocm<version>-<arch>` (e.g. `rocm7.14-gfx1201`) instead of `latest-<arch>`/
`<YYYYMMDD>-<arch>`. This applies to every component, not just the final
image -- `rocblas` and `pytorch` both genuinely vary by ROCm release (rocblas
is rebuilt from a version-pinned source ref; pytorch's own version/ROCm-release
build-args differ from nightly's), so tagging them the same as nightly's plain
`<arch>` would silently overwrite one build with the other instead of keeping
them as separate, versioned artifacts.

The `gfx803` tags in these packages are the odd ones out: Polaris can't be
enumerated by ROCm 7 at all, so they come from `gfx803/Dockerfile.gfx803` on a
ROCm 6.4.4 base, built by a separate manual workflow. Same tag scheme, different
stack -- see [gfx803/README.md](gfx803/README.md).

Torch is also built from source rather than reusing AMD's `rocm/pytorch`
image: current `rocm/pytorch` tags ship TheRock's pip-packaged ROCm SDK (no
`/opt/rocm`), a different runtime layout than MIGraphX's build expects --
mixing the two would risk conflicting `libamdhip64.so` builds loaded into one
process.

Both CK (composable_kernel) and rocMLIR are built from source and enabled
(matching how AMD's own prebuilt images ship), via MIGraphX's own documented
build tool (`rbuild`) rather than apt packages, which don't line up with what
MIGraphX's CMake actually requires (see the Dockerfile's comments).

## Nightly ROCm versioning

Nightly wheel/deb discovery (`scripts/torch-package-build-decide.sh`) applies no ROCm-major
preference -- it takes whatever AMD's index has newest, regardless of major version. As of
2026-07-28/30, that's ROCm **10.x**: AMD bumped TheRock's `version.json` straight from `7.15`
to `10.0.0` to `10.1.0` (skipping 8 and 9 entirely) -- confirmed via that repo's commit history
and the linked issues ([#6932](https://github.com/ROCm/TheRock/issues/6932),
[#7000](https://github.com/ROCm/TheRock/issues/7000)): *"The next release is targeted to ship
as ROCm 10."* This is a deliberate AMD versioning decision, not a stray/unstable preview build
that happened to sort highest.

Practical risk: 7.x nightlies may stop being published entirely once 10.x fully takes over.
Nightly discovery here always floats to newest regardless, so this shouldn't break the build,
but expect the nightly image to move onto ROCm 10.x without any explicit signal beyond this
note.

## Nightly ROCm base image

`BASE_IMAGE` defaults to `ghcr.io/schaka/rocm-builder:latest`, not AMD's own
`rocm/dev-ubuntu-26.04` -- AMD publishes no rolling/nightly tag for that image
at all, only pinned version releases (confirmed against `repo.radeon.com`'s
apt repo: version numbers, alpha/beta/rc, and a `latest` alias that just means
"newest stable release", nothing rolling). So nightly instead self-builds the
ROCm base from TheRock's own nightly `.deb` feed
(`rocm.nightlies.amd.com/packages-multi-arch/deb`) via the Dockerfile's
`rocm-builder` stage, built and published once (`ghcr.io/<owner>/rocm-builder:latest`
and `:<YYYYMMDD>`) before the per-arch matrix runs, not per-arch -- it's
arch-independent (`amdrocm-hpc-sdk` covers every gfx target in one package).

A manual release build overrides `BASE_IMAGE` back to AMD's own pinned tag
instead (see below) -- this self-built path is nightly-only.

Two things worth knowing if you touch this stage:
- TheRock's `.deb` feed has no `latest` alias either, only dated
  `YYYYMMDD-<run-id>` directories, and the bare index URL serves a stale
  cached snapshot -- a cache-busting query string is required to get the
  real, current listing (see the Dockerfile comment).
- The installed package layout differs from AMD's own image: everything lands
  under `/opt/rocm/core-<major.minor>/` with no top-level convenience
  symlinks, and `amdrocm-hpc-sdk` alone doesn't pull HIP's own dev/cmake
  package (`amdrocm-core-dev`/`amdrocm-runtime-dev`) -- both gaps are worked
  around explicitly in the `rocm-builder` stage; validated end-to-end with a
  real gfx1201 MIGraphX + PyTorch build against it.

## GPU support

Officially supported: **gfx900 and above**, i.e. what AMD lists in the ROCm
supported-GPU matrix. Those are the archs in `ROCM_ARCH`, the ones built
nightly, and the only ones worth filing issues against.

There is also an experimental **Polaris / gfx803** variant (RX 460 through RX
590) built from `gfx803/Dockerfile.gfx803` by a separate manual workflow. It is
a different ROCm major on a different base image rather than another arch in
the matrix, because ROCm 7 removed Polaris support from ROCR-Runtime outright.
It is verified on an RX 470 8GB but still slow by construction. Everything about it --
rationale, versions, packages, caveats -- lives in
**[gfx803/README.md](gfx803/README.md)**; nothing in this README applies to it.

## Using this image

Drop-in `BASE_IMAGE` for anything currently pinned to a `rocm/onnxruntime:*`
tag: onnxruntime (built `--use_rocm --use_migraphx`) and torch both live in a
venv at `/opt/venv` (on `PATH`), and `/opt/rocm` has the from-source MIGraphX
plus its ROCm runtime deps.

The combined image (`rocm-migraphx-ort-torch-builder`) is tagged per-arch:
`:latest-<arch>` (e.g. `:latest-gfx1201`) and `:<YYYYMMDD>-<arch>` for nightly
builds, or `:rocm<version>-<arch>` (e.g. `:rocm7.14-gfx1201`) for a manual
release build -- there is no plain `:latest`, pick the tag matching your GPU's
`ROCM_ARCH` value (and whichever build track you want).

```dockerfile
ARG BASE_IMAGE=ghcr.io/<owner>/rocm-migraphx-ort-torch-builder:latest-gfx1201
FROM ${BASE_IMAGE}
RUN python3 -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

## Build args

- `BASE_IMAGE` (default `ghcr.io/schaka/rocm-builder:latest`) - the ROCm
  base. Defaults to the self-built nightly base (see "Nightly ROCm base
  image" above); the manual release workflow overrides this to AMD's own
  pinned `rocm/dev-ubuntu-26.04:<version>-full` instead. Its native `python3`
  is 3.14, but downstream apps like AudioMuse-AI pin `numpy` to a version that
  only resolves against onnx's deps under 3.12, so every wheel built here and
  the final `/opt/venv` all target a uv-managed Python 3.12 instead of the
  base image's interpreter.
- `ROCM_ARCH` (default `gfx900;gfx90c;gfx906;gfx908;gfx90a;gfx942;gfx950;
  gfx1010;gfx1011;gfx1012;gfx1030;gfx1031;gfx1032;gfx1034;gfx1035;
  gfx1036;gfx1100;gfx1101;gfx1102;gfx1103;gfx1150;gfx1151;gfx1152;gfx1153;
  gfx1200;gfx1201`) - semicolon-separated `GPU_TARGETS`/`CMAKE_HIP_ARCHITECTURES`/
  `PYTORCH_ROCM_ARCH` list, matching the breadth AMD's own published images
  build for. Narrow it to your one GPU for a much faster build (quote it if it
  contains a `;`, e.g. `--build-arg ROCM_ARCH=gfx1201`).
- `ROCM_RELEASE` (default empty, `X.Y` e.g. `7.14`) - pins two things
  together: pytorch/torchvision/torchaudio's prebuilt-wheel discovery, and
  (for `gfx900`/`gfx906`/`gfx90c` only) the ROCm line rocBLAS is rebuilt from
  source against. Empty means both float independently -- wheel discovery
  takes whatever's newest (see "Nightly ROCm versioning" above), rocBLAS
  builds from rocm-libraries' own `develop` branch instead of a pinned
  `therock-<release>` tag. Must match `BASE_IMAGE` when set; the manual
  release workflow derives all three from one `rocm_version` input so they
  can't drift apart.
- `MIGRAPHX_REF` (default `develop`) - git ref to build MIGraphX from. The
  manual release workflow overrides this to `release/rocm-rel-<version>`.
- `USE_PREBUILT_PYTORCH` / `USE_PREBUILT_TORCHVISION` / `USE_PREBUILT_TORCHAUDIO`
  (default `1`) - try AMD's prebuilt wheels first, falling back to a
  from-source build per-package if none match. `0` forces a full from-source
  build; the manual release workflow's `use_prebuilt` input sets all three
  together.
- `ORT_VERSION` (default `v1.28.0`) - onnxruntime git tag. Nightly always uses
  this default (never floated, unlike everything else); the manual release
  workflow can override it explicitly.
- `PYTORCH_VERSION` (default `v2.13.0`) - has two roles depending on
  `ROCM_RELEASE`: in release mode (`ROCM_RELEASE` set) it's an exact pin, both
  for prebuilt-wheel discovery and the from-source fallback branch. In
  nightly mode (`ROCM_RELEASE` empty) it's an *optional* pin -- if set,
  nightly wheel discovery filters to that pytorch version while still
  floating on the newest matching ROCm nightly build; empty floats on
  pytorch's version too. Nightly CI passes this as an explicit empty string
  (not omitted) so it actually floats instead of inheriting this default.
- `BUILD_PARALLEL_LEVEL` (default `auto`) - MIGraphX/rocMLIR and PyTorch build
  parallelism (`CMAKE_BUILD_PARALLEL_LEVEL`/`MAX_JOBS`). `auto` sizes it from
  `MemAvailable` at build time (~4GB/job, capped at `nproc`) since the rocMLIR
  LLVM build needs several GB RSS per job and running as many jobs as `nproc`
  can OOM (or segfault, seen in practice) the host. Pass an explicit integer
  to override, e.g. on a CI runner with known dedicated RAM.

## Local build

```
docker build -t rocm-migraphx-ort-builder .
# or, for one GPU only (much faster):
docker build --build-arg ROCM_ARCH=gfx1201 -t rocm-migraphx-ort-builder .
```

Expect 30-60+ minutes per target architecture: this builds LLVM (for
rocMLIR), composable_kernel, ONNX Runtime, and PyTorch from source across
every architecture in `ROCM_ARCH`.

## Testing the CI workflow

The full nightly matrix builds every arch in `ROCM_ARCH`, unhelpful for
confirming a workflow change actually works. Trigger a single-arch run
instead:

```
gh workflow run nightly.yml -f arch=gfx1201
```

or via the Actions tab -> "Nightly build" -> "Run workflow", filling in the
`arch` input. Leaving it empty runs the full matrix, same as the schedule.
A single-arch run still builds the shared `rocm-builder` base first (once,
not per-arch, see "Nightly ROCm base image" above), then fans out to the
four per-arch component builds (migraphx, pytorch, ort, final) for that arch,
via the reusable `build-pipeline.yml` workflow (which in turn calls
`build-component.yml` per component). The `debug` input opens a detached
tmate SSH session into the runner for the build's duration (manual runs
only) when a component needs live inspection.

On first run each package is created **private** and linked to this repo;
flip each to public in its ghcr package settings if downstream pulls need to
be anonymous. `GITHUB_TOKEN` (with `packages: write`) handles the push -- no
PAT required.

## Manual release build

Trigger via the Actions tab -> "Release build" -> "Run workflow", or:

```
gh workflow run release.yml -f rocm_version=7.14 -f pytorch_version=2.13.0
```

Inputs, all optional with sane defaults:

- `rocm_version` (default `7.14.0`) - ROCm version to pin (`X.Y[.Z]`), must
  match a real `rocm/dev-ubuntu-26.04` tag and `repo.radeon.com/rocm/apt/`
  release. Drives `BASE_IMAGE`, `ROCM_RELEASE`, and (unless `migraphx_ref`
  below overrides it) `MIGRAPHX_REF`.
- `migraphx_ref` (default empty = derive `release/rocm-rel-<rocm_version
  major.minor>`) - git ref to build MIGraphX from, independent of
  `rocm_version` when set explicitly -- e.g. pin ROCm/PyTorch/ORT to 7.14
  while still tracking MIGraphX's `develop` for a fix that hasn't landed on
  the `release/rocm-rel-7.14` branch yet.
- `pytorch_version` (default `2.13.0`) - exact pytorch version to pin.
- `ort_version` (default `v1.28.0`) - onnxruntime git tag.
- `use_prebuilt` (default `true`) - try AMD's prebuilt wheels first for
  pytorch/torchvision/torchaudio, falling back to source per-package if none
  match; `false` forces a full from-source build of all three.
- `arch` (default empty = full matrix minus `gfx803`) - single arch to build.

Unlike nightly, this workflow never touches the self-built `rocm-builder`
base at all -- `BASE_IMAGE` is always the AMD-pinned tag derived from
`rocm_version`, regardless of what nightly's own base currently is. See the
component-tag list near the top of this README for how release-build tags
(`<arch>-rocm<version>` per component, `rocm<version>-<arch>` for the final
image) differ from nightly's.

This is also the escape hatch for a future arch losing upstream support (see
the intro above): pin `rocm_version`/`pytorch_version` to the last
combination that arch still builds on, and keep re-running this workflow with
those same inputs indefinitely.
