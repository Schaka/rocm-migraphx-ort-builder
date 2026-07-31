# rocm-migraphx-ort-builder

MIGraphX + ONNX Runtime + PyTorch, all built from source against a ROCm
release with no prebuilt packages for any of them yet, e.g. `rocm/dev-ubuntu-*`
images ahead of what `rocm/onnxruntime`/`rocm/pytorch` publish. Built nightly,
the combined image published to
`ghcr.io/<owner>/rocm-migraphx-ort-torch-builder`.

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

- `rocm-rocblas-builder:<arch>` -- ROCm with rocBLAS rebuilt from source. Only
  exists for the arches that need it (`gfx900`, `gfx906`, `gfx90c`, `gfx803`);
  MIGraphX and PyTorch both start from it there instead of each rebuilding
  rocBLAS themselves
- `rocm-migraphx-builder:<arch>` -- from-source MIGraphX + ROCm deps in `/opt/rocm`
- `rocm-migraphx-torch-builder:<arch>` -- PyTorch wheel
- `rocm-migraphx-ort-builder:<arch>` -- ONNX Runtime wheel (built against MIGraphX)
- `rocm-migraphx-ort-torch-builder:latest-<arch>` / `:<YYYYMMDD>-<arch>` -- the
  combined image, all three installed

All but the last are build plumbing (each an incomplete slice of the stack);
downstream consumers want the combined `rocm-migraphx-ort-torch-builder`.

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
`:latest-<arch>` (e.g. `:latest-gfx1201`) and `:<YYYYMMDD>-<arch>` for pinned
builds -- there is no plain `:latest`, pick the tag matching your GPU's
`ROCM_ARCH` value.

```dockerfile
ARG BASE_IMAGE=ghcr.io/<owner>/rocm-migraphx-ort-torch-builder:latest-gfx1201
FROM ${BASE_IMAGE}
RUN python3 -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

## Build args

- `BASE_IMAGE` (default `rocm/dev-ubuntu-26.04:7.14.0-full`) - the ROCm base.
  Its native `python3` is 3.14, but downstream apps like AudioMuse-AI pin
  `numpy` to a version that only resolves against onnx's deps under 3.12, so
  every wheel built here and the final `/opt/venv` all target a uv-managed
  Python 3.12 instead of the base image's interpreter.
- `ROCM_ARCH` (default `gfx900;gfx90c;gfx906;gfx908;gfx90a;gfx942;gfx950;
  gfx1010;gfx1011;gfx1012;gfx1030;gfx1031;gfx1032;gfx1033;gfx1034;gfx1035;
  gfx1036;gfx1100;gfx1101;gfx1102;gfx1103;gfx1150;gfx1151;gfx1152;gfx1153;
  gfx1200;gfx1201`) - semicolon-separated `GPU_TARGETS`/`CMAKE_HIP_ARCHITECTURES`/
  `PYTORCH_ROCM_ARCH` list, matching the breadth AMD's own published images
  build for. Narrow it to your one GPU for a much faster build (quote it if it
  contains a `;`, e.g. `--build-arg ROCM_ARCH=gfx1201`).
- `ORT_VERSION` (default `v1.28.0`) - onnxruntime git tag.
- `PYTORCH_VERSION` (default `v2.13.0`) - pytorch git tag.
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

The full nightly matrix builds all 13 archs, unhelpful for confirming a
workflow change actually works. Trigger a single-arch run instead:

```
gh workflow run nightly.yml -f arch=gfx1201
```

or via the Actions tab -> "Nightly build" -> "Run workflow", filling in the
`arch` input. Leaving it empty runs the full matrix, same as the schedule.
Even a single-arch run fans out to the four component builds (migraphx, pytorch,
ort, final) for that arch, via the reusable `build-pipeline.yml` workflow (which
in turn calls `build-component.yml` per component). The `debug` input
opens a detached tmate SSH session into the runner for the build's duration
(manual runs only) when a component needs live inspection.

On first run each of the four packages is created **private** and linked to
this repo; flip each to public in its ghcr package settings if downstream pulls
need to be anonymous. `GITHUB_TOKEN` (with `packages: write`) handles the
push -- no PAT required.
