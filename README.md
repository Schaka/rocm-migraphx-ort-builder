# rocm-migraphx-ort-builder

MIGraphX + ONNX Runtime + PyTorch, all built from source against a ROCm
release with no prebuilt packages for any of them yet, e.g. `rocm/dev-ubuntu-*`
images ahead of what `rocm/onnxruntime`/`rocm/pytorch` publish. Built nightly,
published to `ghcr.io/<owner>/rocm-migraphx-ort-builder`.

Baking every arch in `ROCM_ARCH` into one image isn't feasible on GitHub-hosted
runners -- AMD's own `rocm/pytorch` image, which only bakes in 5 archs, is
~54GB uncompressed, well past what a hosted runner's disk holds. CI instead
builds and pushes one tag per arch (`docker/build-push-action` matrix over
`ROCM_ARCH`'s values), rather than one fat multi-arch image.

Torch is also built from source rather than reusing AMD's `rocm/pytorch`
image: current `rocm/pytorch` tags ship TheRock's pip-packaged ROCm SDK (no
`/opt/rocm`), a different runtime layout than MIGraphX's build expects --
mixing the two would risk conflicting `libamdhip64.so` builds loaded into one
process.

Both CK (composable_kernel) and rocMLIR are built from source and enabled
(matching how AMD's own prebuilt images ship), via MIGraphX's own documented
build tool (`rbuild`) rather than apt packages, which don't line up with what
MIGraphX's CMake actually requires (see the Dockerfile's comments).

## Using this image

Drop-in `BASE_IMAGE` for anything currently pinned to a `rocm/onnxruntime:*`
tag: onnxruntime (built `--use_rocm --use_migraphx`) and torch both live in a
venv at `/opt/venv` (on `PATH`), and `/opt/rocm` has the from-source MIGraphX
plus its ROCm runtime deps.

Tags are per-arch: `:latest-<arch>` (e.g. `:latest-gfx1201`) and
`:<YYYYMMDD>-<arch>` for pinned builds -- there is no plain `:latest`, pick the
tag matching your GPU's `ROCM_ARCH` value.

```dockerfile
ARG BASE_IMAGE=ghcr.io/<owner>/rocm-migraphx-ort-builder:latest-gfx1201
FROM ${BASE_IMAGE}
RUN python3 -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

## Build args

- `BASE_IMAGE` (default `rocm/dev-ubuntu-24.04:7.14.0-full`) - the ROCm base.
  Pinned to 24.04 (Python 3.12) since downstream apps like AudioMuse-AI pin
  `numpy` to a version that only resolves against onnx's deps under 3.12.
- `ROCM_ARCH` (default `gfx900;gfx906;gfx908;gfx90a;gfx942;gfx1030;gfx1100;
  gfx1101;gfx1102;gfx1150;gfx1151;gfx1200;gfx1201`) - semicolon-separated
  `GPU_TARGETS`/`CMAKE_HIP_ARCHITECTURES`/`PYTORCH_ROCM_ARCH` list, matching
  the breadth AMD's own published images build for. Narrow it to your one GPU
  for a much faster build (quote it if it contains a `;`, e.g.
  `--build-arg ROCM_ARCH=gfx1201`).
- `ORT_VERSION` (default `v1.23.2`) - onnxruntime git tag.
- `PYTORCH_VERSION` (default `v2.12.0`) - pytorch git tag.
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
