# shellcheck shell=sh
# Sourced, never executed.
#
# Parallel-job sizing shared by every memory-hungry compile stage (rocBLAS's
# Tensile step, rocMLIR/LLVM under MIGraphX, PyTorch itself).
#
# Each -O3 clang job doing IPO/codegen on LLVM's own sources needs several GB
# RSS. Default ninja parallelism (= nproc) can exceed available RAM on the build
# host, which surfaces as clang segfaulting on a different random file each time
# (or the OOM killer taking out unrelated host processes) rather than a clean
# failure -- so cap the job count so peak memory stays bounded.
#
# BUILD_PARALLEL_LEVEL=auto (the default) sizes it from MemAvailable at ~4GB per
# job, capped at nproc. Pass an explicit count instead when the host's RAM is
# known and dedicated, e.g. --build-arg BUILD_PARALLEL_LEVEL=8.

resolve_build_jobs() {
    _jobs="${BUILD_PARALLEL_LEVEL:-auto}"
    if [ "$_jobs" = "auto" ]; then
        _jobs=$(awk '/MemAvailable/{printf "%d", $2/1024/1024/4}' /proc/meminfo)
        _cpu=$(nproc)
        [ "$_jobs" -gt "$_cpu" ] && _jobs=$_cpu
        [ "$_jobs" -lt 1 ] && _jobs=1
    fi
    echo "$_jobs"
}
