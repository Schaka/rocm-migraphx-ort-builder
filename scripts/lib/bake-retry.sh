# shellcheck shell=bash
# Sourced, never executed.
#
# `docker buildx bake --push` dies outright on a 429 from ghcr.io: BuildKit's
# lazy blob pull (for the base image and registry cache-from layers) doesn't
# retry on its own. Nightly's ~26-arch matrix starts every job within seconds
# of each other, so they all hit the same rocm-base/component blobs at once --
# a short burst against ghcr.io's per-package rate limiter, not a quota (the
# 429 body's retry-after is sub-second), so a handful of backoff retries
# clears it.
bake_push_with_retry() {
    _attempt=1
    _max_attempts=5
    _delay=5
    while true; do
        if docker buildx bake --push "$@"; then
            return 0
        fi
        if [ "$_attempt" -ge "$_max_attempts" ]; then
            return 1
        fi
        echo "::warning::bake --push $* failed (attempt $_attempt/$_max_attempts), retrying in ${_delay}s" >&2
        sleep "$_delay"
        _attempt=$((_attempt + 1))
        _delay=$((_delay * 2))
    done
}
