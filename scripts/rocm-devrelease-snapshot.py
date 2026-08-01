#!/usr/bin/env python3
"""Tier-2 PyTorch prebuilt-wheel fallback -- only reached after the real stable-release index
(repo.amd.com/rocm/whl-multi-arch) has no build for the exact pinned PyTorch/ROCm version.
Resolves ONE consistent AMD devreleases snapshot across every ROCm-versioned package PyTorch
needs, rather than letting each package pick its own newest build independently.

Why a single snapshot, not per-package best-effort: a devreleases wheel's own dependency
metadata always declares abstract pins like "rocm-sdk-device-<arch>==7.14.0" regardless of
which build it actually is -- devreleases never actually has a matching build for that exact
pin (confirmed: no clean, non-tagged release of rocm-sdk-device-<arch>, the rocm metapackage,
rocm-sdk-core, or rocm-sdk-libraries exists on this line at all). The metadata is useless for
correlating builds across packages.

Two kinds of coordinated build identifier DO correlate builds across packages, both confirmed
empirically across torch, the per-arch device-SDK package, the rocm metapackage,
rocm-sdk-core, rocm-sdk-libraries, and triton:
  - "aYYYYMMDD" -- AMD's daily nightly build tag, embedded the same way in every package's own
    filename (".../rocm<major>.<minor>.0a<date>-..."). Preferred when available: directly
    sortable, so "newest" has an unambiguous meaning.
  - ".dev0+<git-hash>" -- a per-CI-run build identifier (confirmed: the SAME hash appears
    across torch, rocm, rocm-sdk-core, rocm-sdk-libraries, rocm-sdk-device-<arch>, and triton
    for a given run -- it's a shared TheRock/rocm-libraries orchestration-commit stamp, not
    each package's own separate upstream repo commit, despite looking like one). No inherent
    ordering, so every candidate is tried until one has the full closure available, rather than
    picking a "newest" one.
"rcN" tagged builds are excluded: confirmed they only ever exist for the top-level
torch-device wheel, never for its own sub-dependencies, so they can never form a resolvable
set.

Also accepts "<pytorch_ver_num>a0" (PyTorch's own pre-release alpha tag, cut before the real
"<pytorch_ver_num>" was tagged upstream) as a match for a plain "<pytorch_ver_num>" pin --
intentional: these pre-release builds are what's actually available on a ROCm line AMD moved
on from before the final PyTorch version was cut, and are treated as close enough to use.

Tried FIRST, before any snapshot matching: if the "clean"-tagged top-level wheel (the exact
"<pytorch_ver>+rocm<release>.0" name a plain `pip download` would ask for) actually exists,
each of its sub-dependencies is resolved independently -- exact clean match if available,
else the newest build on the SAME major.minor line, picked with no cross-package consistency
requirement at all. Cheaper and simpler than full snapshot matching, and safe specifically
because the top-level wheel itself is anchored to a real, already-known-good build -- only
reached for a package whose exact pin genuinely doesn't exist as a clean release.

Usage: rocm-devrelease-snapshot.py <dest_dir> <python_tag> <platform_tag> <arch> <pytorch_ver_num> <rocm_release>
Exit 0 on success (files downloaded into dest_dir). Exit 1 (nothing downloaded) if no snapshot
has every required package available -- caller should fall back to a full PyTorch source build.
"""
import re
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

DEVRELEASES = "https://rocm.devreleases.amd.com/whl-multi-arch/"

# Ordinary PyPI packages PyTorch's ROCm build always needs, with no ROCm/snapshot coupling at
# all -- any recent version works, downloaded unpinned.
PLAIN_DEPS = ["filelock", "typing_extensions", "sympy", "networkx", "jinja2", "fsspec",
              "mpmath", "MarkupSafe", "setuptools", "rocm-bootstrap"]


def log(*a):
    print("[rocm-devrelease-snapshot]", *a, file=sys.stderr)


def list_dir(pkg_name):
    url = DEVRELEASES + pkg_name + "/"
    try:
        data = urllib.request.urlopen(url, timeout=30).read().decode(errors="replace")
    except Exception as e:
        log(f"WARN: could not list {url}: {e}")
        return []
    return [urllib.parse.unquote(h) for h in re.findall(r'href="([^"]+)"', data)]


def find_at_tag(pkg_name, tag_substr, py_tag, plat_tag):
    """Return the wheel/sdist filename for pkg_name containing tag_substr (a date tag like
    "rocm7.14.0a20260624", or a bare 40-hex-char build hash), for our python/platform, or
    None. The hash form deliberately ignores the "rocm..." vs "devrocm..." prefix that differs
    between torch-family packages and the rest -- the hash itself is a unique-enough anchor.

    py_tag is optional on a match: rocm-sdk-core/-libraries/-device-<arch> and the rocm
    metapackage ship as pure-Python "py3-none-*" wheels (or a plain sdist), with no
    cp312-specific build at all -- only torch/amd-torch-device-<arch>/triton are actually
    Python-version-specific."""
    for href in list_dir(pkg_name):
        fn = href.rsplit("/", 1)[-1]
        if tag_substr not in fn:
            continue
        if fn.endswith(".whl"):
            if plat_tag not in fn and "any" not in fn:
                continue
            if py_tag not in fn and "py3" not in fn and "py2.py3" not in fn:
                continue
        elif not fn.endswith(".tar.gz"):
            continue
        return fn
    return None


def find_exact(pkg_name, tag_substr, py_tag, plat_tag):
    """Like find_at_tag, but tag_substr must immediately precede the platform/python suffix
    (or .tar.gz) -- i.e. an exact clean-tagged match, not merely containing the substring."""
    for href in list_dir(pkg_name):
        fn = href.rsplit("/", 1)[-1]
        if fn.endswith(".tar.gz"):
            if fn == f"{pkg_name.replace('-', '_')}-{tag_substr}.tar.gz":
                return fn
            continue
        if not fn.endswith(".whl"):
            continue
        if plat_tag not in fn and "any" not in fn:
            continue
        if py_tag not in fn and "py3" not in fn and "py2.py3" not in fn:
            continue
        if fn.split("-")[1] == tag_substr:
            return fn
    return None


def pick_newest_same_line(pkg_name, major_minor, py_tag, plat_tag, mode="prefix"):
    """Newest available build (any suffix -- date, dev0+hash, rcN) on the given major.minor
    line, ranked by PEP 440 precedence (packaging.version already orders dev < aN < rcN <
    final the way this ecosystem's tags actually behave).

    mode="prefix" (rocm/rocm-sdk-*/the rocm metapackage): the package's OWN version field IS
    the ROCm release, so major.minor must be a prefix of it.
    mode="contains" (triton): the package's own version field is ITS OWN numbering (e.g.
    "3.8.0+git<hash>.rocm7.14.0") with the ROCm release embedded as "rocm<major.minor>."
    somewhere inside it, not as the version's own prefix."""
    from packaging.version import InvalidVersion
    from packaging.version import parse as parse_version

    best = None
    best_ver = None
    for href in list_dir(pkg_name):
        fn = href.rsplit("/", 1)[-1]
        if fn.endswith(".tar.gz"):
            ver = fn[len(pkg_name.replace("-", "_")) + 1 : -len(".tar.gz")]
        elif fn.endswith(".whl"):
            if plat_tag not in fn and "any" not in fn:
                continue
            if py_tag not in fn and "py3" not in fn and "py2.py3" not in fn:
                continue
            parts = fn[: -len(".whl")].split("-")
            if len(parts) < 5:
                continue
            ver = parts[1]
        else:
            continue
        if mode == "prefix":
            if not ver.startswith(major_minor + "."):
                continue
        elif f"rocm{major_minor}." not in ver:
            continue
        try:
            parsed = parse_version(ver)
        except InvalidVersion:
            continue
        if best_ver is None or parsed > best_ver:
            best_ver = parsed
            best = fn
    return best


def try_clean_with_independent_substitution(anchored_pkgs, device_pkg, arch, pytorch_ver_num,
                                              rocm_release, py_tag, plat_tag):
    clean_top = find_exact(device_pkg, f"{pytorch_ver_num}+rocm{rocm_release}.0", py_tag, plat_tag)
    if clean_top is None:
        log(f"no clean release wheel for {device_pkg} at pytorch={pytorch_ver_num} rocm={rocm_release} -- skipping independent substitution")
        return None

    log(f"clean release wheel found for {device_pkg}: {clean_top} -- resolving each dependency independently")
    files = {device_pkg: clean_top}
    for pkg in anchored_pkgs:
        if pkg == device_pkg:
            continue
        clean_ver = f"{pytorch_ver_num}+rocm{rocm_release}.0" if pkg == "torch" else f"{rocm_release}.0"
        fn = find_exact(pkg, clean_ver, py_tag, plat_tag)
        if fn is not None:
            files[pkg] = fn
            continue
        mode = "contains" if pkg == "triton" else "prefix"
        fn = pick_newest_same_line(pkg, rocm_release, py_tag, plat_tag, mode=mode)
        if fn is None:
            log(f"{pkg} has no build at all on the {rocm_release}.x line -- independent substitution can't complete")
            return None
        log(f"{pkg}: no clean {clean_ver} build, substituting newest same-line build instead: {fn}")
        files[pkg] = fn
    return files


def download_all(files, dest):
    """Download every {pkg: filename} entry plus PLAIN_DEPS. Returns True on full success.
    On ANY failure, cleans up whatever partial files landed in dest and returns False rather
    than raising -- the caller (and ultimately the Dockerfile) must treat this exactly like
    "no match found", not crash the build."""
    downloaded = []
    try:
        for fn in files.values():
            # The per-package listing page's own hrefs are "../<filename>" -- relative links
            # that resolve to the flat DEVRELEASES root, not nested under the package name.
            url = DEVRELEASES + urllib.parse.quote(fn)
            log(f"downloading: {fn}")
            out = dest / fn
            subprocess.run(["curl", "-sSf", "-o", str(out), url], check=True)
            downloaded.append(out)
        for pkg in PLAIN_DEPS:
            log(f"downloading (unpinned, --no-deps): {pkg}")
            subprocess.run(
                ["pip", "download", "--no-deps", "--index-url", DEVRELEASES, "-d", str(dest), pkg],
                check=True,
            )
        return True
    except subprocess.CalledProcessError as e:
        log(f"download failed ({e}) -- a listed file likely vanished from devreleases before we could fetch it; cleaning up and giving up on this candidate")
        for out in downloaded:
            out.unlink(missing_ok=True)
        return False


def main():
    dest_dir, py_tag, plat_tag, arch, pytorch_ver_num, rocm_release = sys.argv[1:7]
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    device_pkg = f"amd-torch-device-{arch}"
    # Every package that must agree on the same snapshot. amd-torch-device-<arch>/torch/triton
    # also carry the pytorch/triton version in their own filename; the rest don't.
    anchored_pkgs = [device_pkg, "torch", "rocm", "rocm-sdk-core", "rocm-sdk-libraries",
                      f"rocm-sdk-device-{arch}", "triton"]

    chosen_files = try_clean_with_independent_substitution(
        anchored_pkgs, device_pkg, arch, pytorch_ver_num, rocm_release, py_tag, plat_tag
    )
    if chosen_files is not None:
        log("using clean top-level release + independent per-package substitution")
        if download_all(chosen_files, dest):
            log(f"resolved {len(chosen_files) + len(PLAIN_DEPS)} package(s)")
            return 0
        log("clean-release substitution failed at download time -- falling back to full-closure snapshot matching")

    log("falling back to full-closure snapshot matching")
    ver_prefix = f"(?:{re.escape(pytorch_ver_num)}|{re.escape(pytorch_ver_num)}a0)"
    date_re = re.compile(rf"{ver_prefix}\+rocm{re.escape(rocm_release)}\.0a(\d{{8}})")
    hash_re = re.compile(rf"{ver_prefix}\+devrocm{re.escape(rocm_release)}\.0\.dev0\.([0-9a-f]{{40}})")

    # Candidates: every date/hash the top-level device package itself has a cp312/linux_x86_64
    # build for, at our pinned pytorch version (or its "a0" pre-release). No point considering
    # one where even the top-level package doesn't exist. Dates are tried first (newest first,
    # unambiguous ordering); hashes have no inherent order, so every one is tried in listing
    # order until the full closure resolves.
    date_candidates = set()
    hash_candidates = set()
    for href in list_dir(device_pkg):
        fn = urllib.parse.unquote(href.rsplit("/", 1)[-1])
        if py_tag not in fn or plat_tag not in fn:
            continue
        m = date_re.search(fn)
        if m:
            date_candidates.add(m.group(1))
            continue
        m = hash_re.search(fn)
        if m:
            hash_candidates.add(m.group(1))

    candidates = [("date", d, f"rocm{rocm_release}.0a{d}") for d in sorted(date_candidates, reverse=True)]
    candidates += [("hash", h, h) for h in sorted(hash_candidates)]

    if not candidates:
        log(f"no {device_pkg} nightly/dev build exists at all for pytorch={pytorch_ver_num} rocm={rocm_release}")
        return 1

    log(f"{len(candidates)} candidate snapshot(s) for {device_pkg}: "
        f"{len(date_candidates)} date-tagged, {len(hash_candidates)} hash-tagged")

    # Try every candidate in order, not just the first that lists as complete: a candidate can
    # still fail at actual download time (the same live-pruning issue download_all guards
    # against), in which case the next candidate deserves a shot rather than giving up outright.
    for kind, snap_id, tag_substr in candidates:
        files = {}
        ok = True
        for pkg in anchored_pkgs:
            fn = find_at_tag(pkg, tag_substr, py_tag, plat_tag)
            if fn is None:
                ok = False
                log(f"{kind} {snap_id}: {pkg} has no matching build, rejecting this snapshot")
                break
            files[pkg] = fn
        if not ok:
            continue
        log(f"using devreleases snapshot: {kind} {snap_id}")
        if download_all(files, dest):
            log(f"resolved {len(files) + len(PLAIN_DEPS)} package(s) at snapshot {kind} {snap_id}")
            return 0
        log(f"{kind} {snap_id} failed at download time, trying the next candidate")

    log(f"none of {len(candidates)} candidate snapshot(s) actually resolved -- giving up")
    return 1


if __name__ == "__main__":
    sys.exit(main())
