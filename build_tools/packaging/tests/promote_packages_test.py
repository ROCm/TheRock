#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Manual / on-demand test suite for the promote_packages promotion script.

This exercises promote_packages.py end-to-end against **live multi-arch**
release-candidate (RC) artifacts published to
https://rocm.prereleases.amd.com/. Promotion strips the RC suffix from version
strings (e.g. 7.14.0rc1 -> 7.14.0) and, for multi-arch indices, prunes the
per-gfx device wheels down to a requested set of architectures.

It is intentionally a standalone, on-demand script (run by a human before a
release), NOT a pytest/CI target: it pulls large artifacts over the network
from the prerelease index and builds virtualenvs, which is too heavy and too
coupled to live infrastructure for per-PR CI. Keep it runnable by hand.

WHAT IS VALIDATED
  1. Full promotion of the arch-agnostic aggregator packages
     (rocm, rocm_sdk_core, rocm_sdk_devel, rocm_sdk_libraries, torch,
     torchvision, torchaudio, triton) succeeds and installs.
  2. Multi-arch promotion with an explicit keep-list keeps only the requested
     per-gfx device wheels (amd_torch_device_gfx*, amd_torchvision_device_gfx*,
     rocm_sdk_device_gfx*) and deletes the rest, while still stripping the RC
     suffix everywhere.
  3. JAX wheels (jax_rocm7_plugin, jax_rocm7_pjrt) promote correctly: the RC
     suffix is stripped from the `+rocm<ver>` local segment and the wheels stay
     structurally installable (installed --no-deps, since jax/jaxlib live on
     PyPI). Note jaxlib itself is NOT published multi-arch (see ROCm/TheRock#6158).
  4. Partial promotions (only ROCm, or only torch) do NOT produce a coherent,
     installable, single-version set (expected to FAIL).
  5. Promoted filenames drop the RC suffix and every wheel reports the final
     version.

The expected promoted filenames are DERIVED from whatever was actually fetched
(by replacing the RC version string with the final version), so the test does
not carry brittle hardcoded per-arch package lists that rot every release.

PREREQUISITES
  pip install -r ./build_tools/packaging/requirements.txt

USAGE
  # Full run against the latest RC auto-discovered from the multi-arch index:
  python ./build_tools/packaging/tests/promote_packages_test.py

  # Pin a version and cache downloads between runs:
  python ./build_tools/packaging/tests/promote_packages_test.py \
      --version 7.14.0rc1 --cache-dir /tmp/promote_cache

  # Choose which gfx arch(es) to keep for the multi-arch keep-list scenario:
  python ./build_tools/packaging/tests/promote_packages_test.py \
      --keep-arch gfx942 --extra-arch gfx1201

OPEN QUESTIONS (tracked in ROCm/TheRock#6266 follow-up; resolve before merge)
  - Version baseline: auto-discover latest RC (current default) vs. pin a known
    RC per release branch? Auto-discover keeps the test current but makes runs
    non-deterministic across days.
  - Should the run assert on a fixed arch set, or accept whatever the index
    currently publishes for the chosen gfx targets?
  - therock-dist-* multi-arch tarball: include it in the promotion set once the
    tarball-multi-arch/ listing exposes a stable, enumerable name (its directory
    listing is JS-rendered today).
"""

import argparse
import os
import platform as platform_module
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from packaging.version import Version
from pkginfo import Wheel

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import promote_packages

sys.path.insert(0, os.fspath(Path(__file__).parent.parent.parent))
import setup_venv

PRERELEASE_BASE = "https://rocm.prereleases.amd.com"
WHL_INDEX = f"{PRERELEASE_BASE}/whl-multi-arch"

# Arch-agnostic packages: always promoted, never pruned by the keep-list. Their
# per-package index page links back to files hosted at the WHL_INDEX root.
AGGREGATOR_PKG_DIRS = [
    "rocm",
    "rocm-sdk-core",
    "rocm-sdk-devel",
    "rocm-sdk-libraries",
    "torch",
    "torchvision",
    "torchaudio",
    "triton",
]

# Per-gfx device wheels: subject to keep-list pruning during multi-arch
# promotion. Directory name is f"{prefix}-{arch}", e.g. amd-torch-device-gfx942.
DEVICE_PKG_PREFIXES = [
    "amd-torch-device",
    "amd-torchvision-device",
    "rocm-sdk-device",
]

# JAX wheels are arch-agnostic and Linux-only. They carry a `manylinux_*` tag
# (not `linux_x86_64`) and are promoted via the local-version path (the RC
# suffix is stripped from the `+rocm<ver>` segment). jax_rocm7_pjrt is
# py3-none; jax_rocm7_plugin is cpXX-specific.
JAX_PKG_DIRS = [
    "jax-rocm7-plugin",
    "jax-rocm7-pjrt",
]
_MANYLINUX_X86 = re.compile(r"manylinux_[\d_]+x86_64")

PLATFORM_TAGS = {"linux": "linux_x86_64", "windows": "win_amd64"}

# Matches the cpXX-cpXX ABI tag pair present in every Python-tag-specific wheel
# (torch-family aggregators AND per-gfx device wheels). SDK wheels are py3-none
# and do not match, so they are never filtered by python tag.
_CP_TAG_RE = re.compile(r"-(cp\d+)-cp\d+-")


def _http_get(url: str) -> str:
    with urllib.request.urlopen(url, timeout=60) as resp:
        return resp.read().decode("utf-8", errors="replace")


def list_index_files(pkg_dir: str) -> list[str]:
    """Return the real (decoded) filenames listed under a package index dir.

    The prerelease index links to files as `../<file>` (hosted at the index
    root) and URL-encodes `+` as `%2B`; both are normalized here.
    """
    url = f"{WHL_INDEX}/{pkg_dir}/"
    try:
        html = _http_get(url)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return []
        raise
    names = set()
    for href in re.findall(r'href="([^"]+)"', html):
        name = urllib.parse.unquote(href).rsplit("/", 1)[-1]
        if name.endswith(".whl") or name.endswith(".tar.gz"):
            names.add(name)
    return sorted(names)


def discover_latest_rc() -> Version:
    """Find the newest `X.Y.ZrcN` published for the `rocm` sdist."""
    candidates = []
    for name in list_index_files("rocm"):
        m = re.search(r"rocm-(\d+\.\d+\.\d+rc\d+)\.tar\.gz", name)
        if m:
            candidates.append(Version(m.group(1)))
    if not candidates:
        raise RuntimeError(
            f"Could not discover any rocm RC version under {WHL_INDEX}/rocm/"
        )
    return max(candidates)


def _cp_tags(pkg_dir: str, version: str, platform_tag: str) -> set[str]:
    tags = set()
    for name in list_index_files(pkg_dir):
        if version in name and platform_tag in name:
            m = _CP_TAG_RE.search(name)
            if m:
                tags.add(m.group(1))
    return tags


def detect_py_tag(version: str, platform_tag: str, arches: list[str]) -> str:
    """Pick the highest cpXX tag present across torch AND the device dirs.

    The newest tag published for the arch-agnostic `torch` wheel (e.g. cp314)
    is not always built for every per-gfx device wheel, so intersect the
    availability to guarantee a coherent, installable set.
    """
    dirs = ["torch"]
    for arch in arches:
        dirs += [f"amd-torch-device-{arch}", f"amd-torchvision-device-{arch}"]
    tag_sets = [_cp_tags(d, version, platform_tag) for d in dirs]
    tag_sets = [s for s in tag_sets if s]
    if not tag_sets:
        raise RuntimeError(f"No torch wheel found for {version} / {platform_tag}")
    common = set.intersection(*tag_sets) if len(tag_sets) > 1 else tag_sets[0]
    tags = common or tag_sets[0]
    # cp313 > cp312 > ... by trailing integer.
    return max(tags, key=lambda t: int(t[2:]))


def _upstream_version(name: str) -> Version:
    """Extract the upstream version field from a wheel/sdist filename.

    torch-2.11.0+rocm7.14.0rc1-...  -> 2.11.0
    rocm_sdk_core-7.14.0rc1-...     -> 7.14.0rc1
    rocm-7.14.0rc1.tar.gz           -> 7.14.0rc1
    """
    field = name.split("-", 2)[1] if "-" in name else name
    field = field.split("+", 1)[0]
    if field.endswith(".tar.gz"):
        field = field[: -len(".tar.gz")]
    return Version(field)


def _keep_max_upstream(names: list[str]) -> list[str]:
    """A single package dir can publish several upstream versions for one RC
    (e.g. torch 2.10.0 and 2.11.0 both `+rocm7.14.0rc1`). Keep only the newest
    so the promoted set stays single-version and installable."""
    if len(names) <= 1:
        return names
    try:
        newest = max(_upstream_version(n) for n in names)
    except Exception:
        return names
    return [n for n in names if _upstream_version(n) == newest]


def _matches(name: str, version: str, platform_tag: str, py_tag: str) -> bool:
    if version not in name:
        return False
    # rocm-<ver>.tar.gz sdist has no platform tag.
    if name.endswith(".tar.gz"):
        return True
    if platform_tag not in name:
        return False
    # Python-tag-specific wheels (torch-family + device wheels) must match the
    # single chosen cpXX tag; py3-none SDK wheels have no cp tag and pass.
    if _CP_TAG_RE.search(name):
        return f"-{py_tag}-" in name
    return True


def select_packages(
    version: str, platform_tag: str, py_tag: str, arches: list[str]
) -> tuple[list[tuple[str, str]], set[str]]:
    """Return (download list, device-wheel filenames) for the given arches.

    download list is [(base_url, filename), ...]; every file is hosted at the
    WHL_INDEX root, so base_url is the same for all of them.
    """
    base = f"{WHL_INDEX}/"
    downloads: list[tuple[str, str]] = []
    seen: set[str] = set()

    def add(pkg_dir: str) -> list[str]:
        matched = [
            name
            for name in list_index_files(pkg_dir)
            if name not in seen and _matches(name, version, platform_tag, py_tag)
        ]
        picked = _keep_max_upstream(matched)
        for name in picked:
            seen.add(name)
            downloads.append((base, name))
        return picked

    for pkg_dir in AGGREGATOR_PKG_DIRS:
        add(pkg_dir)

    device_files: set[str] = set()
    for prefix in DEVICE_PKG_PREFIXES:
        for arch in arches:
            for name in add(f"{prefix}-{arch}"):
                device_files.add(name)

    return downloads, device_files


def select_jax_packages(
    version: str, py_tag: str
) -> tuple[list[tuple[str, str]], set[str]]:
    """Return (download list, jax filenames) for the Linux JAX wheels.

    JAX uses `manylinux_*` tags rather than `linux_x86_64`, and the index can
    carry two upstream versions per RC on different manylinux baselines (e.g.
    0.10.0 on manylinux_2_27 and 0.9.1 on manylinux_2_28); keep only the newest
    upstream so the promoted set is single-version. jax_rocm7_plugin is cpXX
    specific and must match the chosen py_tag; jax_rocm7_pjrt is py3-none.
    """
    base = f"{WHL_INDEX}/"
    downloads: list[tuple[str, str]] = []
    jax_files: set[str] = set()
    for pkg_dir in JAX_PKG_DIRS:
        matched = [
            name
            for name in list_index_files(pkg_dir)
            if version in name
            and _MANYLINUX_X86.search(name)
            and (not _CP_TAG_RE.search(name) or f"-{py_tag}-" in name)
        ]
        for name in _keep_max_upstream(matched):
            downloads.append((base, name))
            jax_files.add(name)
    return downloads, jax_files


def promoted_name(rc_name: str, version: str, final_version: str) -> str:
    return rc_name.replace(version, final_version)


def checkPromotedFileNames(dir_path: Path, expected: set[str]) -> tuple[bool, str]:
    actual = {p.name for p in dir_path.glob("*") if p.name != ".venv"}
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        return (
            False,
            f"Promoted file set mismatch.\n  missing: {missing}\n  unexpected: {unexpected}",
        )
    return True, ""


def checkAllWheelsSameVersion(
    dir_path: Path, expected_version: Version
) -> tuple[bool, str]:
    for file in dir_path.glob("*.whl"):
        wheel = Wheel(file)
        version = Version(wheel.version)

        local_tag = "rocm" + str(expected_version)
        if str(version) == str(expected_version) and version.local is None:
            continue  # arch-agnostic / device SDK wheels
        # torch-family wheels carry the rocm tag as a local version segment,
        # sometimes prefixed with a git hash (e.g. `git43422b04.rocm7.14.0`).
        elif version.local is not None and version.local.endswith(local_tag):
            continue
        else:
            return (
                False,
                f"{file.name} has version {version}, but expected version is {expected_version}",
            )

    return True, ""


def checkInstallation(dir_path: Path, no_deps: bool = False) -> tuple[bool, str]:
    """Note: dir_path must be a TemporaryDirectory, otherwise clean up .venv yourself.

    no_deps installs the wheels without resolving dependencies -- used for the
    JAX wheels, whose runtime deps (jax, jaxlib) live on PyPI and are not part
    of the promoted set; we only need to confirm the promoted wheels are
    structurally installable.
    """
    try:
        setup_venv.create_venv(dir_path / ".venv")
        python_exe = setup_venv.find_venv_python(dir_path / ".venv")
        if python_exe is None:
            return (
                False,
                "Problem when installing temporary venv: Python executable not found",
            )

        # Install wheels/sdists, not the therock-dist tarball or the venv itself.
        packages = [
            p
            for p in dir_path.glob("*")
            if p.name != ".venv" and "therock-dist" not in p.name
        ]

        cmd = [python_exe, "-m", "pip", "install"]
        if no_deps:
            cmd.append("--no-deps")
        subprocess.run(
            cmd + packages,
            capture_output=True,
            encoding="utf-8",
            check=True,
        )
    except subprocess.CalledProcessError as e:
        return False, e.stderr
    return True, ""


def _stage_inputs(src: Path, dst: Path, names: set[str]) -> None:
    for file in src.glob("*"):
        if file.is_file() and file.name in names:
            shutil.copy2(file, dst)


def _run_checks(
    tmp_dir: Path, checks: list[tuple[str, tuple[bool, str]]], expect_success: bool
) -> bool:
    """Run named checks. Returns whether the overall expectation held."""
    for func_name, res in checks:
        if expect_success and not res[0]:
            print(f"\n[ERROR] {func_name} failed:\n{res[1]}")
            return False
        if not expect_success and res[0]:
            print(
                f"\n[ERROR] {func_name} succeeded but the promotion should NOT have been coherent."
            )
            proc = subprocess.run(
                ["ls", tmp_dir], capture_output=True, encoding="utf-8"
            )
            print(proc.stdout)
            return False
    return True


def _banner(msg: str) -> None:
    line = "=" * 81
    print(f"\n{line}\n{msg}\n{line}")


def checkPromoteEverything(
    dir_path: Path,
    expected_version: Version,
    input_files: set[str],
    expected_files: set[str],
) -> bool:
    _banner("TEST: promotion of all aggregator packages (should SUCCEED)")
    with tempfile.TemporaryDirectory(prefix="PromoteTest-Everything-") as tmp:
        tmp_dir = Path(tmp)
        _stage_inputs(dir_path, tmp_dir, input_files)
        promote_packages.main(tmp_dir, delete=True)
        ok = _run_checks(
            tmp_dir,
            [
                (
                    "checkPromotedFileNames",
                    checkPromotedFileNames(tmp_dir, expected_files),
                ),
                (
                    "checkAllWheelsSameVersion",
                    checkAllWheelsSameVersion(tmp_dir, expected_version),
                ),
                ("checkInstallation", checkInstallation(tmp_dir)),
            ],
            expect_success=True,
        )
    _banner(
        "TEST DONE: promote everything. Result: " + ("SUCCESS" if ok else "FAILURE")
    )
    return ok


def checkPromoteMultiArch(
    dir_path: Path,
    expected_version: Version,
    keep_arch: str,
    input_files: set[str],
    dropped_promoted_names: set[str],
    expected_files: set[str],
) -> bool:
    _banner(f"TEST: multi-arch promotion keeping only {keep_arch} (should SUCCEED)")
    with tempfile.TemporaryDirectory(prefix="PromoteTest-MultiArch-") as tmp:
        tmp_dir = Path(tmp)
        _stage_inputs(dir_path, tmp_dir, input_files)
        promote_packages.main(tmp_dir, delete=True, multi_arch_targets=[keep_arch])

        checks = [
            ("checkPromotedFileNames", checkPromotedFileNames(tmp_dir, expected_files)),
            (
                "checkAllWheelsSameVersion",
                checkAllWheelsSameVersion(tmp_dir, expected_version),
            ),
            ("checkInstallation", checkInstallation(tmp_dir)),
        ]
        ok = _run_checks(tmp_dir, checks, expect_success=True)

        # Explicitly confirm the non-kept arch device wheels were pruned.
        if ok:
            leftover = dropped_promoted_names & {p.name for p in tmp_dir.glob("*")}
            if leftover:
                print(
                    f"\n[ERROR] non-kept-arch device wheels survived: {sorted(leftover)}"
                )
                ok = False
    _banner(
        "TEST DONE: multi-arch promotion. Result: " + ("SUCCESS" if ok else "FAILURE")
    )
    return ok


def checkPromoteJax(
    dir_path: Path,
    expected_version: Version,
    input_files: set[str],
    expected_files: set[str],
) -> bool:
    _banner("TEST: promotion of JAX wheels (should SUCCEED)")
    if not input_files:
        print("[SKIP] no JAX wheels found for this version; skipping.")
        _banner("TEST DONE: promote JAX. Result: SKIPPED")
        return True
    with tempfile.TemporaryDirectory(prefix="PromoteTest-Jax-") as tmp:
        tmp_dir = Path(tmp)
        _stage_inputs(dir_path, tmp_dir, input_files)
        promote_packages.main(tmp_dir, delete=True)
        ok = _run_checks(
            tmp_dir,
            [
                (
                    "checkPromotedFileNames",
                    checkPromotedFileNames(tmp_dir, expected_files),
                ),
                (
                    "checkAllWheelsSameVersion",
                    checkAllWheelsSameVersion(tmp_dir, expected_version),
                ),
                # JAX runtime deps (jax, jaxlib) are on PyPI, not in the promoted
                # set -- install --no-deps to validate the wheels structurally.
                ("checkInstallation", checkInstallation(tmp_dir, no_deps=True)),
            ],
            expect_success=True,
        )
    _banner("TEST DONE: promote JAX. Result: " + ("SUCCESS" if ok else "FAILURE"))
    return ok


def checkPartialPromotion(
    dir_path: Path,
    expected_version: Version,
    input_files: set[str],
    expected_files: set[str],
    match_files: str,
    label: str,
) -> bool:
    _banner(f"TEST: promotion of only {label} packages (should FAIL)")
    with tempfile.TemporaryDirectory(prefix=f"PromoteTest-Only-{label}-") as tmp:
        tmp_dir = Path(tmp)
        _stage_inputs(dir_path, tmp_dir, input_files)
        promote_packages.main(tmp_dir, match_files=match_files, delete=True)
        ok = _run_checks(
            tmp_dir,
            [
                (
                    "checkPromotedFileNames",
                    checkPromotedFileNames(tmp_dir, expected_files),
                ),
                (
                    "checkAllWheelsSameVersion",
                    checkAllWheelsSameVersion(tmp_dir, expected_version),
                ),
                ("checkInstallation", checkInstallation(tmp_dir)),
            ],
            expect_success=False,
        )
    _banner(
        f"TEST DONE: promote only {label}. Result: " + ("SUCCESS" if ok else "FAILURE")
    )
    return ok


def fetchPackage(
    base_url: str, package_name: str, tmp_dir: Path, cache_dir: Path | None
) -> None:
    if cache_dir is not None and (cache_dir / package_name).exists():
        print(f"  Found in cache: {package_name}")
        shutil.copy2(cache_dir / package_name, tmp_dir / package_name)
        return
    print(f"  Downloading {package_name}")
    # Safe-encode: the "+" in torch local versions confuses curl otherwise.
    url = base_url + urllib.parse.quote(package_name)
    subprocess.run(
        ["curl", "-fSL", "--output", tmp_dir / package_name, url], check=True
    )
    if cache_dir is not None:
        shutil.copy2(tmp_dir / package_name, cache_dir / package_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="On-demand test of RC->final multi-arch package promotion."
    )
    parser.add_argument(
        "--platform",
        choices=sorted(PLATFORM_TAGS),
        default="linux" if platform_module.system() != "Windows" else "windows",
        help="Target platform tag to fetch (default: auto-detected).",
    )
    parser.add_argument(
        "--version",
        default=None,
        help="RC version to test (e.g. 7.14.0rc1). Default: auto-discover latest.",
    )
    parser.add_argument(
        "--keep-arch",
        default="gfx942",
        help="gfx arch to KEEP in the multi-arch keep-list scenario.",
    )
    parser.add_argument(
        "--extra-arch",
        default="gfx1201",
        help="Additional gfx arch to fetch but expect to be pruned.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Directory to cache downloaded packages between runs.",
    )
    p = parser.parse_args(sys.argv[1:])

    platform = p.platform
    platform_tag = PLATFORM_TAGS[platform]
    cache_dir = p.cache_dir
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    rc_version = Version(p.version) if p.version else discover_latest_rc()
    rc_str = str(rc_version)
    fin_str = rc_str.split("rc")[0]
    final_version = Version(fin_str)

    keep_arch = p.keep_arch
    extra_arch = p.extra_arch
    arches = [keep_arch, extra_arch]
    py_tag = detect_py_tag(rc_str, platform_tag, arches)

    print(
        f"Testing promotion {rc_version} -> {final_version} on {platform} "
        f"({platform_tag}, {py_tag}); arches fetched: {arches}, kept: {keep_arch}"
    )

    downloads, device_files = select_packages(rc_str, platform_tag, py_tag, arches)
    # JAX is Linux-only (manylinux wheels); skip on Windows.
    jax_downloads: list[tuple[str, str]] = []
    jax_files: set[str] = set()
    if platform == "linux":
        jax_downloads, jax_files = select_jax_packages(rc_str, py_tag)
    # A device wheel belongs to keep_arch iff its filename carries the exact
    # `device_<arch>-` token (trailing `-` avoids gfx1201 matching gfx1200).
    kept_device_files = {n for n in device_files if f"device_{keep_arch}-" in n}
    dropped_device_files = device_files - kept_device_files
    if not dropped_device_files:
        print(
            f"[WARN] no device wheels found for --extra-arch {extra_arch}; "
            "the multi-arch prune assertion will be a no-op."
        )

    with tempfile.TemporaryDirectory(prefix=f"PromoteTest-{platform}-") as tmp:
        tmp_dir = Path(tmp)
        all_downloads = downloads + jax_downloads
        print(f"Fetching {len(all_downloads)} packages (cache: {cache_dir}) ...")
        for base_url, name in all_downloads:
            fetchPackage(base_url, name, tmp_dir, cache_dir)
        print(" ...done")

        all_rc_files = {n for _, n in downloads}
        aggregator_files = all_rc_files - device_files
        multi_arch_input = aggregator_files | device_files
        expected_aggregators = {
            promoted_name(n, rc_str, fin_str) for n in aggregator_files
        }
        # Multi-arch keep-list run: aggregators + kept-arch device wheels survive.
        expected_multi_arch = {
            promoted_name(n, rc_str, fin_str)
            for n in (multi_arch_input - dropped_device_files)
        }
        dropped_promoted_names = {
            promoted_name(n, rc_str, fin_str) for n in dropped_device_files
        }
        expected_jax = {promoted_name(n, rc_str, fin_str) for n in jax_files}

        res_everything = checkPromoteEverything(
            tmp_dir, final_version, aggregator_files, expected_aggregators
        )
        res_multi = checkPromoteMultiArch(
            tmp_dir,
            final_version,
            keep_arch,
            multi_arch_input,
            dropped_promoted_names,
            expected_multi_arch,
        )
        res_only_rocm = checkPartialPromotion(
            tmp_dir,
            final_version,
            aggregator_files,
            expected_aggregators,
            "rocm*",
            "rocm",
        )
        res_only_torch = checkPartialPromotion(
            tmp_dir,
            final_version,
            aggregator_files,
            expected_aggregators,
            "*torch*",
            "torch",
        )
        res_jax = checkPromoteJax(tmp_dir, final_version, jax_files, expected_jax)

        _banner("SUMMARY")
        print(f"checkPromoteEverything:  {'SUCCESS' if res_everything else 'FAILURE'}")
        print(f"checkPromoteMultiArch:   {'SUCCESS' if res_multi else 'FAILURE'}")
        print(f"checkPromoteJax:         {'SUCCESS' if res_jax else 'FAILURE'}")
        print(f"checkPromoteOnlyRocm:    {'SUCCESS' if res_only_rocm else 'FAILURE'}")
        print(f"checkPromoteOnlyTorch:   {'SUCCESS' if res_only_torch else 'FAILURE'}")
        print("=" * 81)

        if not all([res_everything, res_multi, res_jax, res_only_rocm, res_only_torch]):
            sys.exit(1)
