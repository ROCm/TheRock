#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Manual / on-demand test suite for the promote_packages promotion script.

This exercises promote_packages.py end-to-end against a **local directory of
already-downloaded** pre-release artifacts. It does NOT fetch anything over the
network. Point --input-dir at a download_python_packages.py output root: wheels
+ the rocm sdist are read from `<root>/wheels/` and distribution tarballs from
`<root>/tarball-multi-arch/` (either subdir may be absent). Wheels and tarballs
are never read from the same directory.

A real download mixes both OSes and several package versions in `wheels/`; the
input is filtered to one --platform (default linux) and, for the install checks,
to one coherent stack per family (see below), since e.g. multiple torch versions
can't be installed together.

Promotion rewrites the prerelease segment of the version rather than only
stripping "rc": promote_packages.py can strip it (rc/a -> release, e.g.
7.14.0rc1 -> 7.14.0) or replace it (a -> rc, rc -> a). For a multi-arch input it
additionally prunes the per-gfx device wheels down to a keep-list of
architectures; when only the arch set is trimmed (no prerelease-type change) the
version string is left unchanged. These tests drive the common rc/a -> release
path.

It is intentionally a standalone, on-demand script (run by a human before a
release), NOT a pytest/CI target: it installs the wheels into throwaway
virtualenvs. Keep it runnable by hand.

WHAT IS VALIDATED
  1. Full promotion of the arch-agnostic aggregator packages
     (rocm, rocm_sdk_core, rocm_sdk_devel, rocm_sdk_libraries, torch,
     torchvision, torchaudio, triton) succeeds and installs.
  2. Multi-arch promotion with a keep-list keeps only ONE arch's per-gfx device
     wheels (amd_torch_device_gfx*, amd_torchvision_device_gfx*,
     rocm_sdk_device_gfx*) and prunes the rest, while still stripping the RC
     suffix everywhere. The kept arch and the pruned arch(es) are auto-detected
     from whatever device wheels are present.
  3. JAX wheels (jax_rocm7_plugin, jax_rocm7_pjrt) promote correctly: the RC
     suffix is stripped from the `+rocm<ver>` local segment and the wheels stay
     structurally installable (installed --no-deps, since jax/jaxlib live on
     PyPI). Note jaxlib itself is NOT published multi-arch (see ROCm/TheRock#6158).
  4. Partial promotions (only ROCm, or only torch) do NOT produce a coherent,
     installable, single-version set (expected to FAIL).
  5. Every therock-dist tarball present (per-gfx AND multiarch) is promoted by
     its filename rc->final rename (promote_targz_tarball renames by filename,
     not by repacking).
  6. Promoted filenames drop the RC suffix and every wheel reports the final
     version.

The source version, the target platform, and the set of gfx architectures are
all DERIVED from the files in the input directory. Nothing is hard-coded to a
specific release. How the expected output is worked out:

  1. Read the source version from the package itself -- the rocm_sdk_core wheel
     (or the rocm sdist) via pkginfo, e.g. "7.9.0rc20260501".
  2. Drop the prerelease tail to get the final version, e.g. "7.9.0".
  3. For every discovered filename, swap the source version for the final
     version -- the same rename promotion does -- and assert the promoted
     directory contains exactly that set. For example:
       rocm_sdk_core-7.9.0rc20260501-...whl -> rocm_sdk_core-7.9.0-...whl
       torch-2.7.1+rocm7.9.0rc20260501-...whl -> torch-2.7.1+rocm7.9.0-...whl

Coherent install stack: promote_packages works on any subset of wheels, so where
`wheels/` carries several versions of a distribution (torch 2.9..2.14, jax
0.9/0.10) the install checks use only the newest of each, with triton pinned to
whatever the chosen torch requires (`Requires-Dist: triton==<ver>`). A download
also ships every supported CPython (cp310..cp314) while the install checks run
under just one interpreter, so version-specific wheels are further narrowed to
that interpreter's tag (Python-agnostic files -- the rocm sdist, `py3-none-*` SDK
wheels -- always qualify). Promotion is Python-agnostic and behaves identically
for the versions/CPythons left out.

PREREQUISITES
  pip install -r ./build_tools/packaging/requirements.txt

USAGE
  # Promote + verify a download_python_packages.py output tree:
  python ./build_tools/packaging/tests/promote_packages_test.py --input-dir ./download

  # Force a platform (otherwise defaults to linux when both are present):
  python ./build_tools/packaging/tests/promote_packages_test.py \
      --input-dir ./download --platform linux
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from packaging.version import Version
from pkginfo import SDist, Wheel

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import promote_packages

sys.path.insert(0, os.fspath(Path(__file__).parent.parent.parent))
import setup_venv

PLATFORMS = ("linux", "windows")

# Arch-agnostic aggregator wheels: always promoted, never pruned by the
# keep-list. Matched by filename prefix (device wheels use the distinct
# `*_device_gfx<N>` form and are matched separately).
AGGREGATOR_WHEEL_PREFIXES = (
    "rocm_sdk_core-",
    "rocm_sdk_devel-",
    "rocm_sdk_libraries-",
    "torch-",
    "torchvision-",
    "torchaudio-",
    "triton-",
)

# Per-gfx device wheel marker, e.g. `rocm_sdk_device_gfx942-...` /
# `amd_torch_device_gfx1201-...`. Reuses promote_packages._GFX_ARCH (the same
# arch-token shape the promotion script keys off) so the two never drift; the
# trailing `-` anchors the token so `gfx11` can't match `gfx1153`.
_DEVICE_ARCH_RE = re.compile(rf"device_({promote_packages._GFX_ARCH})-")


def _is_aggregator(name: str) -> bool:
    if name.startswith(AGGREGATOR_WHEEL_PREFIXES):
        return name.endswith(".whl")
    # rocm-<digit>... is the meta wheel / sdist; rocm-bootstrap / rocm-profiler
    # start with a letter after `rocm-` and are intentionally excluded.
    if re.match(r"rocm-\d", name):
        return name.endswith(".whl") or name.endswith(".tar.gz")
    return False


def _device_arch(name: str) -> str | None:
    m = _DEVICE_ARCH_RE.search(name)
    return m.group(1) if m else None


def _wheel_platform(name: str) -> str | None:
    """The platform a wheel targets, from its filename tag, or None if it is
    platform-agnostic (the rocm sdist, any `-none-any` wheel). A download's
    `wheels/` dir carries both OSes, so this is used to pick one."""
    if "win_amd64" in name:
        return "windows"
    if "linux" in name:  # linux_x86_64, manylinux_2_XX_x86_64, ...
        return "linux"
    return None


# CPython tag of the interpreter running this script (e.g. "cp312"); the install
# checks build a venv from this interpreter, so only wheels for this CPython (or
# Python-agnostic ones) can be installed.
_RUNNER_PYTAG = f"cp{sys.version_info.major}{sys.version_info.minor}"


def _wheel_pytag(name: str) -> str | None:
    """The CPython tag a wheel is built for (e.g. "cp312"), or None when the
    wheel is Python-version-agnostic (`py3-none-*`, the rocm sdist)."""
    m = re.search(r"-(cp3\d+)-", name)
    return m.group(1) if m else None


def _py_installable(name: str) -> bool:
    """Whether `name` can be installed under the interpreter running the tests.

    A real download ships every supported CPython (cp310..cp314); the install
    checks run under exactly one, so version-specific wheels for other CPythons
    must be dropped -- pip rejects them with "not a supported wheel on this
    platform". Python-agnostic files (rocm sdist, `py3-none-*` SDK wheels) always
    qualify. Promotion itself is Python-agnostic, so this only narrows the
    install/expectation sets, never what promotion is exercised against."""
    tag = _wheel_pytag(name)
    return tag is None or tag == _RUNNER_PYTAG


def _list_files(directory: Path) -> set[str]:
    """Filenames in `directory`, or an empty set if it doesn't exist."""
    if not directory.is_dir():
        return set()
    return {p.name for p in directory.iterdir() if p.is_file()}


def _torch_triton_pin(wheel_path: Path) -> str | None:
    """The exact triton version torch pins (`Requires-Dist: triton==<ver>`).

    triton is singled out (vs torch / torchvision / torchaudio) because torch
    depends on it with an *exact* `==` pin to one specific triton build, while a
    download can carry several triton builds for one RC (e.g. 3.7.0+gitXXXX,
    3.7.1+gitYYYY). The torch family wheels are not cross-pinned to each other
    like this, so only triton needs to be filtered down to torch's chosen build
    to keep the staged set installable. Read it from torch's own metadata rather
    than assuming newest.
    """
    with zipfile.ZipFile(wheel_path) as z:
        meta = next(n for n in z.namelist() if n.endswith(".dist-info/METADATA"))
        text = z.read(meta).decode("utf-8", errors="replace")
    m = re.search(r"^Requires-Dist:\s*triton==([^\s;]+)", text, re.MULTILINE)
    return m.group(1) if m else None


def detect_source_version(input_dir: Path, names: set[str]) -> str:
    """Read the source (pre-promotion) version from package metadata.

    Prefers the rocm_sdk_core wheel, falling back to the rocm sdist, and reads
    the version out of the package metadata via pkginfo rather than parsing the
    filename -- that also implicitly validates the filename encodes the same
    version its metadata claims.
    """
    for name in sorted(names):
        if name.startswith("rocm_sdk_core-") and name.endswith(".whl"):
            return Wheel(input_dir / name).version
    for name in sorted(names):
        if re.match(r"rocm-\d", name) and name.endswith(".tar.gz"):
            return SDist(input_dir / name).version
    raise RuntimeError(
        "Could not determine the source version from the input directory: expected "
        "a rocm_sdk_core-<ver>-*.whl or a rocm-<ver>.tar.gz sdist to be present."
    )


class InputSet:
    """Classification of an input directory into promotion groups.

    Given a directory of downloaded artifacts, this splits the filenames into the
    groups each promotion scenario needs and derives everything else from them:

      source_str / source_version  the pre-promotion version (rc/a), read from
                                    rocm_sdk_core / rocm sdist metadata
      final_version / final_str    that version with the prerelease segment
                                    stripped (the rc/a -> release target)
      src_version_type             "rc" or "a", passed straight to promote_packages
      platform                     linux / windows (forced, or inferred from tags)
      aggregator_files             arch-agnostic wheels + rocm sdist, minus any
                                    triton build torch does not pin
      device_files                 per-gfx device wheels (`*_device_gfx<N>-`)
      jax_files                    wheels with "jax" in the name (installed
                                    --no-deps)
      tarball_files                therock-dist-*.tar.gz distribution tarballs
      present_archs / keep_arch    the gfx archs seen and which one multi-arch
                                    promotion keeps (the rest are pruned)
    """

    def __init__(self, input_dir: Path, platform: str | None) -> None:
        # download_python_packages.py emits wheels + the rocm sdist under
        # `wheels/` and distribution tarballs under `tarball-multi-arch/` -- never
        # mixed. Read each from its own subdir; either may be absent.
        self.wheels_dir = input_dir / "wheels"
        self.tarball_dir = input_dir / "tarball-multi-arch"

        names = _list_files(self.wheels_dir)
        if not names:
            raise RuntimeError(f"No wheels found in {self.wheels_dir}")

        self.source_str = detect_source_version(self.wheels_dir, names)
        self.source_version = Version(self.source_str)
        self.final_version = Version(self.source_version.base_version)
        self.final_str = str(self.final_version)
        # rc vs a comes from the parsed prerelease kind, not a substring test
        # (an alpha input like 7.13.0a20260501 must classify as "a").
        pre = self.source_version.pre
        self.src_version_type = pre[0] if pre else "rc"

        # `wheels/` holds both linux and windows wheels, and they can't install
        # together -- pick one platform and keep only its wheels plus the
        # platform-agnostic files (rocm sdist, `-none-any`). Default to linux
        # unless only windows is present.
        present_platforms = {pl for n in names if (pl := _wheel_platform(n))}
        self.platform = platform or (
            "linux"
            if "linux" in present_platforms
            else next(iter(present_platforms), "linux")
        )
        names = {n for n in names if _wheel_platform(n) in (self.platform, None)}

        # Device wheels feed the multi-arch scenario, which both promotes (prunes
        # non-kept archs) and installs the kept arch. They are narrowed to the
        # runner's CPython (like the aggregator install set) and collapsed to the
        # newest version per package: a download ships several torch/torchvision
        # versions per arch (e.g. amd_torch_device_gfx1010 2.10/2.11/2.12) whose
        # device wheels cannot co-install, and the newest ones line up with the
        # newest torch/torchvision picked for the aggregator install stack.
        # Pruning is still fully exercised across every arch within that CPython.
        self.device_files = self._newest_per_package(
            {n for n in names if _device_arch(n) and _py_installable(n)}
        )
        self.jax_files = {n for n in names if "jax" in n}
        self.aggregator_files = {
            n for n in names if _is_aggregator(n) and n not in self.device_files
        }
        # Keep only the triton build(s) torch pins; drop any other triton wheels.
        self.aggregator_files -= self._non_pinned_triton()

        # therock-dist tarballs come from their own dir, never the wheels dir.
        self.tarball_files = {
            n
            for n in _list_files(self.tarball_dir)
            if n.startswith("therock-dist")
            and n.endswith(".tar.gz")
            and self.source_str in n
        }

        self.present_archs = sorted(
            {a for n in self.device_files if (a := _device_arch(n))}
        )
        # Auto-detect: keep the first present arch, everything else is pruned.
        self.keep_arch = self.present_archs[0] if self.present_archs else None
        self.dropped_device_files = {
            n for n in self.device_files if _device_arch(n) != self.keep_arch
        }

        # A download ships several mutually-exclusive torch/jax versions that
        # can't co-install. promote_packages works on any subset, so the install
        # scenarios use one coherent stack per family (newest of each package,
        # with triton pinned to the chosen torch); promotion behaves identically
        # for the versions left out.
        self.install_set = self._coherent_install_set()
        self.jax_install_set = self._newest_per_package(self.jax_files)

    def _non_pinned_triton(self) -> set[str]:
        """Triton wheels not pinned by any torch wheel in the set.

        Every torch wheel is consulted (a directory can hold more than one), and
        a triton wheel is kept if *any* torch pins it. If torch wheels are present
        but none carry an exact `triton==` pin we raise rather than silently
        staging every triton build (which would make installs incoherent).
        """
        torch_wheels = [n for n in self.aggregator_files if n.startswith("torch-")]
        triton_wheels = {n for n in self.aggregator_files if n.startswith("triton-")}
        if not torch_wheels or not triton_wheels:
            return set()
        pins = {
            pin for n in torch_wheels if (pin := _torch_triton_pin(self.wheels_dir / n))
        }
        if not pins:
            raise RuntimeError(
                f"torch wheel(s) present but none pin an exact triton build "
                f"('Requires-Dist: triton==<ver>') in {self.wheels_dir}; refusing "
                f"to stage triton wheels {sorted(triton_wheels)} blindly."
            )
        matched = {n for n in triton_wheels if any(pin in n for pin in pins)}
        if not matched:
            raise RuntimeError(
                f"torch pins triton=={sorted(pins)} but no matching triton wheel "
                f"is present in {self.wheels_dir} (found: {sorted(triton_wheels)})"
            )
        return triton_wheels - matched

    def _dist_version(self, name: str) -> Version:
        path = self.wheels_dir / name
        meta = SDist(path) if name.endswith(".tar.gz") else Wheel(path)
        return Version(meta.version)

    def _newest_per_package(self, names: set[str]) -> set[str]:
        """Keep only the highest version of each distribution, restricted to
        wheels installable under the running interpreter. A download can ship
        several torch / jax versions (that cannot be installed together) across
        several CPythons (only one of which matches this interpreter)."""
        newest: dict[str, str] = {}
        for n in names:
            if not _py_installable(n):
                continue
            pkg = n.split("-", 1)[0]  # wheel/sdist distribution name (no hyphens)
            if pkg not in newest or self._dist_version(n) > self._dist_version(
                newest[pkg]
            ):
                newest[pkg] = n
        return set(newest.values())

    def _coherent_install_set(self) -> set[str]:
        """One installable stack from the aggregator wheels: the newest of each
        distribution, with triton pinned to whatever the chosen torch requires
        (not merely the newest triton build)."""
        picks = self._newest_per_package(self.aggregator_files)
        torch = next((n for n in picks if n.startswith("torch-")), None)
        if torch is not None:
            pin = _torch_triton_pin(self.wheels_dir / torch)
            picks = {n for n in picks if not n.startswith("triton-")}
            if pin:
                picks |= {
                    n
                    for n in self.aggregator_files
                    if n.startswith("triton-") and pin in n and _py_installable(n)
                }
        return picks

    def promoted(self, names: set[str]) -> set[str]:
        return {promoted_name(n, self.source_str, self.final_str) for n in names}


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
        python_exe = setup_venv.find_venv_python_exe(dir_path / ".venv")
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
            # Cross-platform directory dump (the old `ls` shell-out broke on
            # Windows and masked the real failure).
            for entry in sorted(p.name for p in tmp_dir.glob("*")):
                print(f"    {entry}")
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
    src_version_type: str = "rc",
) -> bool:
    _banner("TEST: promotion of all aggregator packages (should SUCCEED)")
    with tempfile.TemporaryDirectory(prefix="PromoteTest-Everything-") as tmp:
        tmp_dir = Path(tmp)
        _stage_inputs(dir_path, tmp_dir, input_files)
        promote_packages.main(tmp_dir, delete=True, src_version_type=src_version_type)
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
    src_version_type: str = "rc",
) -> bool:
    """Only meaningful for MULTI-ARCH packages: it exercises the keep-list pass
    (`--multi-arch-targets`), which keeps `keep_arch`'s per-gfx device wheels and
    prunes the rest. It needs >=2 gfx archs with device wheels present to have
    anything to prune; single-arch inputs skip this scenario (see __main__)."""
    _banner(f"TEST: multi-arch promotion keeping only {keep_arch} (should SUCCEED)")
    with tempfile.TemporaryDirectory(prefix="PromoteTest-MultiArch-") as tmp:
        tmp_dir = Path(tmp)
        _stage_inputs(dir_path, tmp_dir, input_files)
        promote_packages.main(
            tmp_dir,
            delete=True,
            multi_arch_targets=[keep_arch],
            src_version_type=src_version_type,
        )

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
    src_version_type: str = "rc",
) -> bool:
    _banner("TEST: promotion of JAX wheels (should SUCCEED)")
    if not input_files:
        print("[SKIP] no JAX wheels found for this version; skipping.")
        _banner("TEST DONE: promote JAX. Result: SKIPPED")
        return True
    with tempfile.TemporaryDirectory(prefix="PromoteTest-Jax-") as tmp:
        tmp_dir = Path(tmp)
        _stage_inputs(dir_path, tmp_dir, input_files)
        promote_packages.main(tmp_dir, delete=True, src_version_type=src_version_type)
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


def checkPromoteTarball(
    dir_path: Path,
    tarball_names: set[str],
    version: str,
    final_version: str,
    src_version_type: str = "rc",
) -> bool:
    """Every therock-dist tarball present (per-gfx AND multiarch) is promoted by
    a filename-based rc->final rename (promote_targz_tarball), not by repacking.

    Names are discovered in the separate `tarball-multi-arch/` dir and the
    tarballs are staged and promoted from there; the expected promoted name is
    the discovered name with the rc->final rename applied.
    """
    _banner(
        "TEST: promotion of therock-dist tarballs (per-gfx + multiarch, should SUCCEED)"
    )
    if not tarball_names:
        print("[SKIP] no therock-dist tarballs found in tarball-multi-arch/; skipping.")
        _banner("TEST DONE: promote tarball. Result: SKIPPED")
        return True

    print(f"  discovered {len(tarball_names)} tarball(s):")
    for n in sorted(tarball_names):
        print(f"    {n}")

    ok = True
    with tempfile.TemporaryDirectory(prefix="PromoteTest-Tarball-") as tmp:
        tmp_dir = Path(tmp)
        _stage_inputs(dir_path, tmp_dir, tarball_names)
        promote_packages.main(tmp_dir, delete=True, src_version_type=src_version_type)
        produced = {p.name for p in tmp_dir.glob("*")}
        for rc_name in sorted(tarball_names):
            fin_name = promoted_name(rc_name, version, final_version)
            if fin_name not in produced:
                print(
                    f"\n[ERROR] expected {fin_name} after promotion; got {sorted(produced)}"
                )
                ok = False
            elif rc_name in produced:
                print(f"\n[ERROR] rc tarball {rc_name} survived promotion")
                ok = False
    _banner("TEST DONE: promote tarball. Result: " + ("SUCCESS" if ok else "FAILURE"))
    return ok


def checkPartialPromotion(
    dir_path: Path,
    expected_version: Version,
    input_files: set[str],
    expected_files: set[str],
    match_files: str,
    label: str,
    src_version_type: str = "rc",
) -> bool:
    _banner(f"TEST: promotion of only {label} packages (should FAIL)")
    with tempfile.TemporaryDirectory(prefix=f"PromoteTest-Only-{label}-") as tmp:
        tmp_dir = Path(tmp)
        _stage_inputs(dir_path, tmp_dir, input_files)
        promote_packages.main(
            tmp_dir,
            match_files=match_files,
            delete=True,
            src_version_type=src_version_type,
        )
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


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="On-demand test of pre-release->final package promotion over a local directory."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="download_python_packages.py output root containing wheels/ "
        "and/or tarball-multi-arch/ to promote and test.",
    )
    parser.add_argument(
        "--platform",
        choices=PLATFORMS,
        default=None,
        help="Target platform (default: linux when both are present in wheels/).",
    )
    args = parser.parse_args(argv)
    if not args.input_dir.is_dir():
        parser.error(f"--input-dir {args.input_dir} is not a directory")
    return args


if __name__ == "__main__":
    p = parse_args(sys.argv[1:])

    inputs = InputSet(p.input_dir, p.platform)

    print(
        f"Testing promotion {inputs.source_version} -> {inputs.final_version} on "
        f"{inputs.platform} from {inputs.wheels_dir}"
    )
    print(
        f"  archs present: {inputs.present_archs or '(none)'}; kept: {inputs.keep_arch}"
    )
    print(
        f"  aggregators: {len(inputs.aggregator_files)} "
        f"(install stack: {len(inputs.install_set)}), device: {len(inputs.device_files)}, "
        f"jax: {len(inputs.jax_files)} (install: {len(inputs.jax_install_set)}), "
        f"tarballs: {len(inputs.tarball_files)}"
    )

    multi_arch_input = inputs.install_set | inputs.device_files
    expected_aggregators = inputs.promoted(inputs.install_set)
    expected_multi_arch = inputs.promoted(
        multi_arch_input - inputs.dropped_device_files
    )
    dropped_promoted_names = inputs.promoted(inputs.dropped_device_files)
    expected_jax = inputs.promoted(inputs.jax_install_set)

    src = inputs.src_version_type

    res_everything = checkPromoteEverything(
        inputs.wheels_dir,
        inputs.final_version,
        inputs.install_set,
        expected_aggregators,
        src,
    )

    # The multi-arch keep-list pass only does anything for multi-arch packages:
    # it needs >=2 gfx archs with per-gfx device wheels to have something to
    # prune, so single-arch inputs skip it.
    if inputs.keep_arch is not None and inputs.dropped_device_files:
        res_multi = checkPromoteMultiArch(
            inputs.wheels_dir,
            inputs.final_version,
            inputs.keep_arch,
            multi_arch_input,
            dropped_promoted_names,
            expected_multi_arch,
            src,
        )
    else:
        _banner(
            "TEST: multi-arch promotion — SKIPPED "
            "(need >=2 gfx archs with device wheels to exercise pruning)"
        )
        res_multi = True

    res_only_rocm = checkPartialPromotion(
        inputs.wheels_dir,
        inputs.final_version,
        inputs.install_set,
        expected_aggregators,
        "rocm*",
        "rocm",
        src,
    )
    res_only_torch = checkPartialPromotion(
        inputs.wheels_dir,
        inputs.final_version,
        inputs.install_set,
        expected_aggregators,
        "*torch*",
        "torch",
        src,
    )
    res_jax = checkPromoteJax(
        inputs.wheels_dir,
        inputs.final_version,
        inputs.jax_install_set,
        expected_jax,
        src,
    )
    res_tarball = checkPromoteTarball(
        inputs.tarball_dir,
        inputs.tarball_files,
        inputs.source_str,
        inputs.final_str,
        src,
    )

    _banner("SUMMARY")
    print(f"checkPromoteEverything:  {'SUCCESS' if res_everything else 'FAILURE'}")
    print(f"checkPromoteMultiArch:   {'SUCCESS' if res_multi else 'FAILURE'}")
    print(f"checkPromoteJax:         {'SUCCESS' if res_jax else 'FAILURE'}")
    print(f"checkPromoteTarball:     {'SUCCESS' if res_tarball else 'FAILURE'}")
    print(f"checkPromoteOnlyRocm:    {'SUCCESS' if res_only_rocm else 'FAILURE'}")
    print(f"checkPromoteOnlyTorch:   {'SUCCESS' if res_only_torch else 'FAILURE'}")
    print("=" * 81)

    if not all(
        [res_everything, res_multi, res_jax, res_tarball, res_only_rocm, res_only_torch]
    ):
        sys.exit(1)
