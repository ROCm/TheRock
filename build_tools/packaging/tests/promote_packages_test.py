#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Manual / on-demand test suite for the promote_packages promotion script.

This exercises promote_packages.py end-to-end against a **local directory of
already-downloaded** pre-release artifacts. It does not download packages
itself — point --input-dir at a download_python_packages.py output root: wheels
+ the rocm sdist are read from `<root>/wheels/` and distribution tarballs from
`<root>/tarball-multi-arch/`. Wheels and tarballs are never read from the same
directory. The install checks call `pip install` without `--no-index`, so PyPI
may still be reached for transitive deps (torch pulls sympy, networkx, jinja2,
... which are not part of a download); JAX wheels install with `--no-deps`.

It is intentionally a standalone, on-demand script (run by a human before a
release), NOT a pytest/CI target: it installs the wheels into throwaway
virtualenvs.

LINUX ONLY
  Only the Linux wheels of a download are tested; Windows wheels found in
  `wheels/` are ignored and reported as such. Promotion itself is OS-independent,
  but the install checks build a venv and run `pip install`, and pip refuses a
  `win_amd64` wheel on Linux (and a manylinux wheel on Windows) -- so the
  interesting half of each scenario can only run on matching hardware. Windows
  coverage is not supported yet and needs to run on a Windows host.

PACKAGE TYPES
  A release ships three shapes of artifact, and every scenario below is phrased
  in these terms:

  generic     Packages that are not tied to one gfx arch: the `rocm` meta
              wheel/sdist, rocm_sdk_core / rocm_sdk_devel / rocm_sdk_libraries,
              rocm_profiler, torch, torchvision, torchaudio, triton, apex.
              Version promotion rewrites their metadata; the keep-list may trim
              the arch references *inside* them, but the files themselves are
              always retained.
              (promote_packages.py calls the narrower subset that actually
              carries multi-arch metadata -- rocm, rocm_sdk_*, rocm_profiler,
              torch, torchvision -- "aggregators"; that is a subset of `generic`
              here. A wheel matching none of these types is reported at startup
              rather than silently skipped.)
  device      Per-gfx packages, `<name>_device_gfx<N>-...` (rocm_sdk_device_gfx942,
              amd_torch_device_gfx1201, amd_torchvision_device_gfx1201). These are
              what the keep-list keeps or drops wholesale.
  tarball     Standalone `therock-dist-*.tar.gz` distribution tarballs, promoted
              by renaming the file; their contents are never opened.

WHAT PROMOTION DOES
  Version: promote_packages.py rewrites the prerelease segment rather than only
  stripping "rc" -- it can strip it (rc/a -> release, e.g. 7.14.0rc3 -> 7.14.0)
  or replace it (a -> rc, rc -> a). These tests drive the common rc/a -> release
  path.

  Keep-list: for a multi-arch input, `--multi-arch-targets` takes the archs to
  preserve. Device packages for every arch left out of the list are dropped, AND
  the arch references inside the generic packages (Provides-Extra / Requires-Dist
  device entries, AVAILABLE_TARGET_FAMILIES) are trimmed to match. It is
  independent of the version rewrite (with --skip-version-promotion the version
  is left unchanged), so these tests exercise it together with rc -> release.

HOW THIS FILE WORKS
  1. `parse_args` takes --input-dir (and optionally --fail-fast).
  2. `InputSet` scans the input tree once and derives everything from it: the
     source/final versions, the generic / device / jax / tarball groups, the
     keep-list arch, and the install sets (see its docstring).
  3. Each `check_promote_*` scenario receives that `InputSet`, copies the subset
     it needs into a fresh temp dir, runs promote_packages.main() on it, and
     asserts filenames, versions, arch references and installability. Scenarios
     never share a directory.
  4. `main` prints a per-scenario SUMMARY and exits non-zero on any FAILURE.
     Scenarios whose inputs are absent report SKIPPED, which is not a pass.

WHAT IS VALIDATED (scenario -> what it guarantees)
  check_promote_generic        the generic packages promote and install as one
                               coherent stack; no device packages involved
  check_promote_arch_keep_list the keep-list keeps the chosen arch's device
                               packages, drops every other arch's, and trims the
                               dropped archs out of the generic packages'
                               metadata -- while still stripping the RC suffix
                               everywhere. This is where device packages are
                               promoted and installed. The kept and dropped
                               archs are auto-detected from what is present.
  check_promote_jax            every JAX wheel promotes (the RC suffix is
                               stripped from the `+rocm<ver>` local segment);
                               only the newest set is installed (--no-deps,
                               since their runtime deps come from PyPI, not a
                               download)
  check_promote_tarball        every therock-dist tarball present (per-gfx AND
                               multiarch) is renamed rc->final
                               (promote_targz_tarball renames by filename, it
                               does not repack)
  check_promote_only_rocm      promoting only the `rocm` meta package (not the
  check_promote_only_torch     rest of the SDK) is incoherent and is rejected;
                               likewise for a torch-family-only promotion
                               (`*torch*` = torch / torchvision / torchaudio)

  Assertions used inside the scenarios above, never run standalone:
  check_promoted_file_names     the promoted dir holds exactly the expected
                                names and no rc leftovers
  check_all_wheels_same_version every wheel reports the promoted ROCm version
  check_dropped_archs_absent    no dropped arch survives inside package metadata
  check_installation            the promoted set installs into a fresh venv

HOW THE EXPECTED OUTPUT IS DERIVED
  The source version and the set of gfx architectures come from the files in the
  input directory -- nothing is hard-coded to a release:

  1. Read the source version out of package metadata via pkginfo (the
     rocm_sdk_core wheel, falling back to the rocm sdist), e.g. "7.14.0rc3".
  2. Parse it with packaging.version.Version and take its `base_version` (the
     release segment) as the promotion target, e.g.:
       7.14.0rc3       -> 7.14.0
       7.13.0a20260501 -> 7.13.0
  3. For every discovered filename, substitute the source version string with
     the final one and assert the promoted directory contains exactly that set:
       rocm_sdk_core-7.14.0rc3-...whl      -> rocm_sdk_core-7.14.0-...whl
       torch-2.12.0+rocm7.14.0rc3-...whl   -> torch-2.12.0+rocm7.14.0-...whl
     Note this is a plain string substitution on the *filename*, which is
     deliberately independent of the regex rewriting promote_packages.py applies
     to metadata inside the package -- the test must not re-implement the code
     it is checking.

  All packages in the input must belong to that one source version; a directory
  mixing releases is rejected rather than silently half-tested. Use an input with
  >= 2 gfx archs (multi-arch downloads start at 7.13) so the keep-list scenario
  has something to exclude; otherwise it is SKIPPED.

COHERENT INSTALL STACK
  promote_packages works on any subset of packages, but pip does not: a real
  download carries several mutually-exclusive versions of a distribution (torch
  2.10..2.12, several jax releases) and every supported CPython (cp310..cp314).
  So while every discovered package is promoted, the *install* checks are
  narrowed to one stack that can actually co-install:

  - the newest version of each distribution,
  - restricted to the CPython tag of the interpreter running this script
    (Python-agnostic files -- the rocm sdist, `py3-none-*` SDK wheels -- always
    qualify),
  - with triton pinned to the build the chosen torch requires. triton is the one
    distribution singled out this way because torch depends on it with an exact
    `Requires-Dist: triton==<ver>` pin, while a download can carry several
    triton builds; the other torch-family packages are not cross-pinned like
    this. See `InputSet._coherent_install_set`.

  Promotion is version- and Python-agnostic, so it behaves identically for the
  versions and CPythons left out of the install stack.

INPUT REQUIREMENTS
  - `<root>/wheels/` must exist and contain Linux wheels; an empty or
    Windows-only wheels dir is a hard error even when tarballs are present (a
    tarball-only run is not supported).
  - `<root>/tarball-multi-arch/` may be absent; the tarball scenario is SKIPPED.

PREREQUISITES
  pip install -r ./build_tools/packaging/requirements.txt

USAGE
  # Promote + verify a download_python_packages.py output tree:
  python ./build_tools/packaging/tests/promote_packages_test.py --input-dir ./download

  # Stop at the first failing scenario instead of running all of them:
  python ./build_tools/packaging/tests/promote_packages_test.py \
      --input-dir ./download --fail-fast
"""

import argparse
import functools
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from collections.abc import Callable
from pathlib import Path

from packaging.version import Version
from pkginfo import SDist, Wheel

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import promote_packages

sys.path.insert(0, os.fspath(Path(__file__).parent.parent.parent))
import setup_venv

# Scenario verdicts. SKIPPED is deliberately distinct from SUCCESS: a scenario
# that never ran must not read as coverage in the summary.
SUCCESS = "SUCCESS"
FAILURE = "FAILURE"
SKIPPED = "SKIPPED"

# A deferred check: called with no arguments, returns (passed, failure message).
Check = Callable[[], tuple[bool, str]]

# Generic (non-device) packages: always promoted, and the keep-list only rewrites
# their arch metadata, never drops the file itself. Matched by filename prefix;
# per-gfx device packages use `*_device_gfx<N>` and are matched separately.
GENERIC_WHEEL_PREFIXES = (
    "apex-",
    "rocm_profiler-",
    "rocm_sdk_core-",
    "rocm_sdk_devel-",
    "rocm_sdk_libraries-",
    "torch-",
    "torchvision-",
    "torchaudio-",
    "triton-",
)

# The `rocm` meta package (wheel or sdist). The `\d` is what separates it from
# the other rocm* distributions, whose names continue with a letter or `_`, e.g.:
#   rocm-7.14.0rc3.tar.gz          -> meta package
#   rocm_profiler-7.14.0rc3-*.whl  -> not the meta package (matched by prefix above)
_ROCM_META_RE = re.compile(r"rocm-\d")

# Metadata files inside a wheel/sdist that can carry gfx arch references.
_ARCH_METADATA_FILENAMES = ("METADATA", "PKG-INFO", "requires.txt", "_dist_info.py")

# LINUX_/WINDOWS_TARGET_FAMILIES are cross-platform reference data, NOT a
# declaration of what the release ships -- they record which OS each family is
# published for so the rocm sdist comes out byte-identical from the Linux and
# Windows build jobs and setup.py can attach sys_platform markers to device
# extras (ROCm/TheRock#5368). They are only ever consulted for families still in
# AVAILABLE_TARGET_FAMILIES, and trimming them to the keep-list would empty one
# side and trip the "if not LINUX or not WINDOWS" guard, silently dropping those
# markers. So the keep-list leaves them alone by design and this check skips
# them; AVAILABLE_TARGET_FAMILIES is the list that must be trimmed.
_PLATFORM_FAMILY_LINE_RE = re.compile(
    r"^[ \t]*(?:LINUX|WINDOWS)_TARGET_FAMILIES\.append\(.*\)[ \t]*$", re.MULTILINE
)

_BANNER_WIDTH = 81


def _banner(msg: str) -> None:
    line = "=" * _BANNER_WIDTH
    print(f"\n{line}\n{msg.center(_BANNER_WIDTH)}\n{line}")


def _is_generic(name: str) -> bool:
    """Whether `name` is a generic (non-device) package this test promotes."""
    if name.startswith(GENERIC_WHEEL_PREFIXES):
        return name.endswith(".whl")
    if _ROCM_META_RE.match(name):
        return name.endswith(".whl") or name.endswith(".tar.gz")
    return False


# Per-gfx device package marker, e.g. `rocm_sdk_device_gfx942-...` /
# `amd_torch_device_gfx1201-...`. Reuses promote_packages._GFX_ARCH (the same
# arch-token shape the promotion script keys off) so the two never drift; the
# trailing `-` anchors the token so `gfx11` can't match `gfx1153`.
_DEVICE_ARCH_RE = re.compile(rf"device_({promote_packages._GFX_ARCH})-")


def _device_arch(name: str) -> str | None:
    """The gfx arch a device package targets, or None if it is not one."""
    m = _DEVICE_ARCH_RE.search(name)
    return m.group(1) if m else None


# One concrete gfx target, as opposed to the compute-kernel family bundles that
# also ship as device packages, e.g.:
#   gfx908, gfx90a, gfx1103, gfx1250 -> concrete
#   gfx11, gfx110x, gfx115x, gfx12_0 -> family (too short, or has 'x'/'_')
_CONCRETE_ARCH_RE = re.compile(r"^gfx[0-9][0-9a-f]{2,3}$")


def _wheel_os(name: str) -> str | None:
    """The OS a wheel targets, from its filename tag, or None if it is
    OS-agnostic (the rocm sdist, any `-none-any` wheel)."""
    if "-win_" in name:
        return "windows"
    if "linux" in name:  # linux_x86_64, manylinux_2_XX_x86_64, ...
        return "linux"
    return None


def _py_installable(name: str) -> bool:
    """Whether `name` can be installed under the interpreter running the tests.

    A real download ships every supported CPython (cp310..cp314); the install
    checks run under exactly one, so version-specific wheels for other CPythons
    must be dropped -- pip rejects them with "not a supported wheel on this
    platform". Files with no CPython tag (rocm sdist, `py3-none-*` SDK wheels)
    always qualify. Promotion itself is Python-agnostic, so this only narrows the
    install/expectation sets, never what promotion is exercised against."""
    m = re.search(r"-(cp3\d+)-", name)
    return (
        m is None or m.group(1) == f"cp{sys.version_info.major}{sys.version_info.minor}"
    )


def _list_files(directory: Path) -> set[str]:
    """Filenames in `directory`, or an empty set if it doesn't exist."""
    if not directory.is_dir():
        return set()
    return {p.name for p in directory.iterdir() if p.is_file()}


def _torch_triton_pin(wheel_path: Path) -> str | None:
    """The exact triton version a torch wheel pins (`Requires-Dist: triton==`)."""
    with zipfile.ZipFile(wheel_path) as z:
        meta = next(n for n in z.namelist() if n.endswith(".dist-info/METADATA"))
        text = z.read(meta).decode("utf-8", errors="replace")
    m = re.search(r"^Requires-Dist:\s*triton==([^\s;]+)", text, re.MULTILINE)
    return m.group(1) if m else None


def _triton_wheel_matches_pin(name: str, pin: str) -> bool:
    # Prefix-match the upstream triton version so pin 3.7.1 cannot match 3.7.10.
    return name.startswith(f"triton-{pin}-") or name.startswith(f"triton-{pin}+")


def _arch_metadata_texts(path: Path) -> list[tuple[str, str]]:
    """(member name, contents) of the metadata files inside a wheel/sdist that
    the keep-list rewrites. Used to confirm dropped archs are gone from the
    inside too, not just from the set of files on disk. The member name is kept
    so a failure can point at the exact file, not just the package."""
    texts: list[tuple[str, str]] = []
    if path.name.endswith(".whl"):
        with zipfile.ZipFile(path) as z:
            for member in z.namelist():
                if member.rsplit("/", 1)[-1] in _ARCH_METADATA_FILENAMES:
                    texts.append(
                        (member, z.read(member).decode("utf-8", errors="replace"))
                    )
    elif path.name.endswith(".tar.gz"):
        with tarfile.open(path) as t:
            for member in t.getmembers():
                if (
                    member.isfile()
                    and member.name.rsplit("/", 1)[-1] in _ARCH_METADATA_FILENAMES
                ):
                    handle = t.extractfile(member)
                    if handle is not None:
                        texts.append(
                            (
                                member.name,
                                handle.read().decode("utf-8", errors="replace"),
                            )
                        )
    return texts


def _arch_reference_re(arch: str) -> re.Pattern[str]:
    """Matches `arch` as a whole token. The boundaries reject alphanumerics only,
    so `gfx10` does not match inside `gfx1010` while `gfx94x` still matches
    inside `gfx94x_dcgpu`."""
    return re.compile(rf"(?<![0-9A-Za-z]){re.escape(arch)}(?![0-9A-Za-z])")


class InputSet:
    """Classification of an input directory into promotion groups.

    Given a directory of downloaded artifacts, this splits the filenames into the
    groups each promotion scenario needs and derives everything else from them:

      source_str / source_version  the pre-promotion version (rc/a), read from
                                    rocm_sdk_core / rocm sdist metadata
      final_version / final_str    that version with the prerelease segment
                                    stripped (the rc/a -> release target)
      src_version_type             "rc" or "a", passed straight to promote_packages
      generic_files                non-device wheels + rocm sdist
      device_files                 per-gfx device wheels (`*_device_gfx<N>-`)
      jax_files                    wheels with "jax" in the name (installed
                                    --no-deps)
      tarball_files                therock-dist-*.tar.gz distribution tarballs
      unclassified                 wheels matching none of the above, reported so
                                    a newly shipped package is not silently
                                    skipped
      recognized                   everything except `unclassified`; what the
                                    mixed-release guard is checked against
      present_archs                every gfx arch seen among the device wheels
      keep_archs                   the keep-list handed to promote_packages
      dropped_archs                the archs left out of it, whose device wheels
                                    must disappear and whose metadata references
                                    must be trimmed
      install_set                  one co-installable stack of generic packages
      jax_install_set              the same, for the JAX wheels
    """

    def __init__(self, input_dir: Path) -> None:
        # download_python_packages.py emits wheels + the rocm sdist under
        # `wheels/` and distribution tarballs under `tarball-multi-arch/` -- never
        # mixed. Read each from its own subdir; the tarball one may be absent.
        self.wheels_dir = input_dir / "wheels"
        self.tarball_dir = input_dir / "tarball-multi-arch"
        self._version_cache: dict[str, str] = {}

        names = self._linux_wheel_names()
        self.source_str = self._detect_source_version(names)
        self.source_version = Version(self.source_str)
        self.final_version = Version(self.source_version.base_version)
        self.final_str = str(self.final_version)
        # rc vs a comes from the parsed prerelease kind, not a substring test
        # (an alpha input like 7.13.0a20260501 must classify as "a").
        pre = self.source_version.pre
        self.src_version_type = pre[0] if pre else "rc"

        self._classify(names)
        self._reject_mixed_versions()
        self.install_set = self._coherent_install_set()
        self.jax_install_set = self._newest_version_per_package(self.jax_files)
        # After install_set: the keep-list arch is chosen to agree with it.
        self._select_keep_list()

    def _linux_wheel_names(self) -> set[str]:
        """Filenames of the Linux (and OS-agnostic) wheels. Windows wheels cannot
        be installed here, so they are reported and dropped up front."""
        names = _list_files(self.wheels_dir)
        # Every scenario except the tarball one needs wheels, and the source
        # version is read from them, so a tarball-only tree is a hard error.
        if not names:
            raise RuntimeError(
                f"No wheels found in {self.wheels_dir}; a tarball-only input "
                f"directory is not supported."
            )
        ignored = {n for n in names if _wheel_os(n) == "windows"}
        if ignored:
            print(f"Ignoring {len(ignored)} Windows wheel(s): Linux only for now.")
        names -= ignored
        if not names:
            raise RuntimeError(
                f"{self.wheels_dir} contains only Windows wheels; this test is "
                f"Linux only and must run against a Linux download."
            )
        return names

    def _dist_version_str(self, name: str) -> str:
        """The version a package's own metadata reports, read once per file."""
        if name not in self._version_cache:
            path = self.wheels_dir / name
            meta = SDist(path) if name.endswith(".tar.gz") else Wheel(path)
            self._version_cache[name] = meta.version
        return self._version_cache[name]

    def _dist_version(self, name: str) -> Version:
        return Version(self._dist_version_str(name))

    def _detect_source_version(self, names: set[str]) -> str:
        """Read the source (pre-promotion) version from package metadata.

        Prefers the rocm_sdk_core wheel, falling back to the rocm sdist, and reads
        the version out of the package metadata rather than parsing the filename
        -- that also implicitly validates the filename encodes the same version
        its metadata claims.
        """
        for name in sorted(names):
            if name.startswith("rocm_sdk_core-") and name.endswith(".whl"):
                return self._dist_version_str(name)
        for name in sorted(names):
            if _ROCM_META_RE.match(name) and name.endswith(".tar.gz"):
                return self._dist_version_str(name)
        raise RuntimeError(
            f"Could not determine the source version from {self.wheels_dir}: "
            f"expected a rocm_sdk_core-<ver>-*.whl or a rocm-<ver>.tar.gz sdist."
        )

    def _classify(self, names: set[str]) -> None:
        """Split the input filenames into the groups the scenarios consume."""
        # Device wheels feed the multi-arch scenario, which promotes with a
        # keep-list and installs the kept arch. They are narrowed to the runner's
        # CPython (like the generic install set) and collapsed to the newest
        # version per package: a download ships several torch/torchvision versions
        # per arch (e.g. amd_torch_device_gfx1010 2.10/2.11/2.12) whose device
        # wheels cannot co-install, and the newest ones line up with the newest
        # torch/torchvision picked for the generic install stack. The keep-list is
        # still exercised across every arch within that CPython.
        self.device_files = self._newest_version_per_package(
            {n for n in names if _device_arch(n) and _py_installable(n)}
        )
        self.jax_files = {n for n in names if "jax" in n}
        self.generic_files = {
            n for n in names if _is_generic(n) and n not in self.device_files
        }
        # therock-dist tarballs come from their own dir, never the wheels dir.
        self.tarball_files = {
            n
            for n in _list_files(self.tarball_dir)
            if n.startswith("therock-dist")
            and n.endswith(".tar.gz")
            and self.source_str in n
        }
        # Anything the groups above do not recognise at all. Reported rather than
        # dropped in silence: a release that starts shipping a new distribution
        # would otherwise go untested without anyone noticing. Note this is about
        # unrecognised *names*; wheels deliberately narrowed out of the install
        # sets (other CPythons, older versions) are still recognised here.
        self.unclassified = {
            n
            for n in names
            if not _device_arch(n) and "jax" not in n and not _is_generic(n)
        }
        self.recognized = names - self.unclassified

    def _reject_mixed_versions(self) -> None:
        """Fail on an input tree holding more than one ROCm release.

        Every package of a single download carries the same ROCm version in its
        filename, directly (rocm_sdk_core-7.14.0rc3) or as a local segment
        (torch-2.12.0+rocm7.14.0rc3). A mixed directory would silently promote one
        release and leave the other behind.

        Checks every recognised wheel, not just the ones the scenarios end up
        staging: a stray release hiding among the versions/CPythons that get
        narrowed out would otherwise go unreported.
        """
        strays = sorted(n for n in self.recognized if self.source_str not in n)
        if strays:
            raise RuntimeError(
                f"{self.wheels_dir} mixes releases: expected every package to "
                f"carry {self.source_str}, but found {strays}. Promote one "
                f"release at a time."
            )

    def _keep_list_candidates(self) -> list[str]:
        """The archs that are sane to hand to `--multi-arch-targets` here.

        Two filters, both needed to keep the scenario's install check meaningful:

        1. Concrete archs only. `present_archs` also contains compute-kernel
           family bundles (gfx11, gfx12_0, gfx110x, gfx115x) which name a family
           rather than one target; keeping only a family is not a shape a release
           is cut in, so they make a poor keep-list.
        2. Archs whose device packages agree with the generic install stack. A
           download can lag an arch behind (in 7.14.0rc3 the newest
           amd_torch_device_gfx1250 is 2.11.0 while the newest torch is 2.12.0),
           and keeping such an arch would make the install check fail on that
           version skew rather than on anything promotion did.
        """
        install_versions = {self._dist_version_str(n) for n in self.install_set}
        candidates = []
        for arch in self.present_archs:
            if not _CONCRETE_ARCH_RE.match(arch):
                continue
            device = [n for n in self.device_files if _device_arch(n) == arch]
            if all(self._dist_version_str(n) in install_versions for n in device):
                candidates.append(arch)
        return candidates

    def _select_keep_list(self) -> None:
        """Choose the keep-list the multi-arch scenario runs with.

        Picks the middle candidate rather than the first: the lowest arch is the
        likely DEFAULT_TARGET_FAMILY, and keeping it would never exercise the
        default-repointing path in promote_packages. A single arch is kept so the
        promoted set stays co-installable for the install check.
        """
        self.present_archs = sorted(
            {a for n in self.device_files if (a := _device_arch(n))}
        )
        self.keep_archs: list[str] = []
        self.dropped_archs: list[str] = []
        self.dropped_device_files: set[str] = set()
        if len(self.present_archs) < 2:
            return
        candidates = self._keep_list_candidates()
        if not candidates:
            return
        self.keep_archs = [candidates[len(candidates) // 2]]
        self.dropped_archs = [a for a in self.present_archs if a not in self.keep_archs]
        self.dropped_device_files = {
            n for n in self.device_files if _device_arch(n) not in self.keep_archs
        }

    def _newest_version_per_package(self, names: set[str]) -> set[str]:
        """Keep only the highest version of each distribution, restricted to
        wheels installable under the running interpreter. A download can ship
        several torch / jax versions (that cannot be installed together) across
        several CPythons (only one of which matches this interpreter)."""
        newest: dict[str, tuple[str, Version]] = {}
        for name in names:
            if not _py_installable(name):
                continue
            pkg = name.split("-", 1)[0]  # distribution name (never has a hyphen)
            version = self._dist_version(name)
            if pkg not in newest or version > newest[pkg][1]:
                newest[pkg] = (name, version)
        return {name for name, _ in newest.values()}

    def _coherent_install_set(self) -> set[str]:
        """One installable stack of generic packages: the newest of each
        distribution, with triton pinned to the build the chosen torch requires.

        triton is the only distribution filtered by a dependency pin because torch
        depends on it with an exact `Requires-Dist: triton==<ver>`, while a
        download can carry several triton builds (3.6.0+rocmX, 3.7.1+gitYYYY.rocmX).
        The other torch-family packages are not cross-pinned like this, so newest
        is good enough for them.
        """
        picks = self._newest_version_per_package(self.generic_files)
        torch = next((n for n in picks if n.startswith("torch-")), None)
        triton_wheels = {n for n in self.generic_files if n.startswith("triton-")}
        if torch is None or not triton_wheels:
            return picks

        pin = _torch_triton_pin(self.wheels_dir / torch)
        if pin is None:
            raise RuntimeError(
                f"{torch} does not pin an exact triton build "
                f"('Requires-Dist: triton==<ver>'); refusing to guess which of "
                f"{sorted(triton_wheels)} belongs with it."
            )
        matching = {
            n
            for n in triton_wheels
            if _triton_wheel_matches_pin(n, pin) and _py_installable(n)
        }
        if not matching:
            raise RuntimeError(
                f"{torch} pins triton=={pin} but no matching triton wheel is "
                f"present in {self.wheels_dir} (found: {sorted(triton_wheels)})"
            )
        return {n for n in picks if not n.startswith("triton-")} | matching

    def promoted(self, names: set[str]) -> set[str]:
        """The filenames `names` are expected to have after promotion."""
        return {promoted_name(n, self.source_str, self.final_str) for n in names}


def promoted_name(rc_name: str, version: str, final_version: str) -> str:
    """The filename `rc_name` is expected to have after promotion.

    Substitutes the source version string for the final one, e.g.:
      torch-2.12.0+rocm7.14.0rc3-cp312-...whl -> torch-2.12.0+rocm7.14.0-cp312-...whl
    """
    return rc_name.replace(version, final_version)


def check_promoted_file_names(dir_path: Path, expected: set[str]) -> tuple[bool, str]:
    """Assert the promoted directory holds exactly `expected` (no leftovers)."""
    actual = {p.name for p in dir_path.glob("*") if p.name != ".venv"}
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        return (
            False,
            f"Promoted file set mismatch.\n  missing: {missing}\n  unexpected: {unexpected}",
        )
    return True, ""


def check_all_wheels_same_version(
    dir_path: Path, expected_version: Version
) -> tuple[bool, str]:
    """Assert every wheel's metadata reports the promoted ROCm version.

    Only the ROCm part is compared: SDK wheels carry it as the version itself
    (7.14.0), while torch-family wheels carry it in the local segment on top of
    their own upstream version (2.12.0+rocm7.14.0), so their upstream version is
    deliberately not constrained here.

    Wheels only. The `rocm` sdist is not checked here -- promote_packages repacks
    it rather than rewriting it in place, and check_promoted_file_names already
    asserts it came out under the promoted name.
    """
    local_version = "rocm" + str(expected_version)
    for file in dir_path.glob("*.whl"):
        version = Version(Wheel(file).version)
        if version == expected_version and version.local is None:
            continue  # generic / device SDK wheels
        # torch-family wheels carry the rocm tag as a local version segment,
        # sometimes prefixed with a git hash (e.g. `git43422b04.rocm7.14.0`).
        if version.local is not None and version.local.endswith(local_version):
            continue
        return (
            False,
            f"{file.name} has version {version}, but expected version is {expected_version}",
        )
    return True, ""


def check_dropped_archs_absent(
    dir_path: Path, dropped_archs: list[str]
) -> tuple[bool, str]:
    """Assert the keep-list also trimmed the arch references INSIDE the packages.

    Deleting the device wheels of the dropped archs is only half of what the
    keep-list does: the generic packages still list every arch in their
    Provides-Extra / Requires-Dist / AVAILABLE_TARGET_FAMILIES entries, and those
    must be trimmed too or the promoted `torch` would still declare a dependency
    on a device package that no longer ships.

    The LINUX_/WINDOWS_TARGET_FAMILIES lists are exempt; see
    `_PLATFORM_FAMILY_LINE_RE` for why they are meant to survive the keep-list.

    Reports every offending member of every package rather than stopping at the
    first: when the keep-list misses a metadata shape it usually misses it in all
    the packages that share that shape, and seeing the whole set at once is what
    makes the cause obvious.
    """
    patterns = {arch: _arch_reference_re(arch) for arch in dropped_archs}
    offenders: list[str] = []
    for file in sorted(dir_path.glob("*")):
        if file.name == ".venv" or not _is_generic(file.name):
            continue
        for member, text in _arch_metadata_texts(file):
            text = _PLATFORM_FAMILY_LINE_RE.sub("", text)
            leftover = sorted(a for a, rx in patterns.items() if rx.search(text))
            if leftover:
                offenders.append(f"  {file.name} :: {member}\n    {leftover}")
    if offenders:
        return (
            False,
            "still reference dropped arch(es) after the keep-list pass:\n"
            + "\n".join(offenders),
        )
    return True, ""


def check_installation(
    dir_path: Path, package_names: set[str], no_deps: bool = False
) -> tuple[bool, str]:
    """Install `package_names` from `dir_path` into a throwaway venv built there.

    The caller passes the exact set to install rather than letting this glob, so
    each scenario stays explicit about what it expects to be installable.

    Note: dir_path must be a TemporaryDirectory, otherwise clean up .venv yourself.

    This is NOT an offline check: pip runs without `--no-index`, so transitive
    deps that a download does not carry (sympy, networkx, jinja2, ... for torch)
    are fetched from PyPI.

    no_deps installs without resolving dependencies -- used for the JAX wheels,
    whose runtime deps live on PyPI and are not part of the promoted set; there
    we only confirm the promoted wheels are structurally installable.
    """
    # Distribution tarballs are renamed, never installed.
    packages = sorted(n for n in package_names if "therock-dist" not in n)
    if not packages:
        return False, "Nothing to install: the package list is empty."
    missing = [n for n in packages if not (dir_path / n).is_file()]
    if missing:
        return False, f"Cannot install, missing from the promoted set: {missing}"

    try:
        setup_venv.create_venv(dir_path / ".venv")
        python_exe = setup_venv.find_venv_python_exe(dir_path / ".venv")
        if python_exe is None:
            return (
                False,
                "Problem when installing temporary venv: Python executable not found",
            )
        cmd = [python_exe, "-m", "pip", "install"]
        if no_deps:
            cmd.append("--no-deps")
        subprocess.run(
            cmd + [os.fspath(dir_path / n) for n in packages],
            capture_output=True,
            encoding="utf-8",
            check=True,
        )
    except subprocess.CalledProcessError as e:
        return False, e.stderr
    return True, ""


def _copy_inputs(src: Path, dst: Path, names: set[str]) -> None:
    """Copy the `names` subset of `src` into the scenario's own temp dir, so the
    input tree is never promoted in place and scenarios cannot affect each other.
    Every requested name must exist -- a silent partial copy would make the
    scenario assert against a set it never staged."""
    missing = sorted(n for n in names if not (src / n).is_file())
    if missing:
        raise RuntimeError(f"Cannot stage inputs from {src}, not found: {missing}")
    for name in names:
        shutil.copy2(src / name, dst / name)


def _evaluate(
    tmp_dir: Path, checks: list[tuple[str, Check]], expect_success: bool
) -> str:
    """Run `checks` in order and turn the results into a scenario verdict.

    `checks` are (name, thunk) pairs, where each thunk runs one check and returns
    its (passed, message). They are deliberately not evaluated up front: a check
    can cost a fresh venv and a multi-GB pip install, so evaluation stops at the
    first result that already decides the verdict.

    With expect_success=True every check must pass, so the first failure decides
    it. With expect_success=False the scenario is asserting that an incomplete
    promotion is NOT a usable release, so the first check to reject it decides
    it; only a clean sweep is a failure.
    """
    for name, check in checks:
        passed, message = check()
        if passed:
            continue
        if expect_success:
            print(f"\n[ERROR] {name} failed:\n{message}")
            return FAILURE
        print(f"  incomplete promotion correctly rejected by {name}:\n{message}")
        return SUCCESS

    if expect_success:
        return SUCCESS
    print(
        "\n[ERROR] every check passed, but this promotion was incomplete and "
        "should not have produced a usable release. Checks that passed: "
        + ", ".join(name for name, _ in checks)
    )
    # Cross-platform directory dump (the old `ls` shell-out broke on Windows and
    # masked the real failure).
    for entry in sorted(p.name for p in tmp_dir.glob("*")):
        print(f"    {entry}")
    return FAILURE


def _done(label: str, verdict: str) -> str:
    """Print a scenario's closing banner and pass its verdict back unchanged, so
    a scenario can `return _done(...)` on every exit path."""
    _banner(f"TEST DONE: {label}. Result: {verdict}")
    return verdict


def check_promote_generic(inputs: InputSet) -> str:
    """Promote every generic package in the input and assert the result is
    renamed, re-versioned and installable -- the baseline scenario the others are
    variations of.

    Promotion runs over all of them, including the mutually-exclusive
    torch/triton/apex versions a download carries, since it is version-agnostic.
    Only the install check is narrowed to the one co-installable stack.
    """
    label = "promote generic packages"
    _banner("TEST: promotion of all generic packages (should SUCCEED)")
    if not inputs.generic_files:
        print("[SKIP] no generic packages found in the input.")
        return _done(label, SKIPPED)

    expected = inputs.promoted(inputs.generic_files)
    installable = inputs.promoted(inputs.install_set)
    with tempfile.TemporaryDirectory(prefix="PromoteTest-Generic-") as tmp:
        tmp_dir = Path(tmp)
        _copy_inputs(inputs.wheels_dir, tmp_dir, inputs.generic_files)
        promote_packages.main(
            tmp_dir, delete=True, src_version_type=inputs.src_version_type
        )
        verdict = _evaluate(
            tmp_dir,
            [
                (
                    "check_promoted_file_names",
                    lambda: check_promoted_file_names(tmp_dir, expected),
                ),
                (
                    "check_all_wheels_same_version",
                    lambda: check_all_wheels_same_version(
                        tmp_dir, inputs.final_version
                    ),
                ),
                (
                    "check_installation",
                    lambda: check_installation(tmp_dir, installable),
                ),
            ],
            expect_success=True,
        )
    return _done(label, verdict)


def check_promote_arch_keep_list(inputs: InputSet) -> str:
    """Exercise the keep-list pass (`--multi-arch-targets`): the kept arch's
    device packages survive, the device packages of the dropped archs are gone,
    and the dropped archs no longer appear in the generic packages' metadata.

    Needs >= 2 gfx archs with device packages present for the keep-list to
    exclude anything; single-arch inputs are SKIPPED.
    """
    label = "multi-arch keep-list promotion"
    _banner("TEST: multi-arch promotion with a keep-list (should SUCCEED)")
    if not inputs.keep_archs or not inputs.dropped_device_files:
        print(
            "[SKIP] need >= 2 gfx archs with device packages for the keep-list "
            f"to exclude anything; found {inputs.present_archs or '(none)'}."
        )
        return _done(label, SKIPPED)

    print(f"  keep-list: {inputs.keep_archs}; dropped: {inputs.dropped_archs}")
    # All generic packages, not just the install stack: every one of them carries
    # arch metadata the keep-list has to trim, so they all belong in the
    # check_dropped_archs_absent sweep.
    staged = inputs.generic_files | inputs.device_files
    expected = inputs.promoted(staged - inputs.dropped_device_files)
    installable = inputs.promoted(
        inputs.install_set | (inputs.device_files - inputs.dropped_device_files)
    )
    with tempfile.TemporaryDirectory(prefix="PromoteTest-MultiArch-") as tmp:
        tmp_dir = Path(tmp)
        _copy_inputs(inputs.wheels_dir, tmp_dir, staged)
        promote_packages.main(
            tmp_dir,
            delete=True,
            multi_arch_targets=inputs.keep_archs,
            src_version_type=inputs.src_version_type,
        )
        verdict = _evaluate(
            tmp_dir,
            [
                (
                    "check_promoted_file_names",
                    lambda: check_promoted_file_names(tmp_dir, expected),
                ),
                (
                    "check_all_wheels_same_version",
                    lambda: check_all_wheels_same_version(
                        tmp_dir, inputs.final_version
                    ),
                ),
                (
                    "check_dropped_archs_absent",
                    lambda: check_dropped_archs_absent(tmp_dir, inputs.dropped_archs),
                ),
                (
                    "check_installation",
                    lambda: check_installation(tmp_dir, installable),
                ),
            ],
            expect_success=True,
        )
    return _done(label, verdict)


def check_promote_jax(inputs: InputSet) -> str:
    """Promote every JAX wheel, whose ROCm version sits in the `+rocm<ver>` local
    segment. Promotion runs over all of them (same shape as the generic
    scenario); only the install check is narrowed to the newest co-installable
    set. Installed with --no-deps because their runtime deps come from PyPI
    rather than the download, so this confirms structural installability only.
    Skipped when the input carries no JAX wheels."""
    label = "promote JAX wheels"
    _banner("TEST: promotion of JAX wheels (should SUCCEED)")
    if not inputs.jax_files:
        print("[SKIP] no JAX wheels found for this version.")
        return _done(label, SKIPPED)

    expected = inputs.promoted(inputs.jax_files)
    installable = inputs.promoted(inputs.jax_install_set)
    with tempfile.TemporaryDirectory(prefix="PromoteTest-Jax-") as tmp:
        tmp_dir = Path(tmp)
        _copy_inputs(inputs.wheels_dir, tmp_dir, inputs.jax_files)
        promote_packages.main(
            tmp_dir, delete=True, src_version_type=inputs.src_version_type
        )
        verdict = _evaluate(
            tmp_dir,
            [
                (
                    "check_promoted_file_names",
                    lambda: check_promoted_file_names(tmp_dir, expected),
                ),
                (
                    "check_all_wheels_same_version",
                    lambda: check_all_wheels_same_version(
                        tmp_dir, inputs.final_version
                    ),
                ),
                (
                    "check_installation",
                    lambda: check_installation(tmp_dir, installable, no_deps=True),
                ),
            ],
            expect_success=True,
        )
    return _done(label, verdict)


def check_promote_tarball(inputs: InputSet) -> str:
    """Every therock-dist tarball present (per-gfx AND multiarch) is promoted by
    a filename-based rc->final rename (promote_targz_tarball), not by repacking.

    Names are discovered in the separate `tarball-multi-arch/` dir and the
    tarballs are staged and promoted from there; the expected promoted name is
    the discovered name with the rc->final rename applied.
    """
    label = "promote therock-dist tarballs"
    _banner("TEST: promotion of therock-dist tarballs (should SUCCEED)")
    if not inputs.tarball_files:
        print("[SKIP] no therock-dist tarballs found in tarball-multi-arch/.")
        return _done(label, SKIPPED)

    print(f"  discovered {len(inputs.tarball_files)} tarball(s):")
    for name in sorted(inputs.tarball_files):
        print(f"    {name}")

    verdict = SUCCESS
    with tempfile.TemporaryDirectory(prefix="PromoteTest-Tarball-") as tmp:
        tmp_dir = Path(tmp)
        _copy_inputs(inputs.tarball_dir, tmp_dir, inputs.tarball_files)
        promote_packages.main(
            tmp_dir, delete=True, src_version_type=inputs.src_version_type
        )
        produced = {p.name for p in tmp_dir.glob("*")}
        for rc_name in sorted(inputs.tarball_files):
            final_name = promoted_name(rc_name, inputs.source_str, inputs.final_str)
            if final_name not in produced:
                print(
                    f"\n[ERROR] expected {final_name} after promotion; "
                    f"got {sorted(produced)}"
                )
                verdict = FAILURE
            elif rc_name in produced:
                print(f"\n[ERROR] rc tarball {rc_name} survived promotion")
                verdict = FAILURE
    return _done(label, verdict)


def check_partial_promotion(inputs: InputSet, match_files: str, label: str) -> str:
    """Promote only the `match_files` slice of an otherwise complete stack and
    assert the outcome is NOT a coherent release: leaving e.g. the meta package
    on final while torch stays on rc must be caught by at least one check.

    With deferred evaluation the filename / version checks usually decide first;
    the install check is still listed so a future mix that somehow passes those
    still has to survive pip.
    """
    scenario = f"promote only {label}"
    _banner(f"TEST: promotion of only {label} packages (should FAIL)")
    if not inputs.generic_files:
        print("[SKIP] no generic packages found in the input.")
        return _done(scenario, SKIPPED)

    expected = inputs.promoted(inputs.generic_files)
    with tempfile.TemporaryDirectory(prefix=f"PromoteTest-Only-{label}-") as tmp:
        tmp_dir = Path(tmp)
        _copy_inputs(inputs.wheels_dir, tmp_dir, inputs.generic_files)
        promote_packages.main(
            tmp_dir,
            match_files=match_files,
            delete=True,
            src_version_type=inputs.src_version_type,
        )
        # Install whatever promotion actually left behind, not `expected`: the
        # directory now holds a mix of rc and final packages.
        produced = {p.name for p in tmp_dir.glob("*") if p.name != ".venv"}
        verdict = _evaluate(
            tmp_dir,
            [
                (
                    "check_promoted_file_names",
                    lambda: check_promoted_file_names(tmp_dir, expected),
                ),
                (
                    "check_all_wheels_same_version",
                    lambda: check_all_wheels_same_version(
                        tmp_dir, inputs.final_version
                    ),
                ),
                ("check_installation", lambda: check_installation(tmp_dir, produced)),
            ],
            expect_success=False,
        )
    return _done(scenario, verdict)


SCENARIOS = (
    ("check_promote_generic", check_promote_generic),
    ("check_promote_arch_keep_list", check_promote_arch_keep_list),
    ("check_promote_jax", check_promote_jax),
    ("check_promote_tarball", check_promote_tarball),
    # `rocm-*` matches the meta package only (rocm-7.14.0rc3.tar.gz). A bare
    # `rocm*` would also pull in rocm_profiler / rocm_sdk_* and stop being a
    # true partial promotion.
    (
        "check_promote_only_rocm",
        functools.partial(check_partial_promotion, match_files="rocm-*", label="rocm"),
    ),
    (
        "check_promote_only_torch",
        functools.partial(
            check_partial_promotion, match_files="*torch*", label="torch"
        ),
    ),
)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="On-demand test of pre-release->final package promotion over a "
        "local directory of Linux packages. See the module docstring for what is "
        "validated.",
        epilog=(
            "examples:\n"
            "  %(prog)s --input-dir ./download\n"
            "  %(prog)s --input-dir ./download --fail-fast\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="download_python_packages.py output root. Must contain a wheels/ "
        "with Linux wheels; tarball-multi-arch/ is optional.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at the first failing scenario. Off by default: each scenario "
        "builds its own venv, so a full run is slow but reports every problem at "
        "once.",
    )
    args = parser.parse_args(argv)
    if not args.input_dir.is_dir():
        parser.error(f"--input-dir {args.input_dir} is not a directory")
    return args


def print_input_summary(inputs: InputSet) -> None:
    print(
        f"Testing promotion {inputs.source_version} -> {inputs.final_version} "
        f"(linux) from {inputs.wheels_dir}"
    )
    print(
        f"  archs present: {inputs.present_archs or '(none)'}; "
        f"keep-list: {inputs.keep_archs or '(none)'}"
    )
    print(
        f"  generic: {len(inputs.generic_files)} "
        f"(install stack: {len(inputs.install_set)}), "
        f"device: {len(inputs.device_files)}, "
        f"jax: {len(inputs.jax_files)} (install: {len(inputs.jax_install_set)}), "
        f"tarballs: {len(inputs.tarball_files)}"
    )
    if inputs.unclassified:
        print(
            f"  [WARN] {len(inputs.unclassified)} wheel(s) match no known package "
            f"type and are NOT promoted or installed by any scenario:"
        )
        for name in sorted(inputs.unclassified):
            print(f"    {name}")


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    inputs = InputSet(args.input_dir)
    print_input_summary(inputs)

    results: dict[str, str] = {}
    for name, scenario in SCENARIOS:
        results[name] = scenario(inputs)
        if args.fail_fast and results[name] == FAILURE:
            print(f"\n--fail-fast: stopping after {name}.")
            break

    _banner("SUMMARY")
    width = max(len(name) for name, _ in SCENARIOS) + 1
    for name, _ in SCENARIOS:
        print(f"{name + ':':<{width}} {results.get(name, 'NOT RUN')}")
    print("=" * _BANNER_WIDTH)

    return 1 if FAILURE in results.values() else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
