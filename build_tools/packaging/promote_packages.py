#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Promotes / repackages ROCm and PyTorch Python packages.

This script promotes the version of two kinds of artifact:
  - Python wheels (.whl) and the `rocm` source distribution (`rocm-<ver>.tar.gz`),
    whose internal version metadata is rewritten in place, and
  - standalone `therock-dist-*.tar.gz` distribution tarballs, which are renamed
    to the promoted version (their contents are NOT opened or modified).
It optionally also restricts which gfx target architectures are referenced by the
package metadata.

Promotion is parameterised by:
  --src-version-type   prerelease type to look for in source: 'rc' or 'a'
  --dest-version       form to apply: 'release' (strip prerelease),
                       'rc<N>' (e.g. 'rc1'), or 'a<YYYYMMDD>' (e.g. 'a20260501')

Common workflows:
  - rc -> release (default)  e.g. 7.10.0rc1       -> 7.10.0
  - a  -> release            e.g. 7.13.0a20260501 -> 7.13.0
  - a  -> rc                 e.g. 7.13.0a20260501 -> 7.13.0rc1
Other combinations are mechanically possible but not typical.

The keep-list pass (--multi-arch-targets) is independent of version promotion: it
runs whenever --multi-arch-targets is supplied, alongside any version rewrite. To
run ONLY the keep-list pass and leave the version untouched, use
--skip-version-promotion (which requires --multi-arch-targets and is mutually
exclusive with --dest-version / --src-version-type).

Arch filtering for multi-arch uses a positive list: the operator names the archs they want
preserved, and everything else is dropped — Provides-Extra/Requires-Dist
entries are removed from multi-arch aggregator wheels (rocm, torch, ...),
matching AVAILABLE_TARGET_FAMILIES entries are removed from multi-arch
_dist_info.py (with DEFAULT_TARGET_FAMILY repointed at the first kept arch if
needed), and per-gfx wheels for non-kept archs are skipped or deleted.
Single-arch packages are detected automatically and pass through unchanged.

PREREQUISITES:
  - pip install -r ./build_tools/packaging/requirements.txt

SIDE EFFECTS:
  - Creates NEW promoted package files side-by-side with the original files
  - By default, DOES NOT delete original files (safe to run without --delete-old-on-success)
  - With --delete-old-on-success flag, removes original files after promotion
  - Multi-arch per-gfx wheels for archs NOT in --multi-arch-targets are always skipped
    (and deleted with --delete-old-on-success), since they are not retained

TYPICAL USAGE:
  # Promote all RC packages in a directory (keeps original RC files):
  python ./build_tools/packaging/promote_packages.py --input-dir=./release_candidates/rc1/

  # Promote and delete original RC files on success:
  python ./build_tools/packaging/promote_packages.py --input-dir=./release_candidates/rc1/ --delete-old-on-success

  # Promote only specific files matching a pattern:
  python ./build_tools/packaging/promote_packages.py --input-dir=./release_candidates/ --match-files='*rc2*'

  # Promote some nightly to release (e.g. 7.13.0a20260430 -> 7.13.0):
  python ./build_tools/packaging/promote_packages.py --input-dir=./release_candidates/ --src-version-type=a

  # Promote alpha to release candidate (e.g. 7.13.0a20260430 -> 7.13.0rc1)
  python ./build_tools/packaging/promote_packages.py --input-dir=./release_candidates/ --src-version-type=a --dest-version=rc1

  # Restrict to a positive arch list (no version change); per-gfx wheels for
  # other archs are skipped/deleted, multi-arch aggregator wheels are rewritten:
  python ./build_tools/packaging/promote_packages.py --input-dir=./release_candidates/ --skip-version-promotion --multi-arch-targets=gfx1201,gfx1010,gfx11

TESTING:
  # Point the on-demand test at a directory of already-downloaded RC packages:
  python ./build_tools/packaging/tests/promote_packages_test.py --input-dir ./rc_packages
"""

import argparse
import datetime
import fileinput
import functools
import importlib.util
import pathlib
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile

from packaging.version import Version
from pkginfo import Wheel

# Need dynamic load as change_wheel_version needs to be imported via parent directory
this_file = pathlib.Path(__file__).resolve()
build_tools_dir = this_file.parent.parent
# ../third_party/change_wheel_version/change_wheel_version.py
change_wheel_version_path = (
    build_tools_dir / "third_party" / "change_wheel_version" / "change_wheel_version.py"
)

spec = importlib.util.spec_from_file_location(
    "third_party_change_wheel_version", change_wheel_version_path
)
change_wheel_version = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(change_wheel_version)

if not hasattr(change_wheel_version, "change_wheel_version"):
    raise ImportError(
        "change_wheel_version module does not expose change_wheel_version function"
    )


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="""Promotes packages from release candidate to final release (e.g. 7.10.0rc1 --> 7.10.0).

Promotion works for wheels and .tar.gz.
Wheels version is determined by python library to interact with the wheel.
For tar.gz., the version is extract from <.tar.gz>/PKG-INFO file.
""",
        usage="python ./build_tools/packaging/promote_packages.py --input-dir=./release_candidates/rc1/ --delete-old-on-success",
    )
    parser.add_argument(
        "--input-dir",
        help="Path to the directory that contains .whl and .tar.gz files to promote",
        type=pathlib.Path,
        required=True,
    )
    parser.add_argument(
        "--match-files",
        help="Limits selection in '--input-dir' to files matchings this argument. Use wild cards if needed, e.g. '*rc2*' (default '*' to promote all files in '--input-dir')",
        default="*",
    )
    parser.add_argument(
        "--delete-old-on-success",
        help="Deletes old file after successful promotion",
        action=argparse.BooleanOptionalAction,
        # Sentinel default: lets us distinguish "user did not pass --delete-old-on-success"
        # from "user explicitly passed --delete-old-on-success" (needed for the
        # --skip-version-promotion mutex check below). Resolved to False after that check.
        default=None,
    )
    parser.add_argument(
        "--src-version-type",
        help="Source prerelease type to look for in package versions (e.g. 'a', 'rc'). Defaults to 'rc'.",
        # Sentinel default: lets us distinguish "user did not pass --src-version-type"
        # from "user explicitly passed --src-version-type=rc" (needed for the
        # --skip-version-promotion mutex check below). Resolved to "rc" after that check.
        default=None,
        choices=["rc", "a"],
    )
    parser.add_argument(
        "--dest-version",
        help=(
            "Target version form. Accepts the literal 'release' to strip the source "
            "prerelease entirely (the historical default), or an explicit replacement "
            "like 'rc1', 'rc2', 'a20260501'. Example: --src-version-type=a "
            "--dest-version=rc1 promotes 7.13.0a20260430 -> 7.13.0rc1. "
            "Defaults to 'release'."
        ),
        # Sentinel default: lets us distinguish "user did not pass --dest-version"
        # from "user explicitly passed --dest-version=release" (needed for the
        # --skip-version-promotion mutex check below). Resolved to "release" after that check.
        default=None,
    )
    parser.add_argument(
        "--multi-arch-targets",
        help=(
            "Optional. Comma-separated positive list of gfx targets to retain "
            "(e.g. 'gfx1201,gfx1010,gfx11'). When set, multi-arch aggregator "
            "wheels and per-gfx wheels for archs not in this list are dropped "
            "or rewritten; single-arch packages pass through untouched. "
            "When NOT set (default), no arch filtering is applied — multi-arch "
            "wheels are promoted unchanged with all their gfx targets."
        ),
        default=None,
    )
    parser.add_argument(
        "--skip-version-promotion",
        help=(
            "Only apply --multi-arch-targets and repack; do not change the version. "
            "Requires --multi-arch-targets and is mutually exclusive with "
            "--dest-version / --src-version-type. "
            "NOTE: Will match all versions found, ignoring --src-version-type. "
            "Use --match-files to limit the scope if needed."
        ),
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    args = parser.parse_args(argv)

    # --dest-version format check (only when explicitly provided): must be "release"
    # or a recognised prerelease form.
    if args.dest_version is not None:
        if not re.fullmatch(r"release|a\d{8}|rc\d+", args.dest_version):
            parser.error(
                f"--dest-version={args.dest_version!r} is not recognised. Use 'release', "
                "or an explicit prerelease like 'rc1' or 'a<YYYYMMDD>' (e.g. 'a20260501')."
            )
        # `a<YYYYMMDD>` must be a real calendar date (catches a20260230, a20261301, etc.)
        if args.dest_version.startswith("a"):
            try:
                datetime.datetime.strptime(args.dest_version[1:], "%Y%m%d")
            except ValueError:
                parser.error(
                    f"--dest-version={args.dest_version!r}: 'a' must be followed by a "
                    "valid YYYYMMDD date (e.g. 'a20260501')"
                )

    if args.skip_version_promotion:
        if not args.multi_arch_targets:
            parser.error("--skip-version-promotion requires --multi-arch-targets")
        # In --skip-version-promotion mode, version-related flags are ignored —
        # reject them explicitly rather than silently dropping the user's input.
        # The sentinel defaults (None) tell us whether each flag was passed explicitly.
        if args.dest_version is not None:
            parser.error(
                "--skip-version-promotion is mutually exclusive with --dest-version"
            )
        if args.src_version_type is not None:
            parser.error(
                "--skip-version-promotion is mutually exclusive with --src-version-type"
            )
        # Do not allow deletion of packages when we do not change the version.
        # Could nuke otherwise the packages.
        if args.delete_old_on_success is True:
            parser.error(
                "--skip-version-promotion is mutually exclusive with "
                "--delete-old-on-success"
            )

    # Resolve sentinel defaults after the mutex check.
    if args.dest_version is None:
        args.dest_version = "release"
    if args.src_version_type is None:
        args.src_version_type = "rc"
    if args.delete_old_on_success is None:
        args.delete_old_on_success = False

    return args


def update_metadata_rocm_requires_dist(
    new_dir_path: pathlib.Path,
    package_name_no_version: str,
    old_version: str,
    old_rocm_version: str,
    new_rocm_version: str,
) -> None:
    """Update Requires-Dist lines in METADATA that reference rocm, leaving others unchanged."""
    metadata_path = (
        new_dir_path / f"{package_name_no_version}-{old_version}.dist-info" / "METADATA"
    )
    if metadata_path.exists():
        print(f"      {metadata_path}")
        with fileinput.input(
            files=metadata_path,
            encoding="utf-8",
            inplace=True,
        ) as f:
            for line in f:
                if line.startswith("Summary:") and (
                    "TheRock" in line or "rocm" in line
                ):
                    print(line.replace(old_rocm_version, new_rocm_version), end="")
                elif line.startswith("Requires-Dist") and "rocm" in line:
                    print(line.replace(old_rocm_version, new_rocm_version), end="")
                else:
                    print(line, end="")


def compute_new_version_str(
    version_str: str, src_version_type: str, dest_version: str
) -> str:
    """Return the new version string after applying `dest_version` to `version_str`.

    The source prerelease segment (`src_version_type` plus any trailing digits) is
    either stripped or replaced:
      - `dest_version == "release"`: strip it
        (`7.13.0a20260430` -> `7.13.0`, `2.10.0+rocm7.13.0a20260430` -> `2.10.0+rocm7.13.0`).
      - otherwise: replace it with `dest_version`
        (`7.13.0a20260430` -> `7.13.0rc1` when called with src='a', dest='rc1').
    """
    # src_version_type is constrained to "rc" or "a" by argparse; regex-safe.
    replacement = "" if dest_version == "release" else dest_version
    return re.sub(rf"{src_version_type}\d*", replacement, version_str)


def _update_runpath_version(
    runpath: str, old_rocm_version: str, new_rocm_version: str
) -> str:
    return runpath.replace(old_rocm_version, new_rocm_version)


def _is_elf_file(path: pathlib.Path) -> bool:
    try:
        with path.open("rb") as f:
            return f.read(4) == b"\x7fELF"
    except OSError:
        return False


def _file_contains_bytes(path: pathlib.Path, needle: bytes) -> bool:
    if not needle:
        return False
    overlap = len(needle) - 1
    previous = b""
    try:
        with path.open("rb") as f:
            while chunk := f.read(1024 * 1024):
                data = previous + chunk
                if needle in data:
                    return True
                previous = data[-overlap:] if overlap else b""
    except OSError:
        return False
    return False


def _update_torch_elf_runpath_versions(
    torch_dir: pathlib.Path, old_rocm_version: str, new_rocm_version: str
) -> None:
    if old_rocm_version == new_rocm_version:
        return

    old_rocm_version_bytes = old_rocm_version.encode("utf-8")
    elf_files = [
        file_path
        for file_path in sorted(torch_dir.rglob("*"))
        if not file_path.is_symlink()
        and file_path.is_file()
        and _is_elf_file(file_path)
        and _file_contains_bytes(file_path, old_rocm_version_bytes)
    ]
    if not elf_files:
        return
    if not shutil.which("patchelf"):
        raise RuntimeError("patchelf is required to promote torch ELF RUNPATHs")

    for file_path in elf_files:
        try:
            runpath = subprocess.check_output(
                ["patchelf", "--print-rpath", file_path], text=True
            ).strip()
        except subprocess.CalledProcessError:
            continue
        updated_runpath = _update_runpath_version(
            runpath, old_rocm_version, new_rocm_version
        )
        if updated_runpath == runpath:
            continue
        print(f"      Updating ELF RUNPATH: {file_path}")
        subprocess.check_call(["patchelf", "--set-rpath", updated_runpath, file_path])


# Arch token shape used in wheel filenames, extras names, and dep names. Covers
# the three packaging levels emitted by rocm_bootstrap:
#   family     `gfx9`, `gfx11`, `gfx12`           -> `gfx` + digits
#   target     `gfx906`, `gfx90a`, `gfx942`       -> `gfx` + digits + opt. letter
#   sub-family `gfx9_4`, `gfx11_5`, `gfx12_0`     -> ...plus optional `_<digits>`
# Regex breakdown:
#   gfx              literal prefix
#   [0-9]+           one or more digits (required: rejects junk like `gfx_foo`)
#   [a-z]?           one optional lowercase letter (target suffix, e.g. `90a`)
#   (?:_[0-9]+)?     optional `_<digits>` group for sub-family form (`_0`, `_4`)
#                    `(?:...)` is a non-capturing group so the `?` applies to
#                    the whole `_<digits>` chunk without adding a capture.
# Hyphenated variants like `gfx94x-dcgpu` are matched only up to `gfx94x`;
# if those become first-class keep-list targets, broaden here.
_GFX_ARCH = r"gfx[0-9]+[a-z]?(?:_[0-9]+)?"


def _scan_multiarch_metadata(path: pathlib.Path) -> set[str]:
    """If `path` is multi-arch, return all gfx archs it references. Else empty set.

    Multi-arch is signalled by any `device-all` or `device-gfx<N>` token in a
    `Provides-Extra:` header or an `extra == "..."` qualifier. The returned set
    aggregates gfx archs from those tokens AND from `Requires-Dist: *-gfx<N>`
    lines. Single-arch metadata (only `extra == "device"`, or no `device-*`
    extras at all) returns the empty set.
    """
    text = path.read_text(encoding="utf-8")
    extras_re = re.compile(
        rf'(?:^Provides-Extra:\s*device-|extra\s*==\s*"device-)(all|{_GFX_ARCH})',
        re.MULTILINE,
    )
    requires_re = re.compile(
        rf"^Requires-Dist:\s*[A-Za-z0-9_.-]*-({_GFX_ARCH})\b", re.MULTILINE
    )
    extras_tokens = {m.group(1) for m in extras_re.finditer(text)}
    if not extras_tokens:
        return set()
    archs = {t for t in extras_tokens if t != "all"}
    archs.update(m.group(1) for m in requires_re.finditer(text))
    return archs


def _assert_keep_overlaps_found(
    label: str, found_archs: set[str], keep_archs: list[str]
) -> None:
    """Raise if `keep_archs` shares no element with `found_archs`."""
    if not (found_archs & set(keep_archs)):
        raise ValueError(
            f"--multi-arch-targets={keep_archs} has no overlap with archs in "
            f"{label}: {sorted(found_archs)}"
        )


_GFX_ARCH_DIGITS_RE = re.compile(r"^gfx(\d+)")


def _repoint_priority(keep_archs: list[str]) -> list[str]:
    """Order `keep_archs` for default-arch repointing: prefer 3+ digit gfx
    archs (specific archs like `gfx1103`) over 2-digit family prefixes
    (`gfx11`). Within each tier the user's keep-list order is preserved.

    Different aggregators name device packages differently — `rocm` sdist
    has per-arch packages (`rocm-sdk-device-gfx1103`), `torch` aggregates
    families (`amd-torch-device-gfx11`). The repoint target must exist in
    the file being processed; specific-arch first / family fallback gives
    the best chance of finding a real package.
    """

    def key(arch: str) -> tuple[int, int]:
        m = _GFX_ARCH_DIGITS_RE.match(arch)
        ndigits = len(m.group(1)) if m else 0
        return (0 if ndigits >= 3 else 1, keep_archs.index(arch))

    return sorted(keep_archs, key=key)


def _apply_keep_arch_list_to_metadata(
    path: pathlib.Path, keep_archs: list[str]
) -> None:
    """Drop `Provides-Extra: device-gfx<N>` and `Requires-Dist: *-gfx<N>` lines
    for archs not in `keep_archs`. No-op on single-arch metadata.

    The `Requires-Dist: ... ; extra == "device"` line names the package's
    *default* arch. If that arch is dropped, the line is repointed at the
    keep[0] package rather than removed, so the [device] extra still has a
    dep.

    Handles METADATA and PKG-INFO.
    """
    if not path.exists():
        return
    found_archs = _scan_multiarch_metadata(path)
    if not found_archs:
        print(f"      single-arch metadata (no device-* markers): {path} (no-op)")
        return
    _assert_keep_overlaps_found(str(path), found_archs, keep_archs)
    print(f"      apply keep list {keep_archs} to {path}")

    provides_re = re.compile(rf"^Provides-Extra:\s*device-({_GFX_ARCH})\s*$")
    # Captures full dep name (group 1) and its arch (group 2),
    # e.g. "rocm-sdk-device-gfx1010" / "gfx1010".
    requires_re = re.compile(rf"^Requires-Dist:\s*([A-Za-z0-9_.-]*-({_GFX_ARCH}))\b")
    extra_re = re.compile(r'extra\s*==\s*"([^"]+)"')

    keep_set = set(keep_archs)

    # Pass 1: filter lines, remember the package name for each kept arch, and
    # stash the index of the `extra == "device"` placeholder IFF default arch
    # is dropped. Setuptools emits at most one such line per package; a second
    # is treated as malformed input.
    new_lines: list[str] = []
    default_device_idx: int | None = None
    keep_pkg_names: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.rstrip("\n")

            # `Provides-Extra: device-gfx<N>` -> keep iff arch is kept.
            provides_match = provides_re.match(stripped)
            if provides_match:
                if provides_match.group(1) in keep_set:
                    new_lines.append(line)
                continue

            # `Requires-Dist: <pkg>-gfx<N>...` -> filter on arch.
            requires_match = requires_re.match(stripped)
            if not requires_match:
                # Unrelated line
                new_lines.append(line)
                continue

            pkg_name = requires_match.group(1)
            arch = requires_match.group(2)

            # Capture each kept arch's package name on first sighting, so
            # pass 2 can pick a repoint target that actually exists in this
            # metadata (e.g. `rocm` sdist has no `gfx11` family wheel, only
            # specific archs like `gfx1103`).
            if arch in keep_set and arch not in keep_pkg_names:
                keep_pkg_names[arch] = pkg_name

            extra_match = extra_re.search(stripped)
            qualifier = extra_match.group(1) if extra_match else None

            # `extra == "device"` is the default-arch dep — always kept.
            # Save its index for pass 2 IFF its arch is dropped.
            if qualifier == "device":
                if arch not in keep_set:
                    if default_device_idx is not None:
                        raise ValueError(
                            f"{path}: multiple `Requires-Dist: ... ; "
                            f'extra == "device"` lines found; expected at most one'
                        )
                    default_device_idx = len(new_lines)
                new_lines.append(line)
                continue

            # `extra == "device-gfx<N>"` ties this dep to a specific arch's
            # extra. Drop the line if that arch was cut, regardless of the
            # package arch (e.g. `amd-torch-device-gfx11` for the `gfx1100`
            # extra must go when gfx1100 is not in the keep list).
            if qualifier and qualifier.startswith("device-"):
                qualifier_arch = qualifier[len("device-") :]
                if (
                    re.fullmatch(_GFX_ARCH, qualifier_arch)
                    and qualifier_arch not in keep_set
                ):
                    continue

            # Other Requires-Dist with gfx<N>: keep iff arch is in keep_set.
            if arch in keep_set:
                new_lines.append(line)

    # Pass 2: IFF the default arch got dropped, repoint at the highest-priority
    # keep arch whose package actually appears in this file.
    if default_device_idx is not None:
        chosen_arch = next(
            (a for a in _repoint_priority(keep_archs) if a in keep_pkg_names),
            None,
        )
        if chosen_arch is None:
            raise ValueError(
                f'{path}: cannot repoint `extra == "device"` — no Requires-Dist '
                f"line found for any keep arch {keep_archs}"
            )
        new_pkg_name = keep_pkg_names[chosen_arch]
        old = new_lines[default_device_idx]
        old_pkg = requires_re.match(old.rstrip("\n")).group(1)
        new_lines[default_device_idx] = old.replace(old_pkg, new_pkg_name, 1)
        print(
            f'      extra == "device" default {old_pkg!r} not in keep list; '
            f"repointing at {new_pkg_name!r} (arch={chosen_arch})"
        )

    with path.open("w", encoding="utf-8") as f:
        f.writelines(new_lines)


def _scan_multiarch_requires_txt(path: pathlib.Path) -> set[str]:
    """Same idea as `_scan_multiarch_metadata`, but for setuptools-style
    requires.txt: a `[device-all]` or `[device-gfx<N>]` section header marks
    the file as multi-arch.
    """
    text = path.read_text(encoding="utf-8")
    section_re = re.compile(r"^\[(.+)\]\s*$", re.MULTILINE)
    dep_re = re.compile(rf"^[A-Za-z0-9_.-]*-({_GFX_ARCH})\b", re.MULTILINE)
    is_multi = False
    archs: set[str] = set()
    for m in section_re.finditer(text):
        name = m.group(1)
        if name == "device-all":
            is_multi = True
        elif name.startswith("device-"):
            arch_token = name[len("device-") :]
            if re.fullmatch(_GFX_ARCH, arch_token):
                is_multi = True
                archs.add(arch_token)
    if not is_multi:
        return set()
    archs.update(m.group(1) for m in dep_re.finditer(text))
    return archs


def _apply_keep_list_to_requires_txt(path: pathlib.Path, keep_archs: list[str]) -> None:
    """Drop `[device-gfx<N>]` sections and `*-gfx<N>` deps for non-kept archs.
    No-op on single-arch requires.txt.

    The bare `[device]` section names the package's *default* arch. If that
    arch is dropped, the section's body is repointed at the keep[0] body line
    rather than emptied.

    Section layout (setuptools requires.txt):
        [device]            -> single dep, the default arch
        [device-all]        -> one dep per arch
        [device-gfx<N>]     -> single dep, that specific arch
    """
    if not path.exists():
        return
    found_archs = _scan_multiarch_requires_txt(path)
    if not found_archs:
        print(
            f"      single-arch requires.txt (no [device-*] sections): {path} (no-op)"
        )
        return
    _assert_keep_overlaps_found(str(path), found_archs, keep_archs)
    print(f"      apply keep list {keep_archs} to {path}")

    # Section header: [<name>].
    section_re = re.compile(r"^\[(.+)\]\s*$")
    # Body dep line: <package>-gfx<N>==<version>. Captures the arch.
    dep_re = re.compile(rf"^[A-Za-z0-9_.-]*-({_GFX_ARCH})\b")

    keep_set = set(keep_archs)

    # Pass 1: filter sections/deps; remember each kept arch's [device-gfx<N>]
    # body line and the index of the [device] body line so pass 2 can repoint
    # if the default arch is dropped. Each `[device*]` section is expected to
    # carry at most one body line.
    new_lines: list[str] = []
    section_name = ""
    base_section = ""
    skip_section = False
    default_device_idx: int | None = None
    keep_body_lines: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.rstrip("\n")

            # Section header. Drop the entire `[device-gfx<N>]` block if its
            # arch is not kept; everything else keeps its header.
            section_match = section_re.match(stripped)
            if section_match:
                section_name = section_match.group(1)
                # A section can carry an environment marker, e.g.
                # `[device-gfx942:sys_platform == "linux"]` for a linux-only arch
                # (its plain `[device-gfx942]` section is empty and the real dep
                # lives under the marked one). Strip the marker so the arch is
                # recognized for both the skip decision and the body-line pass.
                base_section = section_name.split(":", 1)[0].strip()
                if base_section.startswith("device-") and base_section != "device-all":
                    arch_token = base_section[len("device-") :]
                    skip_section = (
                        re.fullmatch(_GFX_ARCH, arch_token) is not None
                        and arch_token not in keep_set
                    )
                else:
                    skip_section = False
                if not skip_section:
                    new_lines.append(line)
                continue

            if skip_section:
                continue

            # Body line. Dep lines not containing and gfx arch.
            dep_match = dep_re.match(stripped)
            if not dep_match:
                new_lines.append(line)
                continue

            arch = dep_match.group(1)

            # Remember each kept arch's [device-gfx<N>] body for pass 2.
            if base_section.startswith("device-") and base_section != "device-all":
                section_arch = base_section[len("device-") :]
                if section_arch in keep_set:
                    if section_arch in keep_body_lines:
                        raise ValueError(
                            f"{path}: multiple body lines in "
                            f"[{section_name}]; expected at most one"
                        )
                    keep_body_lines[section_arch] = line

            # `[device]` body is the default-arch dep — always kept.
            # Save its index for pass 2 IFF its arch is dropped.
            if base_section == "device":
                if arch not in keep_set:
                    if default_device_idx is not None:
                        raise ValueError(
                            f"{path}: multiple body lines in [device]; "
                            f"expected at most one"
                        )
                    default_device_idx = len(new_lines)
                new_lines.append(line)
                continue

            # Other body deps with gfx<N>: keep iff arch is in keep_set.
            # (e.g. [device-all] body for a non-kept arch -> drop.)
            if arch in keep_set:
                new_lines.append(line)

    # Pass 2: IFF the default arch got dropped, repoint at the highest-priority
    # keep arch whose [device-gfx<N>] section actually appears in this file.
    if default_device_idx is not None:
        chosen_arch = next(
            (a for a in _repoint_priority(keep_archs) if a in keep_body_lines),
            None,
        )
        if chosen_arch is None:
            raise ValueError(
                f"{path}: cannot repoint [device] — no [device-gfx<N>] body "
                f"line found for any keep arch {keep_archs}"
            )
        new_body = keep_body_lines[chosen_arch]
        old = new_lines[default_device_idx]
        new_lines[default_device_idx] = new_body
        print(
            f"      [device] default {old.strip()!r} not in keep list; "
            f"repointing at {new_body.strip()!r} (arch={chosen_arch})"
        )
    with path.open("w", encoding="utf-8") as f:
        f.writelines(new_lines)


def _scan_multiarch_dist_info_py(path: pathlib.Path) -> set[str]:
    """Multi-arch iff `AVAILABLE_TARGET_FAMILIES.append(...)` references more
    than one distinct arch. Returns the set of appended archs. Duplicate
    appends of the same arch count once, so a single distinct arch (even if
    appended multiple times) is treated as single-arch.
    """
    text = path.read_text(encoding="utf-8")
    append_re = re.compile(
        rf"^\s*AVAILABLE_TARGET_FAMILIES\.append\(['\"]({_GFX_ARCH})['\"]\)\s*$",
        re.MULTILINE,
    )
    archs = {m.group(1) for m in append_re.finditer(text)}
    if len(archs) <= 1:
        return set()
    return archs


def _apply_keep_list_to_dist_info_py(path: pathlib.Path, keep_archs: list[str]) -> None:
    """Drop `AVAILABLE_TARGET_FAMILIES.append('gfx<N>')` lines for non-kept
    archs. If `DEFAULT_TARGET_FAMILY` points to a dropped arch, repoint it at
    the first entry of `keep_archs`. No-op on single-arch _dist_info.py.
    """
    if not path.exists():
        return
    found_archs = _scan_multiarch_dist_info_py(path)
    if not found_archs:
        print(
            f"      single-arch _dist_info.py (<=1 AVAILABLE_TARGET_FAMILIES.append): {path} (no-op)"
        )
        return
    _assert_keep_overlaps_found(str(path), found_archs, keep_archs)
    print(f"      apply keep list {keep_archs} to {path}")
    append_re = re.compile(
        rf"^AVAILABLE_TARGET_FAMILIES\.append\(['\"]({_GFX_ARCH})['\"]\)\s*$"
    )
    default_re = re.compile(
        rf"^DEFAULT_TARGET_FAMILY\s*=\s*['\"]({_GFX_ARCH})['\"]\s*$"
    )
    keep_set = set(keep_archs)
    new_default = next(
        (a for a in _repoint_priority(keep_archs) if a in found_archs),
        None,
    )
    if new_default is None:
        raise ValueError(
            f"{path}: cannot repoint DEFAULT_TARGET_FAMILY — no "
            f"AVAILABLE_TARGET_FAMILIES.append entry for any keep arch {keep_archs}"
        )
    new_lines = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.rstrip("\n")
            append_match = append_re.match(stripped)
            if append_match:
                if append_match.group(1) in keep_set:
                    new_lines.append(line)
                continue
            default_match = default_re.match(stripped)
            if default_match and default_match.group(1) not in keep_set:
                print(
                    f"      DEFAULT_TARGET_FAMILY {default_match.group(1)!r} "
                    f"not in keep list; repointing at {new_default!r}"
                )
                new_lines.append(f'DEFAULT_TARGET_FAMILY = "{new_default}"\n')
                continue
            new_lines.append(line)
    with path.open("w", encoding="utf-8") as f:
        f.writelines(new_lines)


# Wheels whose metadata can carry per-gfx references. Per-arch wheels (e.g.
# `rocm_sdk_device_gfx1010`) are skipped earlier in the main loop based on
# filename, so they never reach this pass; this list is the set of *aggregators*
# whose Provides-Extra/Requires-Dist or AVAILABLE_TARGET_FAMILIES entries we
# may need to trim.
GFX_AWARE_WHEEL_PREFIXES = (
    "rocm-",  # tar.gz sdist
    "rocm_sdk_",  # core, libraries, devel
    "rocm_profiler-",
    "torch-",
    "torchvision-",
)


def wheel_apply_gfx_keep_list(
    new_dir_path: pathlib.Path, keep_archs: list[str]
) -> None:
    """Apply positive-list arch filtering inside an unpacked wheel/sdist dir.

    Only runs on wheels listed in `GFX_AWARE_WHEEL_PREFIXES`; everything else
    (apex, triton, jaxlib, ...) is a no-op early-exit. Within a matched wheel,
    multi-arch METADATA/PKG-INFO/requires.txt/_dist_info.py files are rewritten
    to retain only `keep_archs`; single-arch files are left untouched.
    """
    if not keep_archs:
        return
    # Skip all wheels that do not contain metadata of multiple gfx archs, that includes device-specific wheels,
    # rocm_boostrap, triton, torchaudio, apex, ...
    if (
        not any(new_dir_path.name.startswith(p) for p in GFX_AWARE_WHEEL_PREFIXES)
        or "gfx" in new_dir_path.name
    ):
        print(
            f"    Skipping gfx-keep pass: {new_dir_path.name} has no gfx-arch metadata"
        )
        return
    print(f"    Applying gfx keep list: {keep_archs}")
    # METADATA: aggregator wheels keep it at <root>/<dist>-<version>.dist-info/METADATA
    for metadata in new_dir_path.glob("*.dist-info/METADATA"):
        _apply_keep_arch_list_to_metadata(metadata, keep_archs)
    # PKG-INFO: rocm sdist puts one at the root and one in src/<pkg>.egg-info/.
    # Both must be processed — egg-info is static, not regenerated by our pipeline.
    for pkg_info in new_dir_path.glob("PKG-INFO"):
        _apply_keep_arch_list_to_metadata(pkg_info, keep_archs)

    if new_dir_path.name.startswith("rocm-"):
        for pkg_info in new_dir_path.glob("src/rocm.egg-info/PKG-INFO"):
            _apply_keep_arch_list_to_metadata(pkg_info, keep_archs)
        # requires.txt: setuptools-style extras, only present in the rocm sdist.
        for requires in new_dir_path.glob("src/rocm.egg-info/requires.txt"):
            _apply_keep_list_to_requires_txt(requires, keep_archs)
        for dist_info_py in new_dir_path.glob("src/rocm_sdk/_dist_info.py"):
            _apply_keep_list_to_dist_info_py(dist_info_py, keep_archs)
    # _dist_info.py: <root>/<module>/_dist_info.py for SDK wheels (rocm_sdk_*,
    # rocm_profiler)
    if new_dir_path.name.startswith("rocm_"):
        for dist_info_py in new_dir_path.glob("*/_dist_info.py"):
            _apply_keep_list_to_dist_info_py(dist_info_py, keep_archs)


def wheel_change_extra_files(
    new_dir_path: pathlib.Path,
    old_version: Version,
    new_version: Version,
    multi_arch_targets: list[str] | None = None,
) -> None:
    # Always run the keep-list pass when archs are requested; do this *before*
    # version rewrites so the version replacement sees a consistent file
    # (and so we also cover --skip-version-promotion mode where
    # old_version == new_version).
    if multi_arch_targets:
        wheel_apply_gfx_keep_list(new_dir_path, multi_arch_targets)

    if old_version == new_version:
        # No version change requested — nothing else to do here.
        print("    No version change requested; skipping version-string updates")
        return

    # extract "rocm_sdk_core" from /tmp/tmp3swrl25j/wheel/rocm_sdk_core-7.10.0
    package_name_no_version = new_dir_path.name.split(str(new_version))[0][:-1]

    # correct capitalization and hyphenation
    # of interest for amdgpu arch: wheels are all lower case
    # (e.g.  rocm_sdk_libraries_gfx94x_dcgpu-7.10.0rc1-py3-none-linux_x86_64.whl)
    # but inside we have to match to rocm_sdk_libraries_gfx94X-dcgpu/ with a capital "X-dcgpu" instead of "x_dcgpu"
    if "gfx" in package_name_no_version:
        files = list(new_dir_path.glob("*gfx*"))
        for file in files:
            if len(file.name) == len(package_name_no_version):
                package_name_no_version = file.name

    old_rocm_version = (
        str(old_version)
        if "rocm" not in str(old_version)
        else str(old_version).split("+rocm")[-1]
    )
    new_rocm_version = (
        str(new_version)
        if "rocm" not in str(new_version)
        else str(new_version).split("+rocm")[-1]
    )

    print("    Changing ROCm-specific files that contain the version")

    # Every wheel: rewrite METADATA Summary / Requires-Dist rocm refs.
    # No-op when no rocm refs are present (e.g. triton, apex)
    update_metadata_rocm_requires_dist(
        new_dir_path,
        package_name_no_version,
        old_version,
        old_rocm_version,
        new_rocm_version,
    )
    # Per-arch device wheels (amd_torch_device_gfx, amd_torchvision_device_gfx,
    # rocm_sdk_device_gfx) carry no package-specific files that reference the
    # version, but they DO have METADATA Requires-Dist lines pinning a sister
    # rocm-* package to the build's rocm version. Rewrite those, then return.
    if "device_gfx" in new_dir_path.name:
        return
    if (
        "jax_rocm7_plugin" in package_name_no_version
        or "jax_rocm7_pjrt" in package_name_no_version
    ):
        return

    # rocm packages needing extra handling
    if new_dir_path.name.startswith("rocm"):
        files_to_change = [
            new_dir_path / package_name_no_version / "_dist_info.py",
        ]

        if new_dir_path.name.startswith("rocm_sdk_core"):
            files_to_change.append(
                new_dir_path
                / "_rocm_sdk_core"
                / "share"
                / "therock"
                / "therock_manifest.json"
            )
    # only torch and NOT triton, torchaudio, torchvision
    elif "torch" == package_name_no_version:
        files_to_change = [
            new_dir_path / package_name_no_version / "_rocm_init.py",
            new_dir_path / package_name_no_version / "version.py",
        ]
        _update_torch_elf_runpath_versions(
            new_dir_path / package_name_no_version,
            old_rocm_version,
            new_rocm_version,
        )
    elif "apex" in package_name_no_version:
        files_to_change = [
            new_dir_path / package_name_no_version / "git_version_info_installed.py",
        ]
    else:
        # we have multiple packages that have a version.py that needs updating
        need_change_version_py = ["torchaudio", "torchvision", "jaxlib"]
        if any(pkg in package_name_no_version for pkg in need_change_version_py):
            files_to_change = [
                new_dir_path / package_name_no_version / "version.py",
            ]
        else:
            # no additional (rocm-specific) files needed to be changed that contain the version
            # currently applying to: triton
            return

    for f in files_to_change:
        print(f"      {f}")
    with fileinput.input(files=files_to_change, encoding="utf-8", inplace=True) as f:
        for line in f:
            print(line.replace(old_rocm_version, new_rocm_version), end="")

    print("    ...done")


def promote_wheel(
    filename: pathlib.Path,
    src_version_type: str,
    dest_version: str = "release",
    multi_arch_targets: list[str] | None = None,
    skip_version_promotion: bool = False,
) -> bool:
    print(f"Promoting whl from rc to final: {filename}")

    original_wheel = Wheel(filename)
    original_version = Version(original_wheel.version)

    print(f"  Detected version: {original_version}")

    # Bound callback matching change_wheel_version's expected signature:
    #   callback(new_dir_path: Path, old_version: Version, new_version: Version) -> None
    callback = functools.partial(
        wheel_change_extra_files, multi_arch_targets=multi_arch_targets
    )

    if skip_version_promotion:
        # No version change; just repack with the keep-list callback applied.
        # `multi_arch_targets` presence is enforced at the main() boundary.
        print(
            f"  --skip-version-promotion: applying keep list {multi_arch_targets}, "
            f"version {original_version} unchanged"
        )
        new_wheel_path = change_wheel_version.change_wheel_version(
            filename,
            version=None,
            local_version=None,
            allow_same_version=True,
            callback_func=callback,
        )
        print(f"Repacked wheel at {new_wheel_path}")
        return True

    if original_version.local:  # torch packages
        if src_version_type not in original_version.local:
            print(
                f"  [ERROR] Only prerelease versions of type '{src_version_type}' can be promoted! Skipping!"
            )
            return False
        new_local_version = compute_new_version_str(
            str(original_version.local), src_version_type, dest_version
        )
        new_base_version = str(original_version.public)
    else:  # rocm packages
        if src_version_type not in str(original_version):
            print(
                f"  [ERROR] Only prerelease versions of type '{src_version_type}' can be promoted! Skipping!"
            )
            return False
        new_local_version = None
        # For non-local versions, transform the public version directly.
        new_base_version = compute_new_version_str(
            str(original_version), src_version_type, dest_version
        )

    print(f"  New base version: {new_base_version}")
    print(f"  New local version: {new_local_version}")

    print("  Starting to execute version change")
    new_wheel_path = change_wheel_version.change_wheel_version(
        filename,
        new_base_version,
        new_local_version,
        callback_func=callback,
    )
    print("  Version change done")

    new_wheel = Wheel(new_wheel_path)
    new_version = Version(new_wheel.version)
    print(f"New wheel has {new_version} and path is {new_wheel_path}")
    return True


def promote_targz_sdist(
    filename: pathlib.Path,
    src_version_type: str,
    dest_version: str = "release",
    multi_arch_targets: list[str] | None = None,
    skip_version_promotion: bool = False,
) -> bool:
    print(f"Found tar.gz: {filename}")

    base_dir = filename.parent
    package_name = filename.name.removesuffix(".tar.gz")  # removes .tar.gz

    with tempfile.TemporaryDirectory(prefix=package_name + "-") as tmp_dir:
        print(f"  Extracting tar file to {tmp_dir}", end="")

        tmp_path = pathlib.Path(tmp_dir)

        targz = tarfile.open(filename)
        # PEP 706: refuse members with absolute paths / `..` traversal and strip
        # unsafe metadata when extracting the downloaded sdist. The `filter`
        # keyword requires Python 3.12+ (per the repo's Python baseline).
        targz.extractall(tmp_path, filter="data")
        targz.close()
        print(" ...done")

        with (tmp_path / package_name / "PKG-INFO").open("r") as info:
            for line in info.readlines():
                if line.startswith("Version"):
                    version = Version(line.removeprefix("Version:").strip())

        assert version, f"No version found in {filename}/PKG-INFO."

        print(f"  Detected version: {version}")

        if skip_version_promotion:
            # `multi_arch_targets` presence is enforced at the main() boundary.
            new_version_str = str(version)
        else:
            if src_version_type not in str(version):
                print(
                    f"  [ERROR] Only prerelease versions of type '{src_version_type}' can be promoted! Skipping!"
                )
                return False
            new_version_str = compute_new_version_str(
                str(version), src_version_type, dest_version
            )

        if multi_arch_targets:
            wheel_apply_gfx_keep_list(tmp_path / package_name, multi_arch_targets)

        if new_version_str != str(version):
            print(
                f"  Editing files to change version from {version} to {new_version_str}",
                end="",
            )

            files_to_change = [
                tmp_path / f"{package_name}" / "src" / "rocm.egg-info" / "requires.txt",
                tmp_path / f"{package_name}" / "src" / "rocm.egg-info" / "PKG-INFO",
                tmp_path / f"{package_name}" / "src" / "rocm_sdk" / "_dist_info.py",
                tmp_path / f"{package_name}" / "PKG-INFO",
            ]

            with fileinput.input(
                files=files_to_change, encoding="utf-8", inplace=True
            ) as f:
                for line in f:
                    print(line.replace(str(version), new_version_str), end="")

            print(" ...done")

        print("  Creating new archive for it", end="")
        # Rename temporary directory to package name with promoted version
        package_name_no_version = package_name.removesuffix(str(version))
        new_archive_name = package_name_no_version + new_version_str
        if new_archive_name != package_name:
            (tmp_path / package_name).rename(tmp_path / new_archive_name)

        print(f" {new_archive_name}", end="")

        with tarfile.open(f"{base_dir}/{new_archive_name}.tar.gz", "w|gz") as tar:
            tar.add(tmp_path / f"{new_archive_name}", arcname=new_archive_name)

        print(" ...done")
        print(
            f"\nRepacked {package_name} as release {base_dir}/{new_archive_name}.tar.gz"
        )
        return True


def promote_targz_tarball(
    filename: pathlib.Path,
    delete: bool,
    src_version_type: str,
    dest_version: str = "release",
) -> bool:
    old_name = filename.name.removesuffix(".tar.gz")
    old_version = Version(old_name.split("-")[-1])

    print(f"Promoting tarball from rc to final: {filename.name}")
    print(f"  Detected version: {old_version}")

    if src_version_type not in str(old_version):
        print(
            f"  [ERROR] Only prerelease versions of type '{src_version_type}' can be promoted! Skipping!"
        )
        return False

    new_version_str = compute_new_version_str(
        str(old_version), src_version_type, dest_version
    )
    new_name = old_name.replace(str(old_version), new_version_str) + ".tar.gz"

    print(f"  New version: {new_version_str}")

    if delete:
        filename.rename(filename.parent / new_name)
        print(f"  Rename {filename.name} to {new_name}", end="")
    else:
        print(f"  Copy {filename.name} to {new_name}", end="")
        shutil.copy2(filename, filename.parent / new_name)
        print(" ...done")

    print(f"Repacked {filename.name} as release {filename.parent}/{new_name}")
    return True


def main(
    input_dir: pathlib.Path,
    match_files: str = "*",
    delete: bool = False,
    src_version_type: str = "rc",
    dest_version: str = "release",
    multi_arch_targets: list[str] | None = None,
    skip_version_promotion: bool = False,
) -> None:
    if skip_version_promotion and not multi_arch_targets:
        raise ValueError(
            "skip_version_promotion=True requires a non-empty multi_arch_targets"
        )

    print(f"Looking for .whl and .tar.gz in {input_dir}/{match_files}")

    # Materialize the glob: promotion renames files in place, and a lazy
    # iterator would pick up the newly created destination wheels mid-loop.
    files = sorted(input_dir.glob(match_files))
    keep_set = set(multi_arch_targets) if multi_arch_targets else None

    for file in files:
        print("")
        if file.is_dir():
            print(f"Skipping directory: {file}")
            continue
        if file.name.startswith("rocm_bootstrap"):
            print(f"Skipping rocm_bootstrap file: {file}")
            continue
        # Per-arch wheel handling: `device_gfx<N>` in the filename marks this
        # a multi-arch wheel (e.g. `rocm_sdk_device_gfx1010-...whl`).
        # Keep it iff arch is in the keep list; otherwise skip
        #  (and delete with --delete-old-on-success). Bare
        # `gfx<N>` (no `device_`) denotes a single-arch package (e.g.
        # `rocm_sdk_libraries_gfx94x_dcgpu-...whl`) and falls through untouched.
        #
        # The arch token must be matched exactly: a substring check would let
        # `gfx11` match `device_gfx1153`. Anchor on the trailing `-` (always
        # present in wheel filenames between the arch and the version).
        if keep_set is not None and "device_gfx" in file.name:
            m = re.search(rf"device_({_GFX_ARCH})-", file.name)
            file_arch = m.group(1) if m else None
            if file_arch is not None and file_arch not in keep_set:
                print(f"Skipping per-gfx wheel for non-kept arch: {file.name}")
                if delete:
                    print(f"Removing original per-gfx wheel: {file}")
                    file.unlink()
                continue
        if file.suffix == ".whl":
            if (
                promote_wheel(
                    file,
                    src_version_type,
                    dest_version=dest_version,
                    multi_arch_targets=multi_arch_targets,
                    skip_version_promotion=skip_version_promotion,
                )
                and delete
            ):
                print(f"Removing old wheel: {file}")
                file.unlink()
        elif file.suffixes[-1] == ".gz" and file.suffixes[-2] == ".tar":
            if file.name.startswith("therock-dist"):
                promote_targz_tarball(
                    file, delete, src_version_type, dest_version=dest_version
                )
            else:
                if (
                    promote_targz_sdist(
                        file,
                        src_version_type,
                        dest_version=dest_version,
                        multi_arch_targets=multi_arch_targets,
                        skip_version_promotion=skip_version_promotion,
                    )
                    and delete
                ):
                    print(f"Removing old sdist .tar.gz: {file}")
                    file.unlink()
        else:
            print(f"File found that cannot be promoted: {file}")


if __name__ == "__main__":
    print("Parsing arguments", end="")
    p = parse_arguments(sys.argv[1:])
    print(" ...done")

    multi_arch_targets = (
        [a.strip() for a in p.multi_arch_targets.split(",") if a.strip()]
        if p.multi_arch_targets
        else None
    )

    main(
        p.input_dir,
        p.match_files,
        p.delete_old_on_success,
        p.src_version_type,
        dest_version=p.dest_version,
        multi_arch_targets=multi_arch_targets,
        skip_version_promotion=p.skip_version_promotion,
    )
