# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Manages the rocm-sdk-devel package.

The devel package is special in some key ways:

* Non-link files are distributed directly under the `_rocm_sdk_devel` platform
  package so the package installer owns them.
* Wheel archives do not portably represent symlinks, so their paths and targets
  are stored in a `.devel_links/devel.json` manifest under the
  `rocm_sdk_devel` package.
* The wheel installs a top-level Python package named like
  `_rocm_sdk_devel_linux_x86_64` as a sibling to packages like
  `_rocm_sdk_core_linux_x86_64`; initialization creates its recorded links.
* For links to files already contained in a runtime package, initialization
  creates hardlinks. Directory links remain symlinks.
* RPATH setup relies on this sibling behavior and is already encoded properly
  in the runtime packages.

In order to make this work, we dynamically extend the distribution package on
use, modifying the dist-info RECORD file to include all generated links in
accordance with the PyPA documentation:
  https://packaging.python.org/en/latest/specifications/recording-installed-packages/
Note that this puts us in the category of creating a self-modifying package,
which is strongly discouraged but not prohibited. We deem the tradeoff worth
it, as the alternative is to increase the package size by 2-5x and break
symlink relationships.
"""

import importlib
import importlib.metadata as md
import io
import json
import os
from collections.abc import Callable
from pathlib import Path
import platform
import re
import sys

from . import _dist_info as di

_DEVEL_LINKS_MANIFEST = Path(".devel_links/devel.json")
_DEVEL_INITIALIZATION_LOCK = Path(".devel_links/devel.lock")


def _is_windows():
    return platform.system() == "Windows"


def get_devel_root(*, force_initialize: bool = False) -> Path:
    try:
        import rocm_sdk_devel
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "rocm_sdk_devel module for the ROCm SDK development package is not installed. "
            "This can typically be obtained by installing `rocm[devel]` from your package manager"
        ) from e
    rocm_sdk_devel_path = _get_package_path(rocm_sdk_devel)
    if rocm_sdk_devel_path is None:
        raise ModuleNotFoundError(
            "rocm_sdk_devel expected to be defined by an __init__.py file"
        )
    site_lib_path = rocm_sdk_devel_path.parent
    devel_py_pkg_name = di.ALL_PACKAGES["devel"].get_py_package_name()
    devel_py_pkg_path = site_lib_path / devel_py_pkg_name

    marker_file = rocm_sdk_devel_path / di.DEVEL_INITIALIZED
    if force_initialize or not marker_file.is_file():
        _initialize_devel_package(rocm_sdk_devel_path, site_lib_path)
    _reconcile_device_links(site_lib_path, devel_py_pkg_path, di.__version__)
    return devel_py_pkg_path


# Gets the path of a module presumed to be a package defined by an __init__.py
# file. Returns None if it is a namespace package or another kind of module.
def _get_package_path(m) -> Path | None:
    if m.__file__ is None:
        return None
    p = Path(m.__file__)
    if p.name == "__init__.py":
        return p.parent  # Directory containing __init__.py
    return None


def _load_devel_link_manifest(
    manifest_path: Path,
) -> tuple[str, list[dict[str, str]]]:
    manifest = json.loads(manifest_path.read_text())
    if "version" not in manifest:
        raise ValueError(
            f"Devel link manifest is missing required 'version' field: {manifest_path}"
        )
    if "links" not in manifest:
        raise ValueError(
            f"Devel link manifest is missing required 'links' field: {manifest_path}"
        )
    return manifest["version"], manifest["links"]


def _initialize_devel_package(rocm_sdk_devel_path: Path, site_lib_path: Path) -> None:
    # Resolve the Python package to its distribution package name and find the
    # RECORD file.
    dist_names = md.packages_distributions()["rocm_sdk_devel"]

    # De-duplication, preserving order (handles purelib/platlib duplicates)
    seen_dist_names: set[str] = set()
    dist_names_list = [
        d for d in dist_names if not (d in seen_dist_names or seen_dist_names.add(d))
    ]

    # to preserve fail-fast behavior
    assert len(dist_names_list) >= 1, (
        "No distribution candidates found for 'rocm_sdk_devel'. "
        "Ensure rocm[devel] is installed in the current environment."
    )
    # Try to find candidates until found one with files and a usable RECORD
    record_pkg_file = None
    dist_files = None
    dist_name = None

    for candidate in dist_names_list:
        candidate_files = md.files(candidate)
        if candidate_files is None:
            continue

        # Look for RECORD inside a *.dist-info directory.
        for record_pkg_file in candidate_files:
            if (
                record_pkg_file.name == "RECORD"
                and record_pkg_file.parent.name.endswith(".dist-info")
            ):
                # Found a usable candidate; set dist_name/dist_files
                dist_name = candidate
                dist_files = candidate_files
                break

        if dist_name is not None:
            break

    if dist_files is None:
        raise ImportError(
            "Cannot initialize the `rocm[devel]` package because it was not installed "
            "by a user-mode package manager and is managed by the system. Please "
            "install `rocm[devel]` in a virtual environment."
        )

    if dist_name is None or record_pkg_file is None:
        # We had files for at least one candidate, but did not find RECORD in any
        # Use the original RECORD error message with the first candidate name
        # If dist_name is None- fall back to the first name for message context
        msg_dist_name = dist_name if dist_name is not None else dist_names_list[0]
        raise ImportError(
            f"No distribution RECORD found for the `{msg_dist_name}` distribution package."
        )

    # Resolve to a physical file.
    record_path = record_pkg_file.locate()

    # Files installed from the wheel have hashes in RECORD. Links generated by
    # initialization are recorded with empty hash and size fields, so exclude
    # those entries when checking whether a manifest path conflicts with a
    # wheel-owned file.
    wheel_owned_path_names = {
        str(df) for df in dist_files if getattr(df, "hash", None) is not None
    }
    _initialize_devel_links(
        site_lib_path=site_lib_path,
        rocm_sdk_devel_path=rocm_sdk_devel_path,
        record_path=record_path,
        wheel_owned_path_names=wheel_owned_path_names,
    )


def _resolve_link_target(parent: Path, target: str) -> Path:
    """Resolves a relative link target against parent, collapsing any '..'.

    Link targets carry one '..' per component of the file's relpath, so joining
    one onto parent verbatim yields a string far longer than the file it names.
    Windows applies MAX_PATH to the path as passed, before '..' is collapsed, so
    os.stat and os.link fail on that string even when the collapsed path is well
    under the limit.

    resolve() collapses '..' by walking the path rather than lexically, which
    matters because devel link setup preserves directory symlinks: a
    lexical collapse through a symlinked ancestor names a different file than
    the OS reaches.
    """
    return (parent.resolve() / target).resolve()


def _devel_link_ok(dest_path: Path, target: str) -> bool:
    """True if dest_path already exists as a hardlink to its manifest target.

    A symlink that happens to resolve to the right inode is rejected: the
    reconciler must guarantee hardlinks, so the slow path replaces it.
    """
    if dest_path.is_symlink() or not dest_path.is_file():
        return False
    try:
        return dest_path.samefile(_resolve_link_target(dest_path.parent, target))
    except OSError:
        return False


def _record_has_entries(record_path: Path, names: list[str]) -> bool:
    """True if RECORD exists, has no duplicate rows, and lists every name.

    Returning False on a duplicate row makes the fast path fall through so
    `_ensure_record_entries` can rewrite RECORD and drop the duplicates.
    """
    if not record_path.exists():
        return False
    existing = set()
    for line in record_path.read_text().splitlines():
        if not line.strip():
            continue
        name = line.split(",", 1)[0]
        if name in existing:
            return False
        existing.add(name)
    return all(n in existing for n in names)


def _without_post_release(version: str) -> str:
    """Remove a canonical PEP 440 post-release segment from a version."""
    return re.sub(r"\.post\d+", "", version, count=1)


def _discover_device_link_plans(
    site_lib_path: Path, expected_version: str
) -> list[tuple[Path, list[dict[str, str]]]]:
    """Find installed rocm-sdk-device-* wheels and their devel-link manifests.

    Returns a list of (record_path, links) where links is the list of
    {"relpath", "target"} entries from that wheel's `_devel_links` manifest and
    record_path is that wheel's RECORD (so newly materialized devel links can be
    recorded against the wheel that owns the underlying device files).
    """
    # importlib.metadata caches path scans by directory mtime. On filesystems with
    # coarse mtime resolution, a device wheel installed immediately after a prior
    # scan may otherwise be missed.
    importlib.invalidate_caches()

    plans: list[tuple[Path, list[dict[str, str]]]] = []
    for dist in md.distributions(path=[str(site_lib_path)]):
        name = dist.metadata["Name"]
        if not name:
            continue
        if name != "rocm-sdk-device" and not name.startswith("rocm-sdk-device-"):
            continue

        # The device wheel and the SDK are version-locked except for post-release
        # segments. Other differences may indicate that the wheel's link targets
        # do not line up with this devel tree, so skip it loudly.
        if _without_post_release(dist.version) != _without_post_release(
            expected_version
        ):
            print(
                f"WARNING: skipping {name} {dist.version}: does not match "
                f"rocm-sdk {expected_version}",
                file=sys.stderr,
            )
            continue

        files = dist.files
        if not files:
            continue
        manifest_file = None
        record_file = None
        for f in files:
            if f.name == "RECORD" and f.parent.name.endswith(".dist-info"):
                record_file = f
            elif f.suffix == ".json" and f.parent.name == ".devel_links":
                manifest_file = f
        if manifest_file is None or record_file is None:
            continue
        manifest_path = Path(manifest_file.locate())
        if not manifest_path.is_file():
            continue
        _, links = _load_devel_link_manifest(manifest_path)
        if not links:
            continue
        plans.append((Path(record_file.locate()), links))
    return plans


def _ensure_record_entries(record_path: Path, names: list[str]):
    """Append RECORD entries for materialized devel links, de-duplicated.

    Reads the existing RECORD, drops any duplicate paths, appends the new
    site-packages-relative paths (with empty hash/size, per the PyPA spec), and
    rewrites the file. Writes only when there is something to add or a duplicate
    row to remove.
    """
    lines = []
    seen = set()
    changed = False
    if record_path.exists():
        for line in record_path.read_text().splitlines():
            if not line.strip():
                continue
            path0 = line.split(",", 1)[0]
            if path0 in seen:
                # Drop a duplicate row; rewriting the file removes it.
                changed = True
                continue
            seen.add(path0)
            lines.append(line)
    additions = [n for n in names if n not in seen]
    if not additions and not changed:
        return
    for n in additions:
        lines.append(f"{n},,")
    record_path.write_text("\n".join(lines) + "\n", newline="\n")


def _record_name(site_lib_path: Path, devel_py_pkg_path: Path, relpath: str) -> str:
    """Site-packages-relative RECORD path for a materialized devel link."""
    return (devel_py_pkg_path / relpath).relative_to(site_lib_path).as_posix()


def _ensure_links(
    site_lib_path: Path,
    devel_py_pkg_path: Path,
    plans: list[tuple[Path, list[dict[str, str]]]],
    *,
    should_use_symlink: Callable[[Path, str], bool] = (
        lambda _dest_path, _target: False
    ),
) -> int:
    """Create or repair links and update their owning RECORD files."""
    # Record ownership before creating links so a later failure cannot leave a
    # successfully created link unowned. Missing paths in RECORD are safe.
    for record_path, links in plans:
        recorded_names = [
            _record_name(site_lib_path, devel_py_pkg_path, link["relpath"])
            for link in links
        ]
        _ensure_record_entries(record_path, recorded_names)

    created = 0
    for _record_path, links in plans:
        for link in links:
            relpath = link["relpath"]
            dest_path = devel_py_pkg_path / relpath
            use_symlink = should_use_symlink(dest_path, link["target"])
            # Create the parent first: neither link operation creates it.
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            # Keep an existing link when its type and target already match.
            # Otherwise, validate its replacement target and select the link
            # operation.
            if use_symlink:
                if dest_path.is_symlink() and os.readlink(dest_path) == link["target"]:
                    continue
                link_target = link["target"]
                create_link = dest_path.symlink_to
            else:
                if _devel_link_ok(dest_path, link["target"]):
                    continue
                hardlink_target = _resolve_link_target(dest_path.parent, link["target"])
                if not hardlink_target.is_file():
                    raise FileNotFoundError(
                        f"Hardlink target is not a file: {hardlink_target}"
                    )
                link_target = os.fspath(hardlink_target)
                create_link = dest_path.hardlink_to
            dest_path.unlink(missing_ok=True)
            create_link(link_target)
            created += 1
    return created


def _reconcile_device_links(
    site_lib_path: Path, devel_py_pkg_path: Path, expected_version: str
) -> int:
    """Mirror per-ISA device files into the initialized devel tree.

    Each installed `rocm-sdk-device-*` wheel ships a `.devel_links/<arch>.json`
    manifest listing (relpath, target) pairs. For each entry we hardlink the
    device file from the rocm-sdk-libraries overlay into the devel platform dir
    and record the new path in that device wheel's RECORD so that
    `pip uninstall rocm-sdk-device-<arch>` removes it.

    Idempotent and safe to call on every `get_devel_root()`: links already in
    place are left untouched and add nothing to RECORD. Returns the number of
    links created during this call.

    Note: the core CLI trampolines (hipcc etc., see rocm_sdk_core._cli) only
    reach `get_devel_root()` on the FIRST devel initialization, so a device wheel
    installed after that is linked by an explicit `rocm-sdk init` / `rocm-sdk
    path`, not by subsequent compiler invocations.
    """
    plans = _discover_device_link_plans(site_lib_path, expected_version)
    if not plans:
        return 0

    # Fast path: skip the lock and any RECORD rewrite only when every device file
    # is already a correct hardlink AND its wheel's RECORD cleanly owns every
    # link (present, no duplicates). The RECORD check matters because an existing
    # hardlink can be absent from RECORD; that must still be repaired (otherwise
    # `pip uninstall` would not prune the orphaned link).
    if all(
        _devel_link_ok(devel_py_pkg_path / link["relpath"], link["target"])
        for _record_path, links in plans
        for link in links
    ) and all(
        _record_has_entries(
            record_path,
            [
                _record_name(site_lib_path, devel_py_pkg_path, link["relpath"])
                for link in links
            ],
        )
        for record_path, links in plans
    ):
        return 0

    lock_path = devel_py_pkg_path / ".devel_reconcile.lock"
    with open(lock_path, "a") as lock_file:
        file_lock = FileLock(lock_file)
        try:
            return _ensure_links(
                site_lib_path,
                devel_py_pkg_path,
                plans,
            )
        finally:
            file_lock.unlock()


def _initialize_devel_links(
    site_lib_path: Path,
    rocm_sdk_devel_path: Path,
    record_path: Path,
    wheel_owned_path_names: set[str],
) -> int:
    manifest_path = rocm_sdk_devel_path / _DEVEL_LINKS_MANIFEST
    # The manifest remains installed, so a marker file records whether
    # initialization completed successfully.
    marker_file = rocm_sdk_devel_path / di.DEVEL_INITIALIZED
    devel_py_pkg_path = site_lib_path / di.ALL_PACKAGES["devel"].get_py_package_name()
    if not manifest_path.is_file():
        raise ImportError(f"Missing devel link manifest: {manifest_path}")
    manifest_version, links = _load_devel_link_manifest(manifest_path)
    if manifest_version != di.__version__:
        raise ValueError(
            f"Devel link manifest version {manifest_version!r} does not "
            f"match installed ROCm version {di.__version__!r}"
        )
    for link in links:
        relpath = link["relpath"]
        record_name = _record_name(site_lib_path, devel_py_pkg_path, relpath)
        if record_name in wheel_owned_path_names:
            raise ValueError(
                f"Devel link manifest path is owned by the wheel: {relpath}"
            )
    marker_record_name = marker_file.relative_to(site_lib_path).as_posix()
    if marker_record_name in wheel_owned_path_names:
        raise ValueError("Devel initialization marker path is owned by the wheel")
    lock_path = rocm_sdk_devel_path / _DEVEL_INITIALIZATION_LOCK
    lock_record_name = lock_path.relative_to(site_lib_path).as_posix()
    if lock_record_name in wheel_owned_path_names:
        raise ValueError("Devel initialization lock path is owned by the wheel")

    # Materialize links to files as hardlinks. This saves space, avoids Windows
    # symlink privileges, and lets binaries observe their devel path through
    # /proc/self/exe. Preserve top-level compiler symlinks on non-Windows hosts;
    # directory and dangling links must also remain symlinks.
    PRESERVE_SYMLINKS = [
        "amdclang",
        "amdclang++",
        "amdclang-cl",
        "amdclang-cpp",
        "amdflang",
        "amdlld",
        "amdllvm",
    ]
    # Multiple processes can observe a missing marker file concurrently. Link
    # setup is idempotent and writes the marker file while holding a
    # dedicated initialization lock.
    with open(lock_path, "a") as lock_file:
        file_lock = FileLock(lock_file)
        try:
            # Record the lock before link setup so uninstall removes it even if
            # initialization fails.
            _ensure_record_entries(record_path, [lock_record_name])

            # A failed initialization must leave the marker absent so a later
            # call retries the complete operation.
            marker_file.unlink(missing_ok=True)

            created = _ensure_links(
                site_lib_path,
                devel_py_pkg_path,
                plans=[(record_path, links)],
                should_use_symlink=lambda dest_path, target: (
                    not _is_windows()
                    and dest_path.name in PRESERVE_SYMLINKS
                    and dest_path.parent.name == "bin"
                    and dest_path.parent.parent.name.startswith("_rocm_sdk_devel")
                )
                or not _resolve_link_target(dest_path.parent, target).is_file(),
            )

            # Record the generated marker so uninstall removes it with the
            # devel package.
            _ensure_record_entries(record_path, [marker_record_name])

            marker_file.touch()
            return created
        finally:
            file_lock.unlock()


class FileLock:
    """Small portability shim between fcntl.lockf and msvcrt.locking for our uses."""

    def __init__(self, file: io.TextIOWrapper):
        self.file = file
        # Lock at least one byte. On Windows, msvcrt.locking treats a zero-length
        # range as a no-op (a lock on an empty file does not block), so a lock
        # file that starts empty - like the reconcile sentinel - would not
        # actually serialize callers. Locking one byte at offset 0 is valid even
        # when the file is empty.
        self.lock_size = max(os.path.getsize(file.name), 1)

        if _is_windows():
            # The Windows APIs for file locking apply to only a given range
            # within the file and lock/unlock calls must be balanced. Since we
            # will be appending to the locked file, we lock as much as we know
            # about (the 'nbytes' parameter can continue beyond the end of the
            # file, but we don't know how much we'll be writing ahead of time).
            import msvcrt

            original_position = self.file.tell()
            self.file.seek(0)
            msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, self.lock_size)
            self.file.seek(original_position)
        else:
            # The Unix APIs for file locking apply to the entire file descriptor.
            import fcntl

            fcntl.lockf(self.file, fcntl.LOCK_EX)

    def unlock(self):
        if _is_windows():
            import msvcrt

            original_position = self.file.tell()
            self.file.seek(0)
            msvcrt.locking(self.file.fileno(), msvcrt.LK_UNLCK, self.lock_size)
            self.file.seek(original_position)
        else:
            import fcntl

            fcntl.lockf(self.file, fcntl.LOCK_UN)
