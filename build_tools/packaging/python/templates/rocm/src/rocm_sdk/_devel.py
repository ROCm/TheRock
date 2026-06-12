# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Manages the rocm-sdk-devel package.

The devel package is special in some key ways:

* Since it contains distribution (wheel) unsafe files like symlinks, it is
  distributed under the `rocm_sdk_devel` package as a `_devel.tar` or
  `_devel.tar.xz` file that is intended to be expanded on use.
* This tarball is intended to be expanded into the site-lib directory that
  contains the ROCM distribution packages and will result in a top-level
  python package named like `_rocm_sdk_devel_linux_x86_64` that is a sibling
  to other packages like `_rocm_sdk_core_linux_x86_64`.
* For any files already contained in one of the runtime packages, a relative
  symlink to the correct sibling will be stored.
* Any files not in one of the runtime packages will be included verbatim in the
  tarball.
* RPATH setup relies on this sibling behavior and is already encoded properly
  in the runtime packages.

In order to make this work, we dynamically extend the distribution package on
use, modifying the dist-info RECORD file to include all newly expanded files in
accordance with the PyPA documentation:
  https://packaging.python.org/en/latest/specifications/recording-installed-packages/
Note that this puts us in the category of creating a self-modifying package,
which is strongly discouraged but not prohibited. We deem the tradeoff worth
it, as the alternative is to increase the package size by 2-5x and break
symlink relationships.
"""

import importlib.metadata as md
import io
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tarfile

from . import _dist_info as di


def _is_windows():
    return platform.system() == "Windows"


def get_devel_root() -> Path:
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

    # Skip expanding if the devel package has already been expanded fully.
    # _expand_devel_contents deletes the tarball with tarfile_path.unlink()
    # as the last step of expansion, so presence of the tarball means
    # we haven't expanded yet or expansion failed and we need to retry
    tarfile_path, _ = _find_tarfile(rocm_sdk_devel_path)
    if (devel_py_pkg_path / "__init__.py").exists() and not tarfile_path:
        # The generic devel content is expanded one-shot, but per-ISA device
        # files are owned by independently-installed rocm-sdk-device-* wheels.
        # Reconcile their links on every call so a device wheel installed after
        # the first expansion is picked up on the next `rocm-sdk init`.
        _reconcile_device_links(site_lib_path, devel_py_pkg_path, di.__version__)
        return devel_py_pkg_path

    _expand_devel_contents(rocm_sdk_devel_path, site_lib_path)
    if not (devel_py_pkg_path / "__init__.py").exists():
        raise ImportError(
            f"Expanding {devel_py_pkg_name} did not produce a valid Python package"
        )
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


def _expand_devel_contents(rocm_sdk_devel_path: Path, site_lib_path: Path):
    # Resolve the Python package to its distribution package name and find the
    # RECORD file.
    dist_names = md.packages_distributions()["rocm_sdk_devel"]

    # De-duplication, preserving order (handles purelib/platlib duplicates)
    seen_dist_names = set()
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
            "Cannot expand the `rocm[devel]` package because it was not installed "
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

    # Find the tarfile.
    tarfile_path, tarfile_mode = _find_tarfile(rocm_sdk_devel_path)
    if not tarfile_path:
        raise ImportError(
            f"Expected to find _devel.tar or _devel.tar.xz in {rocm_sdk_devel_path}"
        )

    dist_file_path_names = [str(df) for df in dist_files]
    _lock_and_expand(
        site_lib_path,
        tarfile_path,
        tarfile_mode,
        record_path,
        dist_file_path_names,
    )


def _find_tarfile(rocm_sdk_devel_path: Path):
    tarfile_path = rocm_sdk_devel_path / "_devel.tar.xz"
    if tarfile_path.exists():
        tarfile_mode = "r:xz"
    else:
        tarfile_path = rocm_sdk_devel_path / "_devel.tar"
        if tarfile_path.exists():
            tarfile_mode = "r"
        else:
            return "", ""
    return tarfile_path, tarfile_mode


def _devel_link_ok(dest_path: Path, target: str) -> bool:
    """True if dest_path already exists as a hardlink to its manifest target.

    A symlink that happens to resolve to the right inode is rejected: the
    reconciler must guarantee hardlinks, so the slow path replaces it.
    """
    if dest_path.is_symlink() or not dest_path.is_file():
        return False
    try:
        return dest_path.samefile(dest_path.parent / target)
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


def _discover_device_link_plans(site_lib_path: Path, expected_version: str):
    """Find installed rocm-sdk-device-* wheels and their devel-link manifests.

    Returns a list of (record_path, links) where links is the list of
    {"relpath", "target"} entries from that wheel's `_devel_links` manifest and
    record_path is that wheel's RECORD (so newly materialized devel links can be
    recorded against the wheel that owns the underlying device files).
    """
    # importlib.metadata caches path scans by directory mtime. On filesystems with
    # coarse mtime resolution, a device wheel installed immediately after a prior
    # scan may otherwise be missed.
    md.MetadataPathFinder.invalidate_caches()
    plans = []
    for dist in md.distributions(path=[str(site_lib_path)]):
        name = dist.metadata["Name"]
        if not name:
            continue
        if name != "rocm-sdk-device" and not name.startswith("rocm-sdk-device-"):
            continue
        # The device wheel and the SDK are version-locked. A mismatched wheel's
        # link targets may not line up with this devel tree, so skip it loudly.
        if dist.version != expected_version:
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
        links = json.loads(manifest_path.read_text()).get("links", [])
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


def _reconcile_device_links(
    site_lib_path: Path, devel_py_pkg_path: Path, expected_version: str
) -> int:
    """Mirror per-ISA device files into the expanded devel tree.

    Each installed `rocm-sdk-device-*` wheel ships a `.devel_links/<arch>.json`
    manifest listing (relpath, target) pairs. For each entry we hardlink the
    device file from the rocm-sdk-libraries overlay into the devel platform dir
    and record the new path in that device wheel's RECORD so that
    `pip uninstall rocm-sdk-device-<arch>` removes it.

    Idempotent and safe to call on every `get_devel_root()`: links already in
    place are left untouched and add nothing to RECORD. Returns the number of
    links created during this call.

    Note: the core CLI trampolines (hipcc etc., see rocm_sdk_core._cli) only
    reach `get_devel_root()` on the FIRST devel expansion, so a device wheel
    installed after that is linked by an explicit `rocm-sdk init` / `rocm-sdk
    path`, not by subsequent compiler invocations.
    """
    plans = _discover_device_link_plans(site_lib_path, expected_version)
    if not plans:
        return 0

    # Fast path: skip the lock and any RECORD rewrite only when every device file
    # is already a correct hardlink AND its wheel's RECORD cleanly owns every
    # link (present, no duplicates). The RECORD check matters because a prior run
    # can be interrupted after creating the hardlink but before writing RECORD;
    # that must still be repaired (otherwise `pip uninstall` would not prune the
    # orphaned link).
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
            created = 0
            for record_path, links in plans:
                recorded_names = []
                for link in links:
                    relpath = link["relpath"]
                    dest_path = devel_py_pkg_path / relpath
                    recorded_names.append(
                        _record_name(site_lib_path, devel_py_pkg_path, relpath)
                    )
                    if _devel_link_ok(dest_path, link["target"]):
                        continue
                    # Create the parent first so the relative ".." target can be
                    # resolved through it (the OS cannot traverse a missing dir).
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    hardlink_target = dest_path.parent / link["target"]
                    if not hardlink_target.is_file():
                        # The target ships in the same wheel as this manifest, so
                        # it should exist; skip defensively if it somehow does not.
                        continue
                    if dest_path.exists() or dest_path.is_symlink():
                        dest_path.unlink()
                    dest_path.hardlink_to(hardlink_target)
                    created += 1
                _ensure_record_entries(record_path, recorded_names)
            return created
        finally:
            file_lock.unlock()


def _lock_and_expand(
    site_lib_path: Path,
    tarfile_path: Path,
    tarfile_mode: str,
    record_path: Path,
    dist_file_path_names: set[str],
):
    # When extracting, we note the directory paths of each entry and on the first
    # access, clean it up if it is already present. This works around package manager
    # races where in certain uninstall situations, some amount of the directory tree
    # may not be fully removed (this presently happens with dangling symlinks).
    # Cleaning it ensures consistent re-install behavior.
    clean_dir_paths: set[Path] = set()

    def _clean_dir(dir: Path):
        clean_dir_paths.add(dir)
        if dir.exists():
            shutil.rmtree(dir, ignore_errors=False)

    with open(record_path, "at") as record_file:
        file_lock = FileLock(record_file)
        try:
            with tarfile.open(tarfile_path, tarfile_mode) as tf:
                while ti := tf.next():
                    dest_path = site_lib_path / ti.name
                    if ti.isfile() or ti.issym():
                        parent_path = dest_path.parent
                        if parent_path not in clean_dir_paths:
                            _clean_dir(parent_path)
                        if ti.name not in dist_file_path_names:
                            # CSV record:
                            #   path
                            #   hash (empty)
                            #   size (empty)
                            record_file.write(f"{ti.name},,\n")
                        if ti.issym():
                            # Convert file symlinks into hardlinks on all platforms.
                            # This saves disk space while improving compatibility.
                            # On Windows: symlinks require admin privileges.
                            # On Linux: native binaries that use readlink(/proc/self/exe)
                            #   to determine their location will resolve symlinks and
                            #   report the wrong path (e.g., _rocm_sdk_core instead of
                            #   _rocm_sdk_devel). Hardlinks avoid this issue.
                            # As needed, we could also generate tarfiles with
                            # copies instead of symlinks, at the cost of disk space.
                            parent_path.mkdir(parents=True, exist_ok=True)
                            symlink_target = ti.linkname
                            hardlink_target = dest_path.parent / symlink_target
                            # Only create hardlinks for files, not directories
                            if hardlink_target.is_file():
                                dest_path.hardlink_to(hardlink_target)
                            else:
                                # For directory symlinks, extract as normal
                                tf.extract(ti, path=site_lib_path)
                        else:
                            tf.extract(ti, path=site_lib_path)
                    elif ti.isdir():
                        # We don't generally have directory entries, but handle
                        # them if we do.
                        if dest_path not in clean_dir_paths:
                            _clean_dir(dest_path)
                        tf.extract(ti, path=site_lib_path)
            tarfile_path.unlink()
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
