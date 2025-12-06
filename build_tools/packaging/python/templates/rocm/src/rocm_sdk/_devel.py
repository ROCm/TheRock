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
import os
from pathlib import Path
import platform
import shutil
import tarfile

from . import _dist_info as di


def _is_windows():
    return platform.system() == "Windows"


def get_devel_root() -> Path:
    try:
        import rocm_sdk_devel
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "ROCm SDK development package is not installed. This can typically be "
            "obtained by installing `rocm[devel]` from your package manager"
        ) from e
    rocm_sdk_devel_path = _get_package_path(rocm_sdk_devel)
    if rocm_sdk_devel_path is None:
        raise ModuleNotFoundError(
            "rocm_sdk_devel expected to be defined by an __init__.py file"
        )
    site_lib_path = rocm_sdk_devel_path.parent
    devel_py_pkg_name = di.ALL_PACKAGES["devel"].get_py_package_name()
    devel_py_pkg_path = site_lib_path / devel_py_pkg_name
    if (devel_py_pkg_path / "__init__.py").exists():
        return devel_py_pkg_path

    _expand_devel_contents(rocm_sdk_devel_path, site_lib_path)
    if not (devel_py_pkg_path / "__init__.py").exists():
        raise ImportError(
            f"Expanding {devel_py_pkg_name} did not produce a valid Python package"
        )
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
            "install the `rocm` in a virtual environment."
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
    tarfile_path = rocm_sdk_devel_path / "_devel.tar.xz"
    if tarfile_path.exists():
        tarfile_mode = "r:xz"
    else:
        tarfile_path = rocm_sdk_devel_path / "_devel.tar"
        if not tarfile_path.exists():
            raise ImportError(
                f"Expected to find _devel.tar or _devel.tar.xz in {rocm_sdk_devel_path}"
            )
        tarfile_mode = "r"

    dist_file_path_names = [str(df) for df in dist_files]
    _lock_and_expand(
        site_lib_path,
        tarfile_path,
        tarfile_mode,
        record_path,
        dist_file_path_names,
    )


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
                        if _is_windows() and ti.issym():
                            # Convert symlinks into hardlinks on Windows.
                            # This saves disk space while improving compatibility
                            # on systems without as robust symlink support.
                            # As needed, we could also generate tarfiles with
                            # copies instead of symlinks, at the cost of disk space.
                            # Creating symlinks on Windows also requires admin privileges.
                            parent_path.mkdir(parents=True, exist_ok=True)
                            symlink_target = ti.linkname
                            hardlink_target = dest_path.parent / symlink_target
                            dest_path.hardlink_to(hardlink_target)
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
        self.original_file_size = os.path.getsize(file.name)

        if _is_windows():
            # The Windows APIs for file locking apply to only a given range
            # within the file and lock/unlock calls must be balanced. Since we
            # will be appending to the locked file, we lock as much as we know
            # about (the 'nbytes' parameter can continue beyond the end of the
            # file, but we don't know how much we'll be writing ahead of time).
            import msvcrt

            original_position = self.file.tell()
            self.file.seek(0)
            msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, self.original_file_size)
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
            msvcrt.locking(self.file.fileno(), msvcrt.LK_UNLCK, self.original_file_size)
            self.file.seek(original_position)
        else:
            import fcntl

            fcntl.lockf(self.file, fcntl.LOCK_UN)
