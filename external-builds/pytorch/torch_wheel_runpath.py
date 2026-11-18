# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Point the RUNPATH of a Linux torch wheel's ELF files at the ROCm runtime packages.

PyTorch's CMake build records its link directories in RUNPATH
(`CMAKE_INSTALL_RPATH_USE_LINK_PATH`). Against the pip ROCm SDK those are absolute
`_rocm_sdk_devel` paths from the build machine, which do not exist after install.
`import torch` hides this because `rocm_sdk.initialize_process()` preloads the ROCm
libraries by absolute path, but a native program that links or execs libtorch has
no such hook: GNU ld cannot resolve libtorch's dependencies and the loader cannot
start the program. See https://github.com/ROCm/TheRock/issues/10078.

This module rewrites each ELF file's RUNPATH to keep its `$ORIGIN`-relative entries,
drop absolute ones, and add `$ORIGIN`-relative directories of the installed runtime
packages (`core` and `libraries`, never `devel`) that hold the file's DT_NEEDED
libraries. The wheel is then repacked so RECORD matches the patched files.

Windows DLLs carry no search path, so this is a Linux-only step.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import posixpath
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

# Logical rocm_sdk packages that `torch` requires at run time (`rocm[libraries]`).
RUNTIME_PACKAGES = ("core", "libraries")

ELF_MAGIC = b"\x7fELF"


@dataclass
class RunpathChange:
    relpath: str
    old: str
    new: str


def installed_package_dir(logical_name: str) -> Path:
    """Returns the import directory of an installed rocm_sdk package."""
    from rocm_sdk._dist_info import ALL_PACKAGES, PY_PACKAGE_SUFFIX_NONCE

    package = ALL_PACKAGES[logical_name]
    if package.is_target_specific:
        py_name = f"_{package.pure_py_package_name}{PY_PACKAGE_SUFFIX_NONCE}"
    else:
        py_name = package.get_py_package_name()
    spec = importlib.util.find_spec(py_name)
    if spec is None or not spec.submodule_search_locations:
        raise ModuleNotFoundError(f"rocm_sdk package '{py_name}' is not installed")
    return Path(next(iter(spec.submodule_search_locations)))


def index_runtime_libraries(package_dirs: list[Path]) -> dict[str, str]:
    """Maps each shared library file name to its directory.

    Directories are relative to the parent of the package directories, which is the
    `site-packages` that torch is also installed into.
    """
    index: dict[str, str] = {}
    for package_dir in package_dirs:
        root = package_dir.parent
        for dirpath, _, filenames in sorted(os.walk(package_dir)):
            rel_dir = Path(dirpath).relative_to(root).as_posix()
            for name in sorted(filenames):
                if ".so" in name and name not in index:
                    index[name] = rel_dir
    return index


def is_elf(path: Path) -> bool:
    if path.is_symlink() or not path.is_file():
        return False
    with open(path, "rb") as f:
        return f.read(4) == ELF_MAGIC


def compute_runpath(
    elf_rel_dir: str,
    old_runpath: str,
    needed: list[str],
    local_names: set[str],
    runtime_index: dict[str, str],
) -> str:
    """Returns the new RUNPATH for one ELF file.

    Args:
        elf_rel_dir: Directory of the file, relative to the wheel root.
        old_runpath: Existing RUNPATH, `:`-separated.
        needed: The file's DT_NEEDED entries.
        local_names: File names in the file's own directory.
        runtime_index: Output of `index_runtime_libraries`.
    """
    entries: list[str] = []
    for entry in old_runpath.split(":"):
        if entry and not os.path.isabs(entry) and entry not in entries:
            entries.append(entry)
    for name in needed:
        if name in local_names or name not in runtime_index:
            continue
        rel = posixpath.relpath(runtime_index[name], elf_rel_dir)
        entry = f"$ORIGIN/{rel}"
        if entry not in entries:
            entries.append(entry)
    return ":".join(entries)


def _patchelf(patchelf: str, *args: str) -> str:
    return subprocess.check_output([patchelf, *args], text=True).strip()


def rewrite_tree(
    root: Path, runtime_index: dict[str, str], patchelf: str = "patchelf"
) -> list[RunpathChange]:
    """Rewrites RUNPATH on every ELF file under `root` (an unpacked wheel)."""
    changes: list[RunpathChange] = []
    for path in sorted(root.rglob("*")):
        if not is_elf(path):
            continue
        old = _patchelf(patchelf, "--print-rpath", str(path))
        needed = _patchelf(patchelf, "--print-needed", str(path)).split()
        local_names = {p.name for p in path.parent.iterdir()}
        elf_rel_dir = path.parent.relative_to(root).as_posix()
        new = compute_runpath(elf_rel_dir, old, needed, local_names, runtime_index)
        if new == old:
            continue
        if new:
            _patchelf(patchelf, "--set-rpath", new, str(path))
        else:
            _patchelf(patchelf, "--remove-rpath", str(path))
        changes.append(RunpathChange(path.relative_to(root).as_posix(), old, new))
    return changes


def extract_wheel(wheel: Path, dest: Path) -> Path:
    """Extracts `wheel` keeping file modes and returns the unpacked root.

    The layout matches `wheel unpack`, which is not used because it rejects
    wheels whose RECORD is already stale. `wheel pack` rewrites RECORD anyway.
    """
    name, version = wheel.name.split("-")[:2]
    root = dest / f"{name}-{version}"
    with zipfile.ZipFile(wheel) as zf:
        for info in zf.infolist():
            extracted = Path(zf.extract(info, root))
            mode = info.external_attr >> 16
            if mode and not info.is_dir():
                extracted.chmod(mode & 0o777)
    return root


def rewrite_wheel(wheel: Path, patchelf: str = "patchelf") -> list[RunpathChange]:
    """Rewrites RUNPATH of the ELF files in `wheel` in place."""
    runtime_index = index_runtime_libraries(
        [installed_package_dir(name) for name in RUNTIME_PACKAGES]
    )
    with tempfile.TemporaryDirectory(prefix="torch-runpath-") as td:
        work = Path(td)
        root = extract_wheel(wheel, work / "unpacked")
        changes = rewrite_tree(root, runtime_index, patchelf)
        if not changes:
            return changes
        out_dir = work / "packed"
        out_dir.mkdir()
        # `wheel pack` regenerates RECORD, which pip checks against the files.
        subprocess.check_call(
            [sys.executable, "-m", "wheel", "pack", "--dest-dir", out_dir, root]
        )
        (packed,) = out_dir.iterdir()
        if packed.name != wheel.name:
            raise RuntimeError(f"Repacked wheel renamed {wheel.name} -> {packed.name}")
        shutil.move(packed, wheel)
    return changes


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("wheel", type=Path)
    p.add_argument("--patchelf", default="patchelf")
    args = p.parse_args(argv)
    for change in rewrite_wheel(args.wheel, args.patchelf):
        print(f"  {change.relpath}\n    old: {change.old}\n    new: {change.new}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
