#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Rewrite DT_RPATH on PyTorch wheels so they resolve TheRock ROCm at runtime.

CMake sets ``CMAKE_INSTALL_RPATH`` to ``$ORIGIN`` and
``CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE``. When torch is built against a
TheRock SDK, that bakes the *build machine* ``_rocm_sdk_devel/lib`` directory
into every HIP-linked ``.so`` as an absolute path (for example
``/opt/_internal/cpython-3.13.3/.../_rocm_sdk_devel/lib`` on manylinux, or a
venv path on a developer box).

Those paths do not exist after ``pip install`` into another environment, so the
dynamic loader falls through to the default search path and can load a
system-wide ROCm next to a venv ROCm.

pytorch/.ci/manywheel/repair_wheel.py rewrites this to ``$ORIGIN``-relative
``_rocm_sdk_core`` / ``_rocm_sdk_libraries`` entries, but it is only invoked
from pytorch/.ci/manywheel/build.sh. TheRock's ``build_prod_wheels.py`` (and
``setup.py`` / ``python -m build``) never call it. This module is the TheRock
equivalent: post-process the wheel in place with patchelf.

Windows is a no-op (no ELF RPATH).
"""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
from base64 import urlsafe_b64encode
from pathlib import Path

# Used when the build venv has no _rocm_sdk_* trees to scan (unit tests, or a
# wheel rebuilt without the SDK still installed). Matches the TheRock layout
# that pytorch/.ci/manywheel/repair_wheel.py hard-codes for 7.14+.
FALLBACK_RUNTIME_LIB_DIRS: tuple[str, ...] = (
    "_rocm_sdk_core/lib",
    "_rocm_sdk_core/lib/rocm_sysdeps/lib",
    "_rocm_sdk_core/lib/host-math/lib",
    "_rocm_sdk_libraries/lib",
)

# Absolute builder paths that must not survive in a shipped wheel.
FORBIDDEN_RPATH_SUBSTRINGS: tuple[str, ...] = (
    "/opt/_internal/",
    "/opt/python/",
)

_PATCHELF_CANDIDATES: tuple[str, ...] = (
    "patchelf",
    "/usr/local/bin/patchelf",
    "/usr/bin/patchelf",
)


def discover_rocm_runtime_lib_dirs(search_roots: list[Path] | None = None) -> list[str]:
    """Return site-packages-relative ROCm runtime lib dirs.

    Skips ``_rocm_sdk_devel``: that tree is a compile-time SDK and is what
    CMake baked into RUNPATH. Runtime resolution must go through core /
    libraries (including family-suffixed ``_rocm_sdk_libraries_*``).
    """
    found: list[str] = []
    seen: set[str] = set()
    for root in (
        search_roots if search_roots is not None else _default_site_package_roots()
    ):
        if not root.is_dir():
            continue
        for entry in sorted(root.iterdir()):
            if not entry.is_dir() or not entry.name.startswith("_rocm_sdk_"):
                continue
            if "devel" in entry.name:
                continue
            for rel_lib in (
                Path(entry.name) / "lib",
                Path(entry.name) / "lib" / "rocm_sysdeps" / "lib",
                Path(entry.name) / "lib" / "host-math" / "lib",
            ):
                abs_lib = root / rel_lib
                key = rel_lib.as_posix()
                if abs_lib.is_dir() and key not in seen:
                    seen.add(key)
                    found.append(key)
    if found:
        return found
    return list(FALLBACK_RUNTIME_LIB_DIRS)


def _default_site_package_roots() -> list[Path]:
    roots: list[Path] = []
    try:
        import site

        roots.extend(Path(p) for p in site.getsitepackages())
        user_site = site.getusersitepackages()
        if user_site:
            roots.append(Path(user_site))
    except Exception:
        pass
    for entry in sys.path:
        path = Path(entry)
        if path.name == "site-packages" and path.is_dir():
            roots.append(path)
    # Preserve order, drop duplicates.
    unique: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        resolved = root.resolve() if root.exists() else root
        if resolved not in seen:
            seen.add(resolved)
            unique.append(root)
    return unique


def rpath_for_shared_object(rel_so: Path, runtime_lib_dirs: list[str]) -> str:
    """Build a colon-separated RPATH for a wheel-relative shared object.

    ``rel_so`` is relative to site-packages (the wheel root), e.g.
    ``torch/lib/libtorch_hip.so`` or ``torch/_C.so``. ``$ORIGIN`` is the
    directory containing that file, so the number of ``..`` components depends
    on depth: ``torch/lib/*.so`` uses ``$ORIGIN/../../_rocm_sdk_core/lib``,
    while a top-level ``torch/_C.so`` uses ``$ORIGIN/../_rocm_sdk_core/lib``.
    """
    origin_dir = rel_so.parent
    ups = "/".join(".." for _ in origin_dir.parts)
    entries: list[str] = []
    for lib_dir in runtime_lib_dirs:
        if ups:
            entries.append(f"$ORIGIN/{ups}/{lib_dir}")
        else:
            entries.append(f"$ORIGIN/{lib_dir}")
    entries.append("$ORIGIN")
    if "lib" not in origin_dir.parts:
        entries.append("$ORIGIN/lib")
    return ":".join(entries)


def iter_shared_objects(wheel_root: Path) -> list[Path]:
    """Regular (non-symlink) ELF-candidate files under the unpacked wheel."""
    shared_objects: list[Path] = []
    for path in sorted(wheel_root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        name = path.name
        if ".so" in name or name.endswith(".so"):
            shared_objects.append(path)
    return shared_objects


def find_patchelf() -> Path:
    for candidate in _PATCHELF_CANDIDATES:
        if os.sep in candidate or (os.altsep and os.altsep in candidate):
            path = Path(candidate)
            if path.is_file() and os.access(path, os.X_OK):
                return path
        else:
            found = shutil.which(candidate)
            if found:
                return Path(found)
    raise FileNotFoundError(
        "patchelf is required to rewrite torch wheel RPATH on Linux "
        f"(looked for {_PATCHELF_CANDIDATES})"
    )


def record_digest(data: bytes) -> str:
    digest = hashlib.sha256(data).digest()
    return "sha256=" + urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def write_record(wheel_root: Path) -> None:
    dist_infos = [
        p for p in wheel_root.iterdir() if p.is_dir() and p.name.endswith(".dist-info")
    ]
    if len(dist_infos) != 1:
        raise RuntimeError(
            f"Expected exactly one *.dist-info directory in {wheel_root}, "
            f"found {[p.name for p in dist_infos]}"
        )
    record_path = dist_infos[0] / "RECORD"
    lines: list[str] = []
    for path in sorted(wheel_root.rglob("*")):
        if not path.is_file() or path == record_path:
            continue
        rel = path.relative_to(wheel_root).as_posix()
        data = path.read_bytes()
        lines.append(f"{rel},{record_digest(data)},{len(data)}")
    lines.append(f"{dist_infos[0].name}/RECORD,,")
    record_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _set_rpath(patchelf: Path, sofile: Path, rpath: str) -> None:
    subprocess.run(
        [str(patchelf), "--force-rpath", "--set-rpath", rpath, str(sofile)],
        check=True,
    )


def rewrite_unpacked_wheel(
    wheel_root: Path,
    runtime_lib_dirs: list[str] | None = None,
    *,
    patchelf: Path | None = None,
) -> list[tuple[Path, str]]:
    """Patchelf every shared object under an unpacked wheel. Returns (so, rpath)."""
    lib_dirs = (
        runtime_lib_dirs
        if runtime_lib_dirs is not None
        else discover_rocm_runtime_lib_dirs()
    )
    if patchelf is None:
        patchelf = find_patchelf()
    rewritten: list[tuple[Path, str]] = []
    for sofile in iter_shared_objects(wheel_root):
        rel_so = sofile.relative_to(wheel_root)
        rpath = rpath_for_shared_object(rel_so, lib_dirs)
        _set_rpath(patchelf, sofile, rpath)
        rewritten.append((rel_so, rpath))
    write_record(wheel_root)
    return rewritten


def _repack_wheel(wheel_root: Path, dest_wheel: Path) -> None:
    if dest_wheel.exists():
        dest_wheel.unlink()
    with zipfile.ZipFile(
        dest_wheel, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True
    ) as zf:
        for path in sorted(wheel_root.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(wheel_root).as_posix()
            info = zipfile.ZipInfo.from_file(path, rel)
            # Keep files executable when they were (HIP runtime .so, scripts).
            mode = path.stat().st_mode
            info.external_attr = (mode & 0xFFFF) << 16
            if mode & stat.S_IXUSR:
                info.external_attr |= 0o755 << 16
            with path.open("rb") as src:
                zf.writestr(info, src.read())


def rewrite_wheel_runpath(
    wheel: Path,
    runtime_lib_dirs: list[str] | None = None,
) -> list[tuple[Path, str]]:
    """Rewrite RPATH on ``wheel`` in place. No-op on Windows.

    Returns the list of (wheel-relative so, new rpath) pairs.
    """
    if sys.platform == "win32":
        print(f"Skipping RPATH rewrite on Windows: {wheel}")
        return []
    if not wheel.is_file() or wheel.suffix != ".whl":
        raise FileNotFoundError(f"Expected a .whl file, got {wheel}")

    patchelf = find_patchelf()
    lib_dirs = (
        runtime_lib_dirs
        if runtime_lib_dirs is not None
        else discover_rocm_runtime_lib_dirs()
    )
    print(f"Rewriting RPATH on {wheel.name} using {lib_dirs}")

    with tempfile.TemporaryDirectory(prefix="therock-torch-rpath-") as td:
        root = Path(td) / "unpacked"
        root.mkdir()
        with zipfile.ZipFile(wheel) as zf:
            zf.extractall(root)
        rewritten = rewrite_unpacked_wheel(root, lib_dirs, patchelf=patchelf)
        tmp_wheel = Path(td) / wheel.name
        _repack_wheel(root, tmp_wheel)
        shutil.move(str(tmp_wheel), str(wheel))

    for rel_so, rpath in rewritten:
        print(f"  {rel_so.as_posix()}: {rpath}")
    return rewritten


def rpath_contains_builder_path(rpath: str) -> bool:
    return any(token in rpath for token in FORBIDDEN_RPATH_SUBSTRINGS)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "wheel", type=Path, help="Path to a torch*.whl to rewrite in place"
    )
    args = parser.parse_args(argv)
    rewrite_wheel_runpath(args.wheel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
