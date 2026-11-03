# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Locate SDK libraries without importing or initializing the ROCm runtime."""

from importlib import metadata
from pathlib import Path, PurePosixPath
import platform
import re


def locate_artifact_pair(bin_dir: Path) -> tuple[Path, Path]:
    """Select one physical compiler/HIP pair from the tested artifact prefix.

    This is deployment policy, not SDK default discovery. No system or wheel
    runtime is substituted for the artifact whose correctness is being tested.
    """
    prefix = bin_dir.parent.resolve(strict=True)
    windows = platform.system() == "Windows"
    directories = (bin_dir, prefix / "lib", prefix / "lib64")

    def select(pattern: str, kind: str) -> Path:
        paths = [
            path
            for directory in directories
            if directory.is_dir()
            for path in directory.iterdir()
            if re.fullmatch(pattern, path.name, re.IGNORECASE if windows else 0)
            and path.is_file()
        ]
        physical = {path.resolve(strict=True) for path in paths}
        # Different hard-link spellings also name one physical library.
        identities = {(path.stat().st_dev, path.stat().st_ino) for path in physical}
        if len(identities) != 1:
            raise RuntimeError(
                f"tested artifact has no unambiguous {kind} library under {prefix}"
            )
        chosen = min(paths, key=lambda path: (len(path.name), str(path))).resolve(
            strict=True
        )
        if not chosen.is_relative_to(prefix):
            raise RuntimeError(
                f"artifact {kind} points outside its installation prefix: {chosen}"
            )
        return chosen

    compiler = select(
        r"amd_comgr(?:_?\d+)?\.dll" if windows else r"libamd_comgr\.so(?:\.\d+)*",
        "COMGR",
    )
    hip = select(
        r"amdhip64(?:_?\d+)?\.dll" if windows else r"libamdhip64\.so(?:\.\d+)*", "HIP"
    )
    return compiler, hip


def locate_comgr(*, override: str | None = None) -> Path:
    """Return a physical COMGR file from an override or installed SDK RECORD.

    A nonempty override is authoritative, including when it is invalid. Relative
    overrides are resolved against the caller's current directory. Lookup does
    not load the library; callers must verify that it can be loaded and queried.
    """
    if override:
        path = Path(override).resolve(strict=True)
        if not path.is_file():
            raise RuntimeError(f"COMGR override is not a file: {override}")
        return path

    try:
        distribution = metadata.distribution("rocm-sdk-core")
    except metadata.PackageNotFoundError as e:
        raise RuntimeError(
            "rocm-sdk-core is not installed; install it or provide a COMGR override"
        ) from e

    windows = platform.system() == "Windows"
    directory = "bin" if windows else "lib"
    filename = r"amd_comgr\d*\.dll" if windows else r"libamd_comgr\.so(?:\.\d+)*"
    paths = []
    for entry in distribution.files or ():
        parts = PurePosixPath(str(entry)).parts
        if (
            len(parts) != 3
            or not re.fullmatch(r"_rocm_sdk_core(?:_.+)?", parts[0])
            or parts[1] != directory
            or not re.fullmatch(filename, parts[2])
        ):
            continue
        path = Path(distribution.locate_file(entry))
        if path.is_file():
            paths.append(path)
    if not paths:
        raise RuntimeError(
            "installed rocm-sdk-core has no COMGR payload for this platform; "
            "install a compatible SDK or provide a COMGR override"
        )
    # Prefer the shortest filename (bare library, then SONAME), matching rocKE.
    return min(paths, key=lambda path: (len(path.name), str(path))).resolve(strict=True)
