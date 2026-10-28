# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Locate SDK libraries without importing or initializing the ROCm runtime."""

from importlib import metadata
from pathlib import Path, PurePosixPath
import platform
import re


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
