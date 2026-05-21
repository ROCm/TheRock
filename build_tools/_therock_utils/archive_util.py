# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Utilities for reading and writing zstd/xz compressed tar archives."""

from pathlib import Path
import tarfile


def _get_pyzstd():
    """Lazy import pyzstd with helpful error message."""
    try:
        import pyzstd

        return pyzstd
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "pyzstd is required for zstd artifact compression. "
            "Install it with: pip install pyzstd"
        )


class ZstdTarFile(tarfile.TarFile):
    """TarFile wrapper that manages the underlying ZstdFile lifetime.

    When TarFile receives a fileobj it did not open, it does not close it.
    This leaves the OS file handle open, which on Windows prevents subsequent
    os.unlink() calls from succeeding.
    """

    def __init__(self, path: Path, mode: str = "rb", **zstd_kwargs) -> None:
        pyzstd = _get_pyzstd()
        self._zstd_file = pyzstd.ZstdFile(path, mode=mode, **zstd_kwargs)

        # Trim mode from ZstdFile format to TarFile format
        #   * https://pyzstd.readthedocs.io/en/stable/pyzstd.html#open
        #   * https://docs.python.org/3/library/tarfile.html#tarfile.open
        # "rb" -> "r", "wb" -> "w"
        mode_tarfile = mode[0]

        super().__init__(fileobj=self._zstd_file, mode=mode_tarfile)

    def close(self) -> None:
        super().close()
        self._zstd_file.close()


def open_archive_for_read(path: Path) -> tarfile.TarFile:
    """Open a tar archive for reading, auto-detecting compression from extension."""
    if path.name.endswith(".tar.zst"):
        return ZstdTarFile(path, mode="rb")
    elif path.name.endswith(".tar.xz"):
        return tarfile.TarFile.open(path, mode="r:xz")
    else:
        raise ValueError(f"Unknown archive format: {path}")


def open_archive_for_write(
    path: Path, compression_type: str, compression_level: int | None = None
) -> tarfile.TarFile:
    """Open a tar archive for writing with the specified compression."""
    if compression_type == "zstd":
        level = compression_level if compression_level is not None else 3
        return ZstdTarFile(path, "wb", level_or_option=level)
    elif compression_type == "xz":
        level = compression_level if compression_level is not None else 6
        return tarfile.TarFile.open(path, mode="x:xz", preset=level)
    else:
        raise ValueError(f"Unknown compression type: {compression_type}")
