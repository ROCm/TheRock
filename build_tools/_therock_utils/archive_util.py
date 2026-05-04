# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Utilities for reading and writing zstd/xz compressed tar archives."""

from pathlib import Path
import tarfile

# Maps open_archive mode to the binary mode for ZstdFile and the tarfile mode
# string for xz archives.
_MODES = {
    "r": {"zstd": "rb", "xz": "r:xz"},
    "w": {"zstd": "wb", "xz": "x:xz"},
}

_DEFAULT_LEVELS = {"zstd": 3, "xz": 6}


def get_pyzstd():
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
        pyzstd = get_pyzstd()
        self._zstd_file = pyzstd.ZstdFile(path, mode=mode, **zstd_kwargs)
        # "rb" -> "r", "wb" -> "w"
        super().__init__(fileobj=self._zstd_file, mode=mode[0])

    def close(self) -> None:
        super().close()
        self._zstd_file.close()


def _infer_compression_type(path: Path) -> str:
    name = path.name
    if name.endswith(".tar.zst"):
        return "zstd"
    elif name.endswith(".tar.xz"):
        return "xz"
    raise ValueError(f"Cannot infer compression type from: {path}")


def open_archive(
    path: Path,
    mode: str = "r",
    *,
    compression_type: str | None = None,
    compression_level: int | None = None,
) -> tarfile.TarFile:
    """Open a tar archive for reading or writing.

    Args:
        path: Path to the archive file.
        mode: "r" for reading, "w" for writing.
        compression_type: "zstd" or "xz". Inferred from the file extension
            if not provided.
        compression_level: Compression level (write only). Defaults to 3 for
            zstd, 6 for xz.
    """
    if compression_type is None:
        compression_type = _infer_compression_type(path)

    mode_map = _MODES.get(mode)
    if mode_map is None:
        raise ValueError(f"Unsupported mode: {mode!r} (expected 'r' or 'w')")
    if compression_type not in mode_map:
        raise ValueError(f"Unknown compression type: {compression_type!r}")

    if compression_type == "zstd":
        kwargs = {}
        if compression_level is not None:
            kwargs["level_or_option"] = compression_level
        elif mode == "w":
            kwargs["level_or_option"] = _DEFAULT_LEVELS["zstd"]
        return ZstdTarFile(path, mode_map["zstd"], **kwargs)
    else:
        level = (
            compression_level
            if compression_level is not None
            else _DEFAULT_LEVELS[compression_type]
        )
        if mode == "r":
            return tarfile.TarFile.open(path, mode=mode_map["xz"])
        else:
            return tarfile.TarFile.open(path, mode=mode_map["xz"], preset=level)
