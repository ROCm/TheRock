# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared helpers for build_tools tests."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest import mock


@contextmanager
def intercept_file_write(
    file_path: Path, *, on_write: Callable[[], None]
) -> Iterator[None]:
    """Intercept truncating opens and replacements of a file."""
    real_open = Path.open
    real_replace = Path.replace

    def intercepted_open(path: Path, *args, **kwargs):
        opened_file = real_open(path, *args, **kwargs)
        mode = args[0] if args else kwargs.get("mode", "r")
        if path == file_path and "w" in mode:
            # The file has already been truncated when the callback runs.
            on_write()
        return opened_file

    def intercepted_replace(path: Path, target: Path) -> Path:
        if Path(target) == file_path:
            on_write()
        return real_replace(path, target)

    with (
        mock.patch.object(Path, "open", new=intercepted_open),
        mock.patch.object(Path, "replace", new=intercepted_replace),
    ):
        yield
