# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for rocm-sdk-devel tarball expansion.

File symlinks with an existing target become hardlinks. Any other symlink is
created with os.symlink. A Windows privilege failure must raise and leave the
tarball in place, rather than reporting a short tree as a finished expand.
"""

import io
import os
import sys
import tarfile
from pathlib import Path

import pytest

_PYTHON_DIR = Path(__file__).resolve().parent.parent
_SRC = _PYTHON_DIR / "templates" / "rocm" / "src"
_BUILD_TOOLS = _PYTHON_DIR.parents[1]
sys.path.insert(0, os.fspath(_BUILD_TOOLS))
sys.path.insert(0, os.fspath(_SRC))

from rocm_sdk._devel import DevelExpandError, _lock_and_expand


def _deny_symlink(*_args: object, **_kwargs: object) -> None:
    error = OSError(13, "A required privilege is not held by the client")
    error.winerror = 1314
    raise error


def _add_file(tf: tarfile.TarFile, name: str, payload: bytes = b"x") -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    tf.addfile(info, io.BytesIO(payload))


def _add_symlink(tf: tarfile.TarFile, name: str, linkname: str) -> None:
    info = tarfile.TarInfo(name)
    info.type = tarfile.SYMTYPE
    info.linkname = linkname
    info.size = 0
    tf.addfile(info)


def _expand(
    site: Path,
    tar_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    deny: bool,
) -> None:
    if deny:
        monkeypatch.setattr(os, "symlink", _deny_symlink)
    _lock_and_expand(
        site_lib_path=site,
        tarfile_path=tar_path,
        tarfile_mode="r",
        record_path=site.parent / "RECORD",
        dist_file_path_names=set(),
    )


def test_file_symlink_hardlink_does_not_need_symlink_privilege(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = tmp_path / "site"
    site.mkdir()
    tar_path = tmp_path / "_devel.tar"
    with tarfile.open(tar_path, "w") as tf:
        _add_file(tf, "tree/data.txt", b"hello")
        _add_symlink(tf, "tree/link.txt", "data.txt")

    _expand(site, tar_path, monkeypatch, deny=True)

    data_path = site / "tree/data.txt"
    link_path = site / "tree/link.txt"
    assert data_path.read_bytes() == b"hello"
    assert link_path.samefile(data_path)
    assert not link_path.is_symlink()
    assert not tar_path.exists()


def test_dangling_symlink_privilege_failure_keeps_tarball(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = tmp_path / "site"
    site.mkdir()
    tar_path = tmp_path / "_devel.tar"
    with tarfile.open(tar_path, "w") as tf:
        _add_file(tf, "tree/data.txt", b"hello")
        _add_symlink(tf, "tree/link.txt", "data.txt")
        _add_symlink(tf, "tree/dangling.h", "../nowhere/missing.h")
        _add_file(tf, "tree/after.txt", b"kept")

    with pytest.raises(DevelExpandError, match="Cannot create symlink"):
        _expand(site, tar_path, monkeypatch, deny=True)

    assert (site / "tree/data.txt").read_bytes() == b"hello"
    assert (site / "tree/link.txt").samefile(site / "tree/data.txt")
    assert not (site / "tree/after.txt").exists()
    assert tar_path.is_file()
    record = (tmp_path / "RECORD").read_text(encoding="utf-8")
    assert "tree/dangling.h,," not in record


def test_directory_symlink_privilege_failure_keeps_tarball(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = tmp_path / "site"
    (site / "realdir").mkdir(parents=True)
    tar_path = tmp_path / "_devel.tar"
    with tarfile.open(tar_path, "w") as tf:
        _add_file(tf, "pkg/before.txt", b"before")
        _add_symlink(tf, "pkg/dirlink", "../realdir")
        _add_file(tf, "pkg/after.txt", b"after")

    with pytest.raises(DevelExpandError, match="Cannot create symlink"):
        _expand(site, tar_path, monkeypatch, deny=True)

    assert (site / "pkg/before.txt").read_bytes() == b"before"
    assert not (site / "pkg/after.txt").exists()
    assert tar_path.is_file()


def test_dangling_symlink_is_created_when_privilege_allows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    probe = tmp_path / "probe-link"
    try:
        probe.symlink_to("missing-target")
    except OSError:
        pytest.skip("this process cannot create symlinks")
    probe.unlink()

    site = tmp_path / "site"
    site.mkdir()
    tar_path = tmp_path / "_devel.tar"
    with tarfile.open(tar_path, "w") as tf:
        _add_file(tf, "tree/before.txt", b"before")
        _add_symlink(tf, "tree/dangling.h", "../nowhere/missing.h")
        _add_file(tf, "tree/after.txt", b"after")

    _expand(site, tar_path, monkeypatch, deny=False)

    link = site / "tree/dangling.h"
    assert link.is_symlink()
    assert link.readlink() == Path("../nowhere/missing.h")
    assert (site / "tree/after.txt").read_bytes() == b"after"
    assert not tar_path.exists()
