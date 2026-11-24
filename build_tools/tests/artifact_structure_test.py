# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import io
from pathlib import Path
import sys
import tarfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))
import test_artifact_structure as structure


def write_archive(
    directory: Path,
    *,
    second_type: bytes = tarfile.REGTYPE,
    second_payload: bytes = b"shared",
    second_mode: int = 0o644,
    first_link: str = "",
    second_link: str = "",
) -> Path:
    archive = directory / "example_dev_generic.tar.xz"
    with tarfile.open(archive, "w:xz") as tf:
        manifest = b"a/stage\nb/stage\n"
        entry = tarfile.TarInfo("artifact_manifest.txt")
        entry.size = len(manifest)
        tf.addfile(entry, io.BytesIO(manifest))
        for stage, kind, payload, mode, link in (
            (
                "a",
                tarfile.SYMTYPE if first_link else tarfile.REGTYPE,
                b"shared",
                0o644,
                first_link,
            ),
            ("b", second_type, second_payload, second_mode, second_link),
        ):
            entry = tarfile.TarInfo(f"{stage}/stage/include/header.h")
            entry.type = kind
            entry.mode = mode
            entry.linkname = link
            entry.size = len(payload) if entry.isfile() else 0
            tf.addfile(entry, io.BytesIO(payload) if entry.isfile() else None)
    return archive


@pytest.mark.parametrize(
    "kind,payload,mode,link,accepted",
    [
        (tarfile.REGTYPE, b"shared", 0o644, "", True),
        (tarfile.REGTYPE, b"change", 0o644, "", False),
        (tarfile.REGTYPE, b"shared", 0o755, "", False),
        (tarfile.DIRTYPE, b"", 0o755, "", False),
        (tarfile.SYMTYPE, b"", 0o644, "other.h", False),
        (tarfile.LNKTYPE, b"", 0o644, "a/stage/include/header.h", True),
    ],
    ids=["identical", "contents", "executable", "directory", "symlink", "hardlink"],
)
def test_duplicate_flattened_payloads(tmp_path, kind, payload, mode, link, accepted):
    archive = write_archive(
        tmp_path,
        second_type=kind,
        second_payload=payload,
        second_mode=mode,
        second_link=link,
    )
    if accepted:
        prefixes, paths = structure.list_archive_files(archive)
        assert prefixes == ["a/stage", "b/stage"]
        assert paths == ["include/header.h", "include/header.h"]
    else:
        with pytest.raises(ValueError, match="conflicting flattened path") as error:
            structure.list_archive_files(archive)
        assert "a/stage/include/header.h" in str(error.value)
        assert "b/stage/include/header.h" in str(error.value)


@pytest.mark.parametrize("target", ["other.h", "different.h"])
def test_duplicate_symlink_targets(tmp_path, target):
    archive = write_archive(
        tmp_path,
        first_link="other.h",
        second_type=tarfile.SYMTYPE,
        second_link=target,
    )
    if target == "other.h":
        assert len(structure.list_archive_files(archive)[1]) == 2
    else:
        with pytest.raises(ValueError, match="conflicting flattened path"):
            structure.list_archive_files(archive)


def test_archive_index_fails_on_conflicting_payloads(tmp_path):
    write_archive(tmp_path, second_payload=b"change")
    with pytest.raises(pytest.fail.Exception, match="Invalid artifact archive"):
        structure.archive_index.__wrapped__(tmp_path)
