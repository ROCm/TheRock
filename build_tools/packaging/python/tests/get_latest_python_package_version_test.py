# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Exercise version discovery with pip and a local package index, offline."""

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from get_latest_python_package_version import get_latest_python_package_version


def test_latest_includes_prereleases_without_downloading(tmp_path: Path):
    package_dir = tmp_path / "rocm"
    package_dir.mkdir()
    # These wheels deliberately do not exist: version discovery needs only the
    # index links, not the wheel contents or package metadata.
    (package_dir / "index.html").write_text(
        '<a href="rocm-10.1.0-py3-none-any.whl">stable</a>\n'
        '<a href="rocm-10.2.0a20261015-py3-none-any.whl">nightly</a>\n'
    )
    assert (
        get_latest_python_package_version("rocm", tmp_path.as_uri())
        == "10.2.0a20261015"
    )


def test_missing_package_fails_without_falling_back_to_pypi(tmp_path: Path):
    package_dir = tmp_path / "rocm"
    package_dir.mkdir()
    (package_dir / "index.html").write_text("")
    with pytest.raises(subprocess.CalledProcessError):
        get_latest_python_package_version("rocm", tmp_path.as_uri())
