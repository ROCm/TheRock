# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fetch_pytorch_matrix import generate_pytorch_matrix


def test_full_matrix_linux():
    matrix = generate_pytorch_matrix(None, "", "linux")
    # 5 python versions * 6 pytorch refs - 1 (py3.14 + release/2.8 exclude)
    assert len(matrix) == 29


def test_full_matrix_windows():
    matrix = generate_pytorch_matrix(None, "", "windows")
    # 5 python versions * 5 pytorch refs (no release/2.8)
    assert len(matrix) == 25


def test_single_python_version():
    matrix = generate_pytorch_matrix("3.12", "", "linux")
    assert len(matrix) == 6
    assert all(m["python_version"] == "3.12" for m in matrix)


def test_single_python_version_windows():
    matrix = generate_pytorch_matrix("3.12", "", "windows")
    assert len(matrix) == 5
    assert all(m["python_version"] == "3.12" for m in matrix)


def test_python_314_excludes_release_28():
    matrix = generate_pytorch_matrix("3.14", "", "linux")
    refs = [m["pytorch_git_ref"] for m in matrix]
    assert "release/2.8" not in refs
    assert len(matrix) == 5


def test_gfx1153_excludes_old_pytorch_linux():
    matrix = generate_pytorch_matrix(None, "gfx1153", "linux")
    # 29 - 5 (2.8) - 5 (2.9) + 1 (py3.14+2.8 already excluded) = 20
    assert len(matrix) == 20
    for m in matrix:
        assert m["pytorch_git_ref"] not in ["release/2.8", "release/2.9"]


def test_gfx1153_excludes_old_pytorch_windows():
    matrix = generate_pytorch_matrix(None, "gfx1153", "windows")
    # 25 - 5 (2.9) = 20
    assert len(matrix) == 20
    for m in matrix:
        assert m["pytorch_git_ref"] != "release/2.9"


def test_gfx1153_single_python():
    matrix = generate_pytorch_matrix("3.12", "gfx1153", "linux")
    assert len(matrix) == 4
    refs = [m["pytorch_git_ref"] for m in matrix]
    assert "release/2.8" not in refs
    assert "release/2.9" not in refs
