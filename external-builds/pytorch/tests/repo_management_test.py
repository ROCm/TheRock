# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import importlib.util
from pathlib import Path
import subprocess
from unittest import mock

import pytest


@pytest.fixture(params=["pytorch", "uccl"])
def repo_management(request):
    source = Path(__file__).resolve().parents[2] / request.param / "repo_management.py"
    spec = importlib.util.spec_from_file_location(
        f"{request.param}_repo_management", source
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_empty_gitmodules_is_allowed(repo_management, tmp_path):
    (tmp_path / ".gitmodules").write_text("")
    with mock.patch.object(repo_management, "run_command") as run:
        repo_management.git_config_ignore_submodules(tmp_path)
        run.assert_not_called()


def test_invalid_gitmodules_is_not_ignored(repo_management, tmp_path):
    (tmp_path / ".gitmodules").write_text("[unterminated section")
    with pytest.raises(subprocess.CalledProcessError):
        repo_management.git_config_ignore_submodules(tmp_path)


def test_git_config_write_failure_is_not_ignored(repo_management, tmp_path):
    (tmp_path / ".gitmodules").write_text('[submodule "example"]\npath = example\n')
    failure = subprocess.CalledProcessError(1, ["git", "config"])
    with mock.patch.object(repo_management, "run_command", side_effect=failure):
        with pytest.raises(subprocess.CalledProcessError) as caught:
            repo_management.git_config_ignore_submodules(tmp_path)
    assert caught.value is failure
