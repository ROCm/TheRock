#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Source identity resolution and caller-option forwarding for component CI."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from detect_external_repo_config import ExternalRepoConfig, build_external_repo_config


@pytest.mark.parametrize(
    "repository, expected_repository, expected_path",
    [
        ("rocm-libraries", "ROCm/rocm-libraries", "rocm-libraries"),
        ("rocm-systems", "ROCm/rocm-systems", "rocm-systems"),
        ("ROCm/ROCgdb", "ROCm/ROCgdb", "debug-tools/rocgdb/source"),
    ],
)
def test_source_identity_and_default_build_options(
    repository: str, expected_repository: str, expected_path: str
) -> None:
    config = build_external_repo_config(repository)
    assert config == ExternalRepoConfig(
        repository=expected_repository,
        ref="",
        submodule_path=expected_path,
    )


def test_tested_commit_and_caller_build_options_are_preserved() -> None:
    commit = "0123456789abcdef0123456789abcdef01234567"
    family_overrides = {"gfx94X-dcgpu": {"test_labels_for_family": "quick"}}
    config = build_external_repo_config(
        "ROCm/rocm-systems",
        ref=commit,
        projects="rccl",
        extra_cmake_options='-DFOO=ON -DBAR="two words"',
        family_overrides=family_overrides,
    )
    assert config.ref == commit
    assert config.projects == "rccl"
    assert config.extra_cmake_options == '-DFOO=ON -DBAR="two words"'
    assert config.family_overrides == family_overrides


@pytest.mark.parametrize("repository", ["", "ROCm/unknown-component"])
def test_unresolvable_repository_is_rejected(repository: str) -> None:
    with pytest.raises(ValueError, match="required|Unknown external repository"):
        build_external_repo_config(repository)
