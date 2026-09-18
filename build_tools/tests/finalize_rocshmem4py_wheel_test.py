#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from pathlib import Path
import sys

from packaging.requirements import Requirement

sys.path.insert(
    0,
    str(Path(__file__).parent.parent / "packaging" / "python"),
)

from finalize_rocshmem4py_wheel import _set_runtime_requirement


def test_runtime_requirement_is_exact_and_preserves_other_requirements(tmp_path: Path):
    metadata_path = tmp_path / "METADATA"
    metadata_path.write_text(
        "Metadata-Version: 2.4\n"
        "Name: rocshmem4py\n"
        "Version: 0.1.0\n"
        "Requires-Dist: unrelated-package>=1; extra == 'test'\n"
        "Requires-Dist: rocm-sdk-core==9.9.9\n"
        "\n"
    )

    rocm_version = f"10.1.0.dev0+{'a' * 40}"
    _set_runtime_requirement(metadata_path, rocm_version)

    requirements = [
        Requirement(line.removeprefix("Requires-Dist: "))
        for line in metadata_path.read_text().splitlines()
        if line.startswith("Requires-Dist: ")
    ]
    assert [requirement.name for requirement in requirements].count(
        "rocm-sdk-core"
    ) == 1
    core = next(
        requirement
        for requirement in requirements
        if requirement.name == "rocm-sdk-core"
    )
    assert str(core.specifier) == f"=={rocm_version}"
    assert any(requirement.name == "unrelated-package" for requirement in requirements)
