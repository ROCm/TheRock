# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import importlib.util
from pathlib import Path


def _load_wrapper_module():
    wrapper_path = Path(__file__).resolve().parents[1] / "artifact_manager.py"
    spec = importlib.util.spec_from_file_location(
        "artifact_manager_wrapper_under_test", wrapper_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_adds_default_topology_for_artifact_commands():
    wrapper = _load_wrapper_module()
    argv = ["fetch", "--run-id=12345", "--output-dir=build"]

    updated = wrapper._with_default_topology(argv)

    assert updated[:-1] == argv
    assert updated[-1] == f"--topology={wrapper._DEFAULT_TOPOLOGY_PATH}"
    assert wrapper._DEFAULT_TOPOLOGY_PATH == (
        Path(__file__).resolve().parents[2] / "BUILD_TOPOLOGY.toml"
    )


def test_preserves_explicit_topology():
    wrapper = _load_wrapper_module()
    argv = ["fetch", "--topology", "custom.toml", "--output-dir=build"]

    assert wrapper._with_default_topology(argv) == argv
    assert wrapper._with_default_topology(["fetch", "--topology=custom.toml"]) == [
        "fetch",
        "--topology=custom.toml",
    ]


def test_leaves_top_level_invocations_unchanged():
    wrapper = _load_wrapper_module()

    assert wrapper._with_default_topology([]) == []
    assert wrapper._with_default_topology(["--help"]) == ["--help"]
