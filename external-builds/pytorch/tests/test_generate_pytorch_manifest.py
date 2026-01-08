"""
Tests for generate_pytorch_manifest.py

This test suite validates the PyTorch external-build manifest generator used by
TheRock CI workflows. The goal is to ensure correct, reproducible manifest
generation for externally built PyTorch wheels, independent of the execution
environment.

What is covered
----------------
1. Wheel filename parsing (PEP 427)
   - Correctly parses wheel filenames to extract:
     * distribution
     * version (including ROCm suffixes)
     * optional build tag
     * python tag
     * ABI tag
     * platform tag
   - Ignores non-wheel files (sdist, zip, text) and malformed filenames

2. GitHub Actions environment handling
   - Simulates a GitHub Actions runtime using environment variables
   - Verifies that CI metadata is captured only when GITHUB_ACTIONS=true
   - Validates recorded fields:
     * run_id
     * job_id
     * repository URL
     * commit SHA
     * git ref

3. End-to-end manifest generation
   - Creates a temporary wheel output directory
   - Invokes generate_pytorch_manifest.main() as a CLI entrypoint
   - Generates exactly one manifest JSON file under <output-dir>/manifests
   - Verifies TheRock naming conventions for the manifest file

4. Manifest content validation
   - Validates top-level metadata:
     * project == "TheRock"
     * component == "pytorch"
     * artifact_group == "pytorch-wheels"
   - Validates artifact entries:
     * relative path
     * file size
     * parsed wheel labels

How to run
----------
From the repository root:

    pytest external-builds/pytorch/tests/test_generate_pytorch_manifest.py
"""

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "external-builds" / "pytorch" / "generate_pytorch_manifest.py"


def _load_manifest_module() -> ModuleType:
    assert SCRIPT_PATH.exists(), f"Missing script at {SCRIPT_PATH}"

    spec = importlib.util.spec_from_file_location(
        "generate_pytorch_manifest", SCRIPT_PATH
    )
    assert spec and spec.loader

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def gha_env():
    keys = [
        "GITHUB_ACTIONS",
        "GITHUB_RUN_ID",
        "GITHUB_JOB",
        "GITHUB_SERVER_URL",
        "GITHUB_REPOSITORY",
        "GITHUB_SHA",
        "GITHUB_REF",
    ]

    original_values: dict[str, str | None] = {}
    for key in keys:
        original_values[key] = os.environ.get(key)

    os.environ["GITHUB_ACTIONS"] = "true"
    os.environ["GITHUB_RUN_ID"] = "123456"
    os.environ["GITHUB_JOB"] = "build_pytorch_wheels"
    os.environ["GITHUB_SERVER_URL"] = "https://github.com"
    os.environ["GITHUB_REPOSITORY"] = "ROCm/TheRock"
    os.environ["GITHUB_SHA"] = "aabbccdd"
    os.environ["GITHUB_REF"] = "refs/heads/main"

    try:
        yield
    finally:
        for key, value in original_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _run_main_with_args(module: ModuleType, argv: list[str]) -> None:
    original_argv = sys.argv[:]
    sys.argv = ["generate_pytorch_manifest.py", *argv]
    try:
        module.main()
    finally:
        sys.argv = original_argv


def test_parse_wheel_name_no_build_tag():
    m = _load_manifest_module()
    meta = m.parse_wheel_name("torch-2.7.0-cp312-cp312-manylinux_2_28_x86_64.whl")
    assert meta == {
        "distribution": "torch",
        "version": "2.7.0",
        "python_tag": "cp312",
        "abi_tag": "cp312",
        "platform_tag": "manylinux_2_28_x86_64",
    }


def test_parse_wheel_name_with_build_tag():
    m = _load_manifest_module()
    meta = m.parse_wheel_name(
        "torch-2.7.0+rocm7.10.0a20251120-1-cp312-cp312-win_amd64.whl"
    )
    assert meta == {
        "distribution": "torch",
        "version": "2.7.0+rocm7.10.0a20251120",
        "build_tag": "1",
        "python_tag": "cp312",
        "abi_tag": "cp312",
        "platform_tag": "win_amd64",
    }


def test_parse_wheel_name_ignores_non_wheel():
    m = _load_manifest_module()
    assert m.parse_wheel_name("torch-2.7.0.tar.gz") == {}
    assert m.parse_wheel_name("torch-2.7.0.zip") == {}
    assert m.parse_wheel_name("README.txt") == {}


def test_parse_wheel_name_too_few_fields():
    m = _load_manifest_module()
    assert m.parse_wheel_name("torch-2.7.0.whl") == {}


def test_manifest_generation_end_to_end(tmp_path: Path, gha_env):
    m = _load_manifest_module()

    out_dir = tmp_path / "dist"
    out_dir.mkdir(parents=True)

    (out_dir / "torch-2.7.0-cp312-cp312-manylinux_2_28_x86_64.whl").write_bytes(b"x")

    argv = [
        "--output-dir",
        str(out_dir),
        "--rocm-sdk-version",
        "7.10.0a20251120",
        "--pytorch-rocm-arch",
        "gfx94X",
        "--version-suffix",
        "+rocm7.10.0a20251120",
    ]

    _run_main_with_args(m, argv)

    manifest_dir = out_dir / "manifests"
    file = [manifest_dir / "therock_torch_manifest.json"]
    assert file[0].exists()
    assert len(file) == 1

    data = json.loads(file[0].read_text(encoding="utf-8"))
    assert data["project"] == "TheRock"
    assert data["component"] == "pytorch"
    assert data["artifact_group"] == "pytorch-wheels"

    assert data["run_id"] == "123456"
    assert data["job_id"] == "build_pytorch_wheels"
    assert data["therock"]["repo"] == "https://github.com/ROCm/TheRock"
    assert data["therock"]["commit"] == "aabbccdd"
    assert data["therock"]["ref"] == "refs/heads/main"

    assert len(data["artifacts"]) == 1
    labels = data["artifacts"][0]["labels"]
    assert labels["distribution"] == "torch"
    assert labels["version"] == "2.7.0"
