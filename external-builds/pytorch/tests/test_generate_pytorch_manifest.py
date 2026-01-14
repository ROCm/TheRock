"""
Unit tests for external-builds/pytorch/generate_pytorch_manifest.py.
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
    # Simulate a GitHub Actions environment so CI metadata fields are deterministic.
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
    # Standard wheel filenames should parse without a build_tag.
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
    # Wheel filenames with build tags should expose build_tag separately.
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
    # Non-wheel artifacts should be ignored.
    m = _load_manifest_module()
    assert m.parse_wheel_name("torch-2.7.0.tar.gz") == {}
    assert m.parse_wheel_name("torch-2.7.0.zip") == {}
    assert m.parse_wheel_name("README.txt") == {}


def test_parse_wheel_name_too_few_fields():
    # Malformed wheel filenames (too few fields) should be rejected.
    m = _load_manifest_module()
    assert m.parse_wheel_name("torch-2.7.0.whl") == {}


def test_manifest_generation_end_to_end(tmp_path: Path, gha_env):
    # End-to-end: create a fake dist dir and ensure exactly one manifest is written.
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
        "--python-version",
        "3.12",
        "--pytorch-git-ref",
        "nightly",
    ]

    _run_main_with_args(m, argv)

    manifest_dir = out_dir / "manifests"

    # Exactly one manifest should be produced.
    manifests = list(manifest_dir.glob("*.json"))
    assert len(manifests) == 1, f"Expected exactly one manifest, found: {manifests}"

    # Verify naming convention for nightly builds.
    manifest_name = "therock-manifest_torch_py3.12_nightly.json"
    manifest_path = manifest_dir / manifest_name
    assert manifest_path.exists(), f"Expected manifest not found: {manifest_path}"

    data = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert data["project"] == "TheRock"
    assert data["component"] == "pytorch"
    assert data["artifact_group"] == "pytorch-wheels"

    # CI metadata should be captured when running under GitHub Actions.
    assert data["run_id"] == "123456"
    assert data["job_id"] == "build_pytorch_wheels"
    assert data["therock"]["repo"] == "https://github.com/ROCm/TheRock"
    assert data["therock"]["commit"] == "aabbccdd"
    assert data["therock"]["ref"] == "refs/heads/main"

    assert len(data["artifacts"]) == 1
    labels = data["artifacts"][0]["labels"]
    assert labels["distribution"] == "torch"
    assert labels["version"] == "2.7.0"


def test_manifest_filename_release_28(tmp_path: Path, gha_env):
    # release/2.8 should normalize to release-2.8 in the manifest filename.
    m = _load_manifest_module()

    out_dir = tmp_path / "dist"
    out_dir.mkdir(parents=True)

    (out_dir / "torch-2.8.0-cp312-cp312-manylinux_2_28_x86_64.whl").write_bytes(b"x")

    _run_main_with_args(
        m,
        [
            "--output-dir",
            str(out_dir),
            "--python-version",
            "3.12",
            "--pytorch-git-ref",
            "release/2.8",
        ],
    )

    manifest_path = (
        out_dir / "manifests" / "therock-manifest_torch_py3.12_release-2.8.json"
    )
    assert manifest_path.exists(), f"Missing manifest: {manifest_path}"
