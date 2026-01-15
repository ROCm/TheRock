"""
Unit tests for external-builds/pytorch/generate_pytorch_manifest.py.
"""

import json
import os
import sys
from pathlib import Path

import pytest

THIS_DIR = Path(__file__).resolve().parent
PYTORCH_DIR = THIS_DIR.parent
sys.path.insert(0, os.fspath(PYTORCH_DIR))

import generate_pytorch_manifest as m


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

    # Save environment state (only keys that exist)
    saved_env: dict[str, str] = {}
    for key in keys:
        if key in os.environ:
            saved_env[key] = os.environ[key]

    # Clean environment for test
    for key in keys:
        if key in os.environ:
            del os.environ[key]

    # Set test environment
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
        # Clean environment after test
        for key in keys:
            if key in os.environ:
                del os.environ[key]
        # Restore saved environment
        for key, value in saved_env.items():
            os.environ[key] = value


def _run_main_with_args(argv: list[str]) -> None:
    original_argv = sys.argv[:]
    sys.argv = ["generate_pytorch_manifest.py", *argv]
    try:
        m.main()
    finally:
        sys.argv = original_argv


def test_parse_wheel_name_no_build_tag():
    # Standard wheel filenames should parse without a build_tag.
    meta = m.parse_wheel_name("torch-2.7.0-cp312-cp312-manylinux_2_28_x86_64.whl")
    assert meta.distribution == "torch"
    assert meta.version == "2.7.0"
    assert meta.build_tag is None
    assert meta.python_tag == "cp312"
    assert meta.abi_tag == "cp312"
    assert meta.platform_tag == "manylinux_2_28_x86_64"


def test_parse_wheel_name_with_build_tag():
    # Wheel filenames with build tags should expose build_tag separately.
    meta = m.parse_wheel_name(
        "torch-2.7.0+rocm7.10.0a20251120-1-cp312-cp312-win_amd64.whl"
    )
    assert meta.distribution == "torch"
    assert meta.version == "2.7.0+rocm7.10.0a20251120"
    assert meta.build_tag == "1"
    assert meta.python_tag == "cp312"
    assert meta.abi_tag == "cp312"
    assert meta.platform_tag == "win_amd64"


def test_parse_wheel_name_ignores_non_wheel():
    # Non-wheel artifacts should be ignored.
    meta = m.parse_wheel_name("torch-2.7.0.tar.gz")
    assert meta.distribution is None
    assert meta.version is None
    assert meta.build_tag is None
    assert meta.python_tag is None
    assert meta.abi_tag is None
    assert meta.platform_tag is None


def test_parse_wheel_name_too_few_fields():
    # Malformed wheel filenames (too few fields) should be rejected.
    meta = m.parse_wheel_name("torch-2.7.0.whl")
    assert meta.distribution is None
    assert meta.version is None
    assert meta.build_tag is None
    assert meta.python_tag is None
    assert meta.abi_tag is None
    assert meta.platform_tag is None


def test_manifest_generation_end_to_end(tmp_path: Path, gha_env):
    # End-to-end: create a fake dist dir and ensure exactly one manifest is written.
    out_dir = tmp_path / "dist"
    out_dir.mkdir(parents=True)

    (out_dir / "torch-2.7.0-cp312-cp312-manylinux_2_28_x86_64.whl").write_bytes(b"x")

    manifest_dir = out_dir / "manifests"

    argv = [
        "--output-dir",
        str(out_dir),
        "--manifest-dir",
        str(manifest_dir),
        "--artifact-group",
        "pytorch-wheels",
        "--amdgpu-family",
        "gfx110X-all",
        "--rocm-sdk-version",
        "7.10.0a20251120",
        "--pytorch-rocm-arch",
        "gfx94X",
        "--python-version",
        "3.12",
        "--pytorch-git-ref",
        "nightly",
    ]

    _run_main_with_args(argv)

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
    out_dir = tmp_path / "dist"
    out_dir.mkdir(parents=True)

    (out_dir / "torch-2.8.0-cp312-cp312-manylinux_2_28_x86_64.whl").write_bytes(b"x")

    manifest_dir = out_dir / "manifests"

    _run_main_with_args(
        [
            "--output-dir",
            str(out_dir),
            "--manifest-dir",
            str(manifest_dir),
            "--artifact-group",
            "pytorch-wheels",
            "--amdgpu-family",
            "gfx110X-all",
            "--rocm-sdk-version",
            "7.10.0a20251120",
            "--python-version",
            "3.12",
            "--pytorch-git-ref",
            "release/2.8",
        ]
    )

    manifest_path = manifest_dir / "therock-manifest_torch_py3.12_release-2.8.json"
    assert manifest_path.exists(), f"Missing manifest: {manifest_path}"
