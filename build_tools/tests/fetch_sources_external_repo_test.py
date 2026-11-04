# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Exercise index overrides, stage selection, and provenance with real Git repos."""

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

import fetch_sources
from generate_therock_manifest import build_manifest_schema


def git(repo: Path, *args: str, input: str | None = None) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=repo, text=True, input=input
    ).strip()


def commit(repo: Path, message: str) -> str:
    tree = git(repo, "write-tree")
    sha = git(repo, "commit-tree", tree, "-m", message)
    git(repo, "update-ref", "refs/heads/main", sha)
    git(repo, "symbolic-ref", "HEAD", "refs/heads/main")
    return sha


@pytest.fixture
def source_tree(tmp_path, monkeypatch):
    # Isolate Git config and rewrite the production GitHub URL to a local remote.
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_ALLOW_PROTOCOL", "file")
    for role in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{role}_NAME", "Source fixture")
        monkeypatch.setenv(f"GIT_{role}_EMAIL", "fixture@example.invalid")
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "safe.directory")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "*")

    upstream = tmp_path / "upstream"
    upstream.mkdir()
    git(upstream, "init")
    old = commit(upstream, "upstream")
    fork = tmp_path / "fork"
    fork.mkdir()
    git(fork, "init")
    new = commit(fork, "fork-only commit")
    monkeypatch.setenv("GIT_CONFIG_KEY_1", f"url.{fork.as_uri()}.insteadOf")
    monkeypatch.setenv("GIT_CONFIG_VALUE_1", "https://github.com/test/fork.git")
    monkeypatch.setenv("GIT_CONFIG_COUNT", "2")

    repo = tmp_path / "TheRock"
    repo.mkdir()
    git(repo, "init")
    for name in ("rocm-systems", "rocm-libraries"):
        git(repo, "config", "--file", ".gitmodules", f"submodule.{name}.path", name)
        git(
            repo,
            "config",
            "--file",
            ".gitmodules",
            f"submodule.{name}.url",
            upstream.as_uri(),
        )
    git(repo, "add", ".gitmodules")
    for name in ("rocm-systems", "rocm-libraries"):
        git(repo, "update-index", "--add", "--cacheinfo", f"160000,{old},{name}")
    base = commit(repo, "TheRock base")
    topology = repo / "BUILD_TOPOLOGY.toml"
    topology.write_text(
        textwrap.dedent(
            """
        [source_sets.systems]
        submodules = ["rocm-systems"]
        [source_sets.libraries]
        submodules = ["rocm-libraries"]
        [build_stages.math-libs]
        description = "Both source repositories"
        artifact_groups = ["math"]
        [build_stages.cv-libs]
        description = "Libraries only"
        artifact_groups = ["cv"]
        [artifact_groups.math]
        description = "Math"
        type = "generic"
        source_sets = ["systems", "libraries"]
        [artifact_groups.cv]
        description = "CV"
        type = "generic"
        source_sets = ["libraries"]
    """
        )
    )
    patches = repo / "patches" / "amd-mainline" / "rocm-systems"
    patches.mkdir(parents=True)
    # A bad patch must never be applied to the external PR checkout.
    (patches / "0001.patch").write_text("deliberately invalid patch\n")
    config = {"repository": "test/fork", "ref": new, "submodule_path": "rocm-systems"}
    monkeypatch.setenv("EXTERNAL_REPO_CONFIG", json.dumps(config))
    monkeypatch.delenv("THEROCK_GIT_MIRROR_DIR", raising=False)
    monkeypatch.setattr(fetch_sources, "THEROCK_DIR", repo)
    monkeypatch.setattr(fetch_sources, "TOPOLOGY_PATH", topology)
    monkeypatch.setattr(
        fetch_sources, "BRANCH_CONFIG_PATH", repo / "BRANCH_CONFIG.json"
    )
    monkeypatch.setattr(fetch_sources, "PATCHES_DIR", repo / "patches")
    return repo, base, old, new, config


def test_selected_override_and_default_source(source_tree):
    repo, base, old, new, config = source_tree
    fetch_sources.main(["--stage", "math-libs", "--depth", "1"])
    assert git(repo / "rocm-systems", "rev-parse", "HEAD") == new
    assert git(repo / "rocm-libraries", "rev-parse", "HEAD") == old
    assert git(repo, "rev-parse", "HEAD") == base
    assert (
        git(repo / "rocm-systems", "remote", "get-url", "origin")
        == (repo.parent / "fork").as_uri()
    )
    assert fetch_sources.get_submodule_revision("rocm-systems") == new
    manifest = build_manifest_schema(repo, base, external_repo_config=config)
    rows = {row["submodule_name"]: row for row in manifest["submodules"]}
    assert rows["rocm-systems"]["pin_sha"] == new
    assert rows["rocm-systems"]["submodule_url"] == "https://github.com/test/fork.git"
    assert rows["rocm-systems"]["patches"] == []
    assert rows["rocm-libraries"]["pin_sha"] == old


def test_unselected_override_is_reported_without_checkout(source_tree):
    repo, base, old, new, config = source_tree
    fetch_sources.main(["--stage", "cv-libs", "--depth", "1"])
    assert not (repo / "rocm-systems" / ".git").exists()
    assert git(repo / "rocm-libraries", "rev-parse", "HEAD") == old
    manifest = build_manifest_schema(repo, base, external_repo_config=config)
    rows = {row["submodule_name"]: row for row in manifest["submodules"]}
    assert rows["rocm-systems"]["pin_sha"] == new


def test_no_external_config_uses_original_gitlinks(source_tree, monkeypatch):
    repo, base, old, _, _ = source_tree
    monkeypatch.delenv("EXTERNAL_REPO_CONFIG")
    fetch_sources.main(["--stage", "math-libs", "--no-apply-patches"])
    assert git(repo / "rocm-systems", "rev-parse", "HEAD") == old
    assert git(repo / "rocm-libraries", "rev-parse", "HEAD") == old
    assert git(repo, "diff", "--cached", "--name-only") == ""
    manifest = build_manifest_schema(repo, base)
    assert all(row["pin_sha"] == old for row in manifest["submodules"])


def test_existing_checkout_changes_remote_and_pin(source_tree):
    repo, _, old, new, config = source_tree
    # Initialize using the original URL/pin before enabling the override.
    git(repo, "submodule", "update", "--init", "--", "rocm-systems")
    assert git(repo / "rocm-systems", "rev-parse", "HEAD") == old
    fetch_sources.main(["--stage", "math-libs", "--depth", "1"])
    assert git(repo / "rocm-systems", "rev-parse", "HEAD") == new
    # Exercise reapplying an index override after normal fetch bookkeeping.
    fetch_sources.main(["--stage", "math-libs", "--depth", "1"])
    assert git(repo / "rocm-systems", "rev-parse", "HEAD") == new


@pytest.mark.parametrize("windows", [False, True])
def test_external_checkout_uses_normal_dvc_projects(source_tree, monkeypatch, windows):
    repo, _, _, _, _ = source_tree
    monkeypatch.setattr(fetch_sources, "is_windows", lambda: windows)
    fetch_sources.main(["--stage", "math-libs", "--depth", "1"])
    for name in ("rocm-systems", "rocm-libraries"):
        dvc_dir = repo / name / ".dvc"
        dvc_dir.mkdir()
        (dvc_dir / "config").write_text("[core]\nremote = fixture\n")
    pulled = []

    def pull(project_dir, *, jobs):
        pulled.append(project_dir)
        return SimpleNamespace(fetched=0, cached=0, skipped=0)

    monkeypatch.setattr(fetch_sources.fetch_dvc_artifacts, "pull", pull)
    fetch_sources.main(["--stage", "math-libs", "--depth", "1"])
    expected = [repo / "rocm-libraries"]
    if windows:
        expected.append(repo / "rocm-systems")
    assert pulled == expected


@pytest.mark.parametrize(
    "field,value", [("ref", "main"), ("submodule_path", "missing")]
)
def test_invalid_override_leaves_index_unchanged(
    source_tree, monkeypatch, field, value
):
    repo, _, _, _, config = source_tree
    original_index = git(repo, "ls-files", "--stage")
    config[field] = value
    monkeypatch.setenv("EXTERNAL_REPO_CONFIG", json.dumps(config))
    with pytest.raises(ValueError):
        fetch_sources.main(["--stage", "math-libs"])
    assert git(repo, "ls-files", "--stage") == original_index
