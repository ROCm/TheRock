#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Generate PyTorch build manifests before any checkouts or builds.

Resolves git refs to commit SHAs and fetches version files via the GitHub
API, producing one manifest JSON per pytorch_git_ref. The manifests pin
exact commits so that downstream build and test jobs use identical source
revisions.

Usage::

    python generate_pytorch_manifest_upfront.py \
        --rocm-version 7.13.0a20260501 \
        --version-suffix "+rocm7.13.0a20260501" \
        --manifest-dir /tmp/manifests \
        --pytorch-git-refs "release/2.11"
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

_BUILD_TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BUILD_TOOLS_DIR))

from github_actions.github_actions_api import (
    gha_fetch_file_contents,
    gha_resolve_git_ref,
)
from github_actions.manifest_utils import (
    GitSourceInfo,
    detect_therock_source_info,
    log,
    normalize_ref_for_filename,
)

DEFAULT_PYTORCH_GIT_REFS = [
    "release/2.8",
    "release/2.9",
    "release/2.10",
    "release/2.11",
    "nightly",
]


@dataclass(frozen=True)
class RepoConfig:
    """Configuration for a pytorch ecosystem repository."""

    stable_repo: str
    nightly_repo: str
    nightly_branch: str | None = None
    version_file: str | None = None
    # Key in ROCm/pytorch's ``related_commits`` file (e.g. "torchaudio").
    # When set, stable builds resolve from related_commits; when None,
    # the repo uses custom resolution logic (pytorch itself, triton).
    related_commits_key: str | None = None
    # Platforms this repo is excluded from. Empty means all platforms.
    exclude_platforms: tuple[str, ...] = ()


REPOS: dict[str, RepoConfig] = {
    "pytorch": RepoConfig(
        stable_repo="ROCm/pytorch",
        nightly_repo="pytorch/pytorch",
        nightly_branch="nightly",
        version_file="version.txt",
    ),
    "pytorch_audio": RepoConfig(
        stable_repo="pytorch/audio",
        nightly_repo="pytorch/audio",
        nightly_branch="nightly",
        version_file="version.txt",
        related_commits_key="torchaudio",
    ),
    "pytorch_vision": RepoConfig(
        stable_repo="pytorch/vision",
        nightly_repo="pytorch/vision",
        nightly_branch="nightly",
        version_file="version.txt",
        related_commits_key="torchvision",
    ),
    "triton": RepoConfig(
        stable_repo="ROCm/triton",
        nightly_repo="ROCm/triton",
    ),
    "apex": RepoConfig(
        stable_repo="ROCm/apex",
        nightly_repo="ROCm/apex",
        nightly_branch="master",
        version_file="version.txt",
        related_commits_key="apex",
        exclude_platforms=("windows",),
    ),
}


def _resolve_ref(repo: str, ref: str) -> str:
    sha = gha_resolve_git_ref(repo, ref)
    log(f"  {repo}@{ref} -> {sha[:12]}")
    return sha


def _parse_related_commits(content: str) -> dict[str, dict[str, str]]:
    """Parse ROCm/pytorch's ``related_commits`` file.

    Returns a dict keyed by project name (e.g. "torchaudio") with
    "origin" and "commit" fields. Only ``centos`` entries are returned
    (used for both Linux and Windows builds).
    """
    pins: dict[str, dict[str, str]] = {}
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("|")
        if len(parts) != 6:
            log(f"  WARNING: skipping malformed related_commits line: {line}")
            continue
        rec_os, _source, rec_project, _branch, rec_commit, rec_origin = parts
        if rec_os == "centos":
            pins[rec_project] = {"origin": rec_origin, "commit": rec_commit}
    return pins


def _resolve_triton(
    pytorch_repo: str,
    pytorch_sha: str,
    *,
    nightly: bool,
    version_suffix: str,
    platform: str,
) -> GitSourceInfo:
    """Resolve triton commit and version from pytorch's pin files.

    The triton base version lives in pytorch's ``.ci/docker/triton_version.txt``.
    On Linux the commit comes from ``ci_commit_pins/triton.txt``; on Windows
    from ``ci_commit_pins/triton-windows.txt`` using a different repo.
    """
    is_windows = platform == "windows"

    if is_windows:
        triton_repo = "triton-lang/triton-windows"
        pin_file = ".ci/docker/ci_commit_pins/triton-windows.txt"
        fallback_branch = "main-windows"
    else:
        config = REPOS["triton"]
        triton_repo = config.nightly_repo if nightly else config.stable_repo
        pin_file = ".ci/docker/ci_commit_pins/triton.txt"
        fallback_branch = None

    # Base version is always in pytorch's triton_version.txt.
    base_version = gha_fetch_file_contents(
        pytorch_repo, ".ci/docker/triton_version.txt", pytorch_sha
    ).strip()
    version = f"{base_version}{version_suffix}"
    log(f"  triton: {base_version} -> {version}")

    if not is_windows and nightly:
        major, minor, *_ = base_version.split(".")
        branch = f"release/{major}.{minor}.x"
        sha = _resolve_ref(triton_repo, branch)
        return GitSourceInfo(
            commit=sha,
            repo=f"https://github.com/{triton_repo}.git",
            branch=branch,
            version=version,
        )

    # Stable (both platforms) or Windows nightly: use ci_commit_pins.
    try:
        pin = gha_fetch_file_contents(pytorch_repo, pin_file, pytorch_sha).strip()
        log(f"  triton pin: {pin[:12]}")
        return GitSourceInfo(
            commit=pin,
            repo=f"https://github.com/{triton_repo}.git",
            version=version,
        )
    except Exception:
        if fallback_branch:
            log(f"  triton: no pin file, falling back to {fallback_branch}")
            sha = _resolve_ref(triton_repo, fallback_branch)
            return GitSourceInfo(
                commit=sha,
                repo=f"https://github.com/{triton_repo}.git",
                branch=fallback_branch,
                version=version,
            )
        raise


def resolve_sources(
    pytorch_ref: str, version_suffix: str, platform: str
) -> dict[str, GitSourceInfo]:
    """Resolve all source commits for a given pytorch_git_ref."""
    nightly = pytorch_ref == "nightly"
    sources: dict[str, GitSourceInfo] = {}

    # Resolve pytorch first — other repos depend on it for pin files.
    pytorch_config = REPOS["pytorch"]
    pytorch_repo = (
        pytorch_config.nightly_repo if nightly else pytorch_config.stable_repo
    )
    pytorch_sha = _resolve_ref(pytorch_repo, pytorch_ref)
    sources["pytorch"] = GitSourceInfo(
        commit=pytorch_sha,
        repo=f"https://github.com/{pytorch_repo}.git",
        branch=pytorch_ref,
    )

    # For stable builds, load related_commits once (used by repos that have
    # a related_commits_key).
    pins: dict[str, dict[str, str]] = {}
    if not nightly:
        related_content = gha_fetch_file_contents(
            pytorch_repo, "related_commits", pytorch_sha
        )
        pins = _parse_related_commits(related_content)

    # Resolve remaining repos.
    for name, config in REPOS.items():
        if name == "pytorch":
            continue

        if platform in config.exclude_platforms:
            continue

        # Triton has its own pin mechanism.
        if name == "triton":
            sources[name] = _resolve_triton(
                pytorch_repo,
                pytorch_sha,
                nightly=nightly,
                version_suffix=version_suffix,
                platform=platform,
            )
            continue

        if nightly:
            sha = _resolve_ref(config.nightly_repo, config.nightly_branch)
            sources[name] = GitSourceInfo(
                commit=sha,
                repo=f"https://github.com/{config.nightly_repo}.git",
                branch=config.nightly_branch,
            )
        elif config.related_commits_key and config.related_commits_key in pins:
            pin = pins[config.related_commits_key]
            sources[name] = GitSourceInfo(commit=pin["commit"], repo=pin["origin"])
        else:
            fallback = config.nightly_branch or "main"
            sha = _resolve_ref(config.stable_repo, fallback)
            sources[name] = GitSourceInfo(
                commit=sha, repo=f"https://github.com/{config.stable_repo}.git"
            )

    return sources


def fetch_versions(
    sources: dict[str, GitSourceInfo], version_suffix: str
) -> dict[str, GitSourceInfo]:
    """Fetch version.txt for each repo and return updated GitSourceInfo entries."""
    updated: dict[str, GitSourceInfo] = {}
    for name, info in sources.items():
        version_file = REPOS[name].version_file
        if version_file is None:
            updated[name] = info
            continue

        repo = (
            info.repo.removeprefix("https://github.com/")
            .removesuffix(".git")
            .rstrip("/")
        )
        base_version = gha_fetch_file_contents(repo, version_file, info.commit).strip()
        full_version = f"{base_version}{version_suffix}"
        log(f"  {name}: {base_version} -> {full_version}")
        updated[name] = GitSourceInfo(
            commit=info.commit, repo=info.repo, branch=info.branch, version=full_version
        )
    return updated


def generate_manifest(
    *,
    pytorch_git_ref: str,
    rocm_version: str,
    version_suffix: str,
    platform: str,
    therock_commit: str,
    therock_repo: str,
    therock_branch: str,
) -> dict[str, object]:
    """Generate a single manifest for one pytorch_git_ref."""
    log(f"Generating manifest for {pytorch_git_ref} ({platform})")

    sources = resolve_sources(pytorch_git_ref, version_suffix, platform)
    sources = fetch_versions(sources, version_suffix)

    manifest: dict[str, object] = {
        name: info.to_dict() for name, info in sources.items()
    }
    manifest["therock"] = {
        "commit": therock_commit,
        "repo": therock_repo,
        "branch": therock_branch,
    }
    manifest["rocm_version"] = rocm_version
    manifest["version_suffix"] = version_suffix
    return manifest


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        description="Generate PyTorch build manifests (before checkout/build)"
    )
    parser.add_argument("--rocm-version", required=True, help="e.g. 7.13.0a20260501")
    parser.add_argument(
        "--version-suffix", required=True, help="e.g. +rocm7.13.0a20260501"
    )
    parser.add_argument(
        "--platform",
        choices=["linux", "windows"],
        default="linux",
        help="Target platform (affects repo selection and exclusions)",
    )
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument(
        "--therock-commit", help="Override TheRock commit (default: detect from git)"
    )
    parser.add_argument(
        "--therock-repo", help="Override TheRock repo URL (default: detect from git)"
    )
    parser.add_argument(
        "--therock-branch", help="Override TheRock branch (default: detect from git)"
    )
    parser.add_argument(
        "--pytorch-git-refs",
        default="",
        help="Space-separated pytorch refs (empty = all defaults)",
    )
    args = parser.parse_args(argv)

    refs = (
        args.pytorch_git_refs.split()
        if args.pytorch_git_refs
        else DEFAULT_PYTORCH_GIT_REFS
    )

    # Detect TheRock source info from the local repo, then apply CLI overrides.
    therock_root = Path(__file__).resolve().parents[2]
    therock_info = detect_therock_source_info(therock_root)
    therock_commit = args.therock_commit or therock_info.commit
    therock_repo = args.therock_repo or therock_info.repo
    therock_branch = args.therock_branch or therock_info.branch

    log(f"ROCm version: {args.rocm_version}, suffix: {args.version_suffix}")
    log(f"Platform: {args.platform}")
    log(f"TheRock: {therock_commit[:12]} ({therock_branch})")
    log(f"PyTorch refs: {refs}")
    log("")

    args.manifest_dir.mkdir(parents=True, exist_ok=True)

    for ref in refs:
        manifest = generate_manifest(
            pytorch_git_ref=ref,
            rocm_version=args.rocm_version,
            version_suffix=args.version_suffix,
            platform=args.platform,
            therock_commit=therock_commit,
            therock_repo=therock_repo,
            therock_branch=therock_branch,
        )
        filename = f"therock-manifest_torch_{args.platform}_{normalize_ref_for_filename(ref)}.json"
        out_path = args.manifest_dir / filename
        out_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8"
        )
        log(f"Wrote {out_path}\n")


if __name__ == "__main__":
    main(sys.argv[1:])
