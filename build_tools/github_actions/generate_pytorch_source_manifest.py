#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Generate PyTorch source manifests.

Resolves PyTorch ecosystem refs and pin files to exact commit SHAs and records
expected package versions in manifest JSON files.

Usage::

    python generate_pytorch_source_manifest.py \
        --rocm-version 7.13.0a20260501 \
        --version-suffix "+rocm7.13.0a20260501" \
        --manifest-dir /tmp/manifests \
        --pytorch-git-refs "release/2.11"
"""

import argparse
import json
import platform as platform_module
import sys
from dataclasses import dataclass
from pathlib import Path

_BUILD_TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BUILD_TOOLS_DIR))

from github_actions.github_actions_api import (
    gha_fetch_text_file_contents,
    gha_resolve_git_ref,
)
from github_actions.determine_version import derive_version_suffix
from github_actions.manifest_utils import (
    GitSourceInfo,
    detect_therock_source_info,
    log,
    normalize_ref_for_filename,
)

DEFAULT_PYTORCH_GIT_REFS = [
    "release/2.10",
    "release/2.11",
    "release/2.12",
    "release/2.13",
    "nightly",
]
SCHEMA_VERSION = 1

THEROCK_DIR = Path(__file__).resolve().parents[2]
TRITON_WINDOWS_REPO = "triton-lang/triton-windows"

Manifest = dict[str, GitSourceInfo]


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
        # Triton does not use a floating nightly branch. PyTorch's pin files
        # resolve the exact commit and the base package version.
        # Windows release Triton is not enabled by default until PyTorch repos
        # publish a shared pin format for release branches.
        exclude_platforms=("windows",),
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


def _split_words(value: str) -> list[str]:
    return value.replace(";", " ").split() if value else []


def validate_projects(projects: list[str]) -> None:
    """Validate requested project names before resolving refs."""
    unknown = sorted(set(projects) - set(REPOS))
    if unknown:
        available = ", ".join(REPOS)
        raise ValueError(
            f"Unknown PyTorch manifest project(s): {', '.join(unknown)}. "
            f"Available projects: {available}"
        )
    if "pytorch" not in projects:
        raise ValueError("pytorch must be in the projects list")


def manifest_filename(*, platform: str, pytorch_git_ref: str) -> str:
    ref = normalize_ref_for_filename(pytorch_git_ref)
    return f"therock-manifest_torch_{platform}_{ref}.json"


def write_manifest_file(path: Path, manifest: Manifest) -> None:
    serialized_manifest: dict[str, object] = {"schema_version": SCHEMA_VERSION}
    serialized_manifest.update(
        {name: entry.to_dict() for name, entry in manifest.items()}
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(serialized_manifest, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Failed to write manifest: {path}")


def _parse_related_commits(content: str) -> dict[str, dict[str, str]]:
    """Parse ROCm/pytorch's ``related_commits`` file.

    Returns a dict keyed by project name (e.g. "torchaudio") with
    "origin" and "commit" fields. Some release branches list both
    ``ubuntu`` and ``centos`` entries with the same pin, while others only
    list one platform. Since the manifest needs one commit per project, reject
    duplicate project rows that disagree.
    """
    pins: dict[str, dict[str, str]] = {}
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Example:
        # ubuntu|pytorch|torchaudio|release/2.10|<commit>|https://github.com/pytorch/audio
        parts = line.split("|")
        if len(parts) != 6:
            raise ValueError(f"Malformed related_commits line: {line!r}")
        rec_os, _source, rec_project, _branch, rec_commit, rec_origin = parts
        pin = {"origin": rec_origin, "commit": rec_commit}
        previous = pins.get(rec_project)
        if previous is not None and previous != pin:
            raise ValueError(
                f"Conflicting related_commits entries for {rec_project!r}: "
                f"{previous!r} vs {pin!r} on {rec_os!r}"
            )
        pins[rec_project] = pin
    return pins


def read_triton_windows_pin() -> str:
    """Read TheRock's current Windows Triton commit pin."""
    pin_file = (
        THEROCK_DIR
        / "external-builds"
        / "pytorch"
        / "ci_commit_pins"
        / "triton-windows.txt"
    )
    if not pin_file.is_file():
        raise FileNotFoundError(f"Windows Triton pin file does not exist: {pin_file}")
    pin = pin_file.read_text(encoding="utf-8").strip()
    if not pin:
        raise ValueError(f"Windows Triton pin file is empty: {pin_file}")
    return pin


def _resolve_triton(
    pytorch_repo: str,
    pytorch_ref: str,
    pytorch_sha: str,
    *,
    version_suffix: str,
    platform: str,
) -> GitSourceInfo:
    """Resolve triton commit and version from pytorch's pin files.

    The triton base version lives in pytorch's ``.ci/docker/triton_version.txt``.
    On Linux the commit comes from ``ci_commit_pins/triton.txt``.
    See:
    https://github.com/pytorch/pytorch/blob/main/.ci/docker/triton_version.txt
    https://github.com/pytorch/pytorch/blob/main/.ci/docker/ci_commit_pins/triton.txt
    """
    is_windows = platform == "windows"

    if is_windows and pytorch_ref != "nightly":
        raise ValueError(
            "Windows Triton manifest generation currently supports only "
            "PyTorch nightly"
        )

    # Base version is always in pytorch's triton_version.txt.
    base_version = gha_fetch_text_file_contents(
        pytorch_repo, ".ci/docker/triton_version.txt", pytorch_sha
    ).strip()
    version = f"{base_version}{version_suffix}"
    log(f"  triton: {base_version} -> {version}")

    if is_windows:
        pin = read_triton_windows_pin()
        log(f"  triton-windows pin: {pin[:12]}")
        return GitSourceInfo(
            commit=pin,
            repo=f"https://github.com/{TRITON_WINDOWS_REPO}",
            version=version,
        )

    config = REPOS["triton"]
    triton_repo = (
        config.nightly_repo if pytorch_ref == "nightly" else config.stable_repo
    )
    pin_file = ".ci/docker/ci_commit_pins/triton.txt"

    # Use PyTorch's explicit Triton pin. triton_version.txt only provides the
    # package version; the matching release branch can move independently.
    pin = gha_fetch_text_file_contents(pytorch_repo, pin_file, pytorch_sha).strip()
    log(f"  triton pin: {pin[:12]}")
    return GitSourceInfo(
        commit=pin,
        repo=f"https://github.com/{triton_repo}",
        version=version,
    )


def default_projects_for_platform(platform: str) -> list[str]:
    """Return the default project list for a platform."""
    return [
        name
        for name, config in REPOS.items()
        if platform not in config.exclude_platforms
    ]


def default_projects_for_pytorch_ref(platform: str, pytorch_ref: str) -> list[str]:
    """Return default projects for a platform and PyTorch ref."""
    projects = default_projects_for_platform(platform)

    if False:
        # TODO: Flip this once Windows Triton nightly builds are ready by
        # default. Release branches still need a shared PyTorch-hosted pin
        # format.
        if (
            platform == "windows"
            and pytorch_ref == "nightly"
            and "triton" not in projects
        ):
            projects.append("triton")
    return projects


def resolve_sources(
    pytorch_ref: str,
    version_suffix: str,
    platform: str,
    projects: list[str],
) -> dict[str, GitSourceInfo]:
    """Resolve source commits for the requested projects."""
    validate_projects(projects)

    nightly = pytorch_ref == "nightly"
    sources: dict[str, GitSourceInfo] = {}

    # Resolve pytorch first — other repos depend on it for pin files.
    pytorch_config = REPOS["pytorch"]
    pytorch_repo = (
        pytorch_config.nightly_repo if nightly else pytorch_config.stable_repo
    )
    pytorch_sha = gha_resolve_git_ref(pytorch_repo, pytorch_ref)
    log(f"  {pytorch_repo}@{pytorch_ref} -> {pytorch_sha[:12]}")
    sources["pytorch"] = GitSourceInfo(
        commit=pytorch_sha,
        repo=f"https://github.com/{pytorch_repo}",
        branch=pytorch_ref,
    )

    # For stable builds, load related_commits once (used by repos that have
    # a related_commits_key).
    pins: dict[str, dict[str, str]] = {}
    needs_related_commits = any(
        name != "pytorch"
        and name in projects
        and REPOS[name].related_commits_key is not None
        for name in REPOS
    )
    if not nightly and needs_related_commits:
        related_content = gha_fetch_text_file_contents(
            pytorch_repo, "related_commits", pytorch_sha
        )
        pins = _parse_related_commits(related_content)

    # Resolve remaining repos.
    for name, config in REPOS.items():
        if name == "pytorch" or name not in projects:
            continue

        # Triton has its own pin mechanism.
        if name == "triton":
            sources[name] = _resolve_triton(
                pytorch_repo,
                pytorch_ref,
                pytorch_sha,
                version_suffix=version_suffix,
                platform=platform,
            )
            continue

        if nightly:
            sha = gha_resolve_git_ref(config.nightly_repo, config.nightly_branch)
            log(f"  {config.nightly_repo}@{config.nightly_branch} -> {sha[:12]}")
            sources[name] = GitSourceInfo(
                commit=sha,
                repo=f"https://github.com/{config.nightly_repo}",
                branch=config.nightly_branch,
            )
        elif config.related_commits_key and config.related_commits_key in pins:
            pin = pins[config.related_commits_key]
            sources[name] = GitSourceInfo(commit=pin["commit"], repo=pin["origin"])
        elif config.related_commits_key:
            raise ValueError(
                f"{pytorch_ref}: related_commits is missing "
                f"{config.related_commits_key!r} for {name!r}"
            )
        else:
            raise ValueError(
                f"{name!r} does not have a stable PyTorch release pin policy"
            )

    return sources


def fetch_versions(
    sources: dict[str, GitSourceInfo], version_suffix: str
) -> dict[str, GitSourceInfo]:
    """Fetch version.txt for each repo and return updated GitSourceInfo entries.

    PyTorch ecosystem projects store their base Python package versions in
    plain text files such as ``version.txt``. For example:
    https://github.com/pytorch/pytorch/blob/nightly/version.txt

    This step reads those base versions at the already-resolved commits and
    appends TheRock's ROCm version suffix. Projects without a version_file, such
    as Triton, must have their version filled in by their custom resolver.
    """
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
        base_version = gha_fetch_text_file_contents(
            repo, version_file, info.commit
        ).strip()
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
    projects: list[str],
    therock_commit: str,
    therock_repo: str,
    therock_branch: str,
) -> Manifest:
    """Generate a single manifest for one pytorch_git_ref."""
    log(f"Generating manifest for {pytorch_git_ref} ({platform})")

    sources = resolve_sources(pytorch_git_ref, version_suffix, platform, projects)
    sources = fetch_versions(sources, version_suffix)
    sources["therock"] = GitSourceInfo(
        repo=therock_repo,
        commit=therock_commit,
        branch=therock_branch,
        version=rocm_version,
    )

    return sources


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Generate PyTorch source manifests")
    parser.add_argument("--rocm-version", required=True, help="e.g. 7.13.0a20260501")
    parser.add_argument(
        "--version-suffix",
        default="",
        help="e.g. +rocm7.13.0a20260501 (default: derive from --rocm-version)",
    )
    parser.add_argument(
        "--platform",
        choices=["linux", "windows"],
        default=platform_module.system().lower(),
        help=(
            "Target platform (affects repo selection and exclusions; "
            "default: current system)"
        ),
    )
    output_group = parser.add_mutually_exclusive_group(required=True)
    output_group.add_argument(
        "--output",
        type=Path,
        help="Write a single manifest to this exact path (requires one --pytorch-git-refs entry)",
    )
    output_group.add_argument(
        "--manifest-dir",
        type=Path,
        help="Write manifests to this directory (filenames are computed from refs)",
    )
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
        "--projects",
        default="",
        help=(
            "Semicolon- or space-separated list of projects to include in the "
            "manifest (default: all projects for the platform)"
        ),
    )
    parser.add_argument(
        "--pytorch-git-refs",
        default="",
        help="Semicolon- or space-separated pytorch refs (empty = all defaults)",
    )
    args = parser.parse_args(argv)

    refs = _split_words(args.pytorch_git_refs) or DEFAULT_PYTORCH_GIT_REFS

    if args.output and len(refs) != 1:
        parser.error("--output requires exactly one --pytorch-git-refs entry")
    explicit_projects = _split_words(args.projects) or None
    version_suffix = args.version_suffix or derive_version_suffix(args.rocm_version)

    # Detect TheRock source info from the local repo, then apply CLI overrides.
    therock_root = Path(__file__).resolve().parents[2]
    therock_info = detect_therock_source_info(therock_root)
    therock_commit = args.therock_commit or therock_info.commit
    therock_repo = args.therock_repo or therock_info.repo
    therock_branch = args.therock_branch or therock_info.branch

    log(f"ROCm version: {args.rocm_version}, suffix: {version_suffix}")
    log(
        f"Platform: {args.platform}, projects: "
        f"{explicit_projects or 'default per PyTorch ref'}"
    )
    log(f"TheRock: {therock_commit[:12]} ({therock_branch})")
    log(f"PyTorch refs: {refs}")
    log("")

    for ref in refs:
        projects = explicit_projects or default_projects_for_pytorch_ref(
            args.platform, ref
        )
        log(f"Projects for {ref}: {projects}")
        manifest = generate_manifest(
            pytorch_git_ref=ref,
            rocm_version=args.rocm_version,
            version_suffix=version_suffix,
            platform=args.platform,
            projects=projects,
            therock_commit=therock_commit,
            therock_repo=therock_repo,
            therock_branch=therock_branch,
        )

        if args.output:
            out_path = args.output
        else:
            out_path = args.manifest_dir / manifest_filename(
                platform=args.platform, pytorch_git_ref=ref
            )

        write_manifest_file(out_path, manifest)
        log(f"Wrote {out_path}")
        log(out_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main(sys.argv[1:])
