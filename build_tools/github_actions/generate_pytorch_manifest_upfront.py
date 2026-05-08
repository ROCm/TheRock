#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Generate PyTorch build manifests upfront, before any checkouts or builds.

Resolves git refs to commit SHAs and fetches version files via the GitHub
API, producing one manifest JSON per pytorch_git_ref. The manifests pin
exact commits so that downstream build and test jobs use identical source
revisions.

For stable releases (e.g. release/2.10), the ``related_commits`` file in
the ROCm/pytorch fork is parsed to resolve torchaudio, torchvision, apex,
and triton pins. For nightly builds, each repo is resolved at its nightly
(or master) branch independently.

Usage::

    # Generate manifests for all default pytorch refs:
    python generate_pytorch_manifest_upfront.py \
        --rocm-version 7.13.0a20260501 \
        --manifest-dir /tmp/manifests

    # Generate for a single ref:
    python generate_pytorch_manifest_upfront.py \
        --rocm-version 7.13.0a20260501 \
        --manifest-dir /tmp/manifests \
        --pytorch-git-refs "release/2.11"
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

_BUILD_TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BUILD_TOOLS_DIR))

from github_actions.github_actions_api import (
    gha_fetch_file_contents,
    gha_resolve_git_ref,
)
from github_actions.manifest_utils import (
    log,
    normalize_ref_for_filename,
)

logger = logging.getLogger(__name__)

# Default pytorch refs to generate manifests for.
DEFAULT_PYTORCH_GIT_REFS = [
    "release/2.8",
    "release/2.9",
    "release/2.10",
    "release/2.11",
    "nightly",
]

# Repo configuration: (owner/repo, default nightly branch)
REPO_CONFIG = {
    "pytorch": {
        "stable_repo": "ROCm/pytorch",
        "nightly_repo": "pytorch/pytorch",
        "nightly_branch": "nightly",
        "version_file": "version.txt",
    },
    "pytorch_audio": {
        "stable_repo": "pytorch/audio",
        "nightly_repo": "pytorch/audio",
        "nightly_branch": "nightly",
        "version_file": "version.txt",
    },
    "pytorch_vision": {
        "stable_repo": "pytorch/vision",
        "nightly_repo": "pytorch/vision",
        "nightly_branch": "nightly",
        "version_file": "version.txt",
    },
    "triton": {
        "stable_repo": "ROCm/triton",
        "nightly_repo": "ROCm/triton",
        "nightly_branch": None,  # Resolved from pytorch's triton_version.txt
        "version_file": None,  # Version is complex (git hash); not computed here
    },
    "apex": {
        "stable_repo": "ROCm/apex",
        "nightly_repo": "ROCm/apex",
        "nightly_branch": "master",
        "version_file": "version.txt",
    },
}


def _resolve_ref(repo: str, ref: str) -> str:
    """Resolve a git ref and log the result."""
    sha = gha_resolve_git_ref(repo, ref)
    log(f"  Resolved {repo}@{ref} -> {sha[:12]}")
    return sha


def parse_related_commits(content: str) -> dict[str, dict[str, str]]:
    """Parse the ROCm/pytorch ``related_commits`` file.

    Returns a dict keyed by project name (e.g. "torchaudio"), with values
    containing "origin" and "commit" fields. Only ``centos`` entries are
    returned (used for both Linux and Windows builds).
    """
    pins: dict[str, dict[str, str]] = {}
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("|")
        if len(parts) != 6:
            log(f"  WARNING: Could not parse related_commits line: {line}")
            continue
        rec_os, _source, rec_project, _branch, rec_commit, rec_origin = parts
        if rec_os == "centos":
            pins[rec_project] = {"origin": rec_origin, "commit": rec_commit}
    return pins


def resolve_nightly_sources(
    pytorch_ref: str,
) -> dict[str, dict[str, str]]:
    """Resolve source commits for a nightly build.

    Each repo is resolved at its nightly branch independently.
    """
    sources: dict[str, dict[str, str]] = {}

    for name, config in REPO_CONFIG.items():
        repo = config["nightly_repo"]
        branch = config["nightly_branch"]

        if name == "pytorch":
            branch = pytorch_ref  # "nightly"

        if branch is None:
            # triton: resolve from pytorch's triton pin file
            continue

        sha = _resolve_ref(repo, branch)
        entry: dict[str, str] = {
            "commit": sha,
            "repo": f"https://github.com/{repo}.git",
            "branch": branch,
        }
        sources[name] = entry

    # Resolve triton from pytorch's pin file
    pytorch_sha = sources["pytorch"]["commit"]
    pytorch_repo = REPO_CONFIG["pytorch"]["nightly_repo"]
    triton_version_content = gha_fetch_file_contents(
        pytorch_repo, ".ci/docker/triton_version.txt", pytorch_sha
    )
    triton_version = triton_version_content.strip()
    triton_major, triton_minor, *_ = triton_version.split(".")
    triton_branch = f"release/{triton_major}.{triton_minor}.x"
    triton_repo = REPO_CONFIG["triton"]["nightly_repo"]
    triton_sha = _resolve_ref(triton_repo, triton_branch)
    sources["triton"] = {
        "commit": triton_sha,
        "repo": f"https://github.com/{triton_repo}.git",
        "branch": triton_branch,
    }

    return sources


def resolve_stable_sources(
    pytorch_ref: str,
) -> dict[str, dict[str, str]]:
    """Resolve source commits for a stable release.

    Pytorch is resolved at the given ref in ROCm/pytorch. Other repos
    are resolved from the ``related_commits`` file in that checkout.
    """
    pytorch_repo = REPO_CONFIG["pytorch"]["stable_repo"]
    pytorch_sha = _resolve_ref(pytorch_repo, pytorch_ref)

    sources: dict[str, dict[str, str]] = {
        "pytorch": {
            "commit": pytorch_sha,
            "repo": f"https://github.com/{pytorch_repo}.git",
            "branch": pytorch_ref,
        },
    }

    # Fetch related_commits to resolve other repos
    try:
        related_content = gha_fetch_file_contents(
            pytorch_repo, "related_commits", pytorch_sha
        )
    except Exception as e:
        log(f"  WARNING: Could not fetch related_commits: {e}")
        log("  Falling back to nightly branches for sub-repos")
        # Fall back: resolve each at its default branch
        for name in ["pytorch_audio", "pytorch_vision", "apex", "triton"]:
            config = REPO_CONFIG[name]
            repo = config["stable_repo"]
            branch = config["nightly_branch"] or "main"
            sha = _resolve_ref(repo, branch)
            sources[name] = {
                "commit": sha,
                "repo": f"https://github.com/{repo}.git",
                "branch": branch,
            }
        return sources

    pins = parse_related_commits(related_content)

    # Resolve audio, vision, apex from related_commits
    project_name_map = {
        "torchaudio": "pytorch_audio",
        "torchvision": "pytorch_vision",
        "apex": "apex",
    }

    for project_name, internal_name in project_name_map.items():
        if project_name in pins:
            pin = pins[project_name]
            sources[internal_name] = {
                "commit": pin["commit"],
                "repo": pin["origin"],
            }
        else:
            config = REPO_CONFIG[internal_name]
            repo = config["stable_repo"]
            branch = config["nightly_branch"] or "main"
            sha = _resolve_ref(repo, branch)
            sources[internal_name] = {
                "commit": sha,
                "repo": f"https://github.com/{repo}.git",
                "branch": branch,
            }

    # Resolve triton from ci_commit_pins (not related_commits).
    # This matches the logic in pytorch_triton_repo.py.
    triton_repo = REPO_CONFIG["triton"]["stable_repo"]
    try:
        triton_pin = gha_fetch_file_contents(
            pytorch_repo, ".ci/docker/ci_commit_pins/triton.txt", pytorch_sha
        ).strip()
        log(f"  Triton pin from ci_commit_pins: {triton_pin[:12]}")
        sources["triton"] = {
            "commit": triton_pin,
            "repo": f"https://github.com/{triton_repo}.git",
        }
    except Exception as e:
        log(f"  WARNING: Could not fetch triton pin: {e}")
        sha = _resolve_ref(triton_repo, "main")
        sources["triton"] = {
            "commit": sha,
            "repo": f"https://github.com/{triton_repo}.git",
            "branch": "main",
        }

    return sources


def fetch_versions(
    sources: dict[str, dict[str, str]],
    version_suffix: str,
) -> None:
    """Fetch version.txt from each repo and add computed versions to sources.

    Modifies ``sources`` in place, adding a "version" key to each entry
    that has a version_file configured.
    """
    for name, config in REPO_CONFIG.items():
        version_file = config.get("version_file")
        if version_file is None or name not in sources:
            continue

        entry = sources[name]
        repo_url = entry["repo"]
        # Extract owner/repo from URL like https://github.com/ROCm/pytorch.git
        repo = repo_url.replace("https://github.com/", "").rstrip(".git").rstrip("/")
        commit = entry["commit"]

        try:
            content = gha_fetch_file_contents(repo, version_file, commit)
            base_version = content.strip()
            full_version = f"{base_version}{version_suffix}"
            entry["version"] = full_version
            log(f"  {name}: {base_version} -> {full_version}")
        except Exception as e:
            log(f"  WARNING: Could not fetch {version_file} for {name}: {e}")


def generate_manifest(
    *,
    pytorch_git_ref: str,
    rocm_version: str,
    version_suffix: str,
) -> dict[str, object]:
    """Generate a single manifest for one pytorch_git_ref."""
    log(f"Generating manifest for pytorch_git_ref={pytorch_git_ref}")

    is_nightly = pytorch_git_ref == "nightly"

    if is_nightly:
        sources = resolve_nightly_sources(pytorch_git_ref)
    else:
        sources = resolve_stable_sources(pytorch_git_ref)

    fetch_versions(sources, version_suffix)

    # Add therock info from the CI environment
    server_url = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    repo = os.environ.get("GITHUB_REPOSITORY", "ROCm/TheRock")
    sha = os.environ.get("GITHUB_SHA", "unknown")
    ref = os.environ.get("GITHUB_REF", "")

    therock_branch = "unknown"
    if ref.startswith("refs/heads/"):
        therock_branch = ref[len("refs/heads/") :]
    elif ref:
        therock_branch = ref

    manifest: dict[str, object] = {}
    manifest.update(sources)
    manifest["therock"] = {
        "commit": sha,
        "repo": f"{server_url}/{repo}.git",
        "branch": therock_branch,
    }
    manifest["rocm_version"] = rocm_version
    manifest["version_suffix"] = version_suffix

    return manifest


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        description="Generate PyTorch build manifests upfront (before checkout/build)"
    )
    parser.add_argument(
        "--rocm-version",
        required=True,
        help="ROCm version (e.g. 7.13.0a20260501)",
    )
    parser.add_argument(
        "--version-suffix",
        required=True,
        help="Version suffix for wheel versions (e.g. +rocm7.13.0a20260501)",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        required=True,
        help="Output directory for manifest JSON files",
    )
    parser.add_argument(
        "--pytorch-git-refs",
        type=str,
        default="",
        help=(
            "Space-separated list of pytorch git refs to generate manifests for. "
            "Empty means all default refs."
        ),
    )
    args = parser.parse_args(argv)

    refs = (
        args.pytorch_git_refs.split()
        if args.pytorch_git_refs
        else DEFAULT_PYTORCH_GIT_REFS
    )
    log(f"ROCm version: {args.rocm_version}")
    log(f"Version suffix: {args.version_suffix}")
    log(f"PyTorch refs: {refs}")
    log("")

    manifest_dir = args.manifest_dir.resolve()
    manifest_dir.mkdir(parents=True, exist_ok=True)

    manifest_paths: list[str] = []

    for ref in refs:
        manifest = generate_manifest(
            pytorch_git_ref=ref,
            rocm_version=args.rocm_version,
            version_suffix=args.version_suffix,
        )

        filename = f"therock-manifest_torch_{normalize_ref_for_filename(ref)}.json"
        out_path = manifest_dir / filename
        out_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )
        manifest_paths.append(str(out_path))
        log(f"Wrote {out_path}")
        log("")

    # Write summary of generated manifests
    log(f"Generated {len(manifest_paths)} manifest(s)")
    for p in manifest_paths:
        log(f"  {p}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
