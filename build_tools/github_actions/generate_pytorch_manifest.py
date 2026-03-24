#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Generate a manifest for PyTorch external builds.

Writes a JSON manifest containing:
  - pytorch/pytorch_audio/pytorch_vision(/triton)(/apex): git commit + origin repo (+ branch best-effort)
  - therock: repo + commit + branch from GitHub Actions env (best-effort)

Filename format:
  therock-manifest_torch_py<python_version>_<release_track>.json
"""

import argparse
import json
import os
from pathlib import Path
import sys

_BUILD_TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BUILD_TOOLS_DIR))

from github_actions.manifest_utils import (
    GitSourceInfo,
    git_branch_best_effort,
    git_head,
    log,
    normalize_python_version_for_filename,
    normalize_ref_for_filename,
    resolve_branch,
)


def manifest_filename(*, python_version: str, pytorch_git_ref: str) -> str:
    py = normalize_python_version_for_filename(python_version)
    track = normalize_ref_for_filename(pytorch_git_ref)
    return f"therock-manifest_torch_py{py}_{track}.json"


def build_sources(
    *,
    pytorch_dir: Path,
    pytorch_audio_dir: Path,
    pytorch_vision_dir: Path,
    triton_dir: Path | None,
    apex_dir: Path | None,
    pytorch_git_ref: str,
    pytorch_audio_git_ref: str | None,
    pytorch_vision_git_ref: str | None,
    triton_git_ref: str | None,
    apex_git_ref: str | None,
) -> dict[str, dict[str, str]]:
    pt = git_head(pytorch_dir, label="pytorch")
    aud = git_head(pytorch_audio_dir, label="pytorch_audio")
    vis = git_head(pytorch_vision_dir, label="pytorch_vision")

    pt_branch = resolve_branch(
        inferred=git_branch_best_effort(pytorch_dir),
        provided=pytorch_git_ref,
    )
    aud_branch = resolve_branch(
        inferred=git_branch_best_effort(pytorch_audio_dir),
        provided=pytorch_audio_git_ref,
    )
    vis_branch = resolve_branch(
        inferred=git_branch_best_effort(pytorch_vision_dir),
        provided=pytorch_vision_git_ref,
    )

    sources: dict[str, dict[str, str]] = {
        "pytorch": GitSourceInfo(
            commit=pt.commit, repo=pt.repo, branch=pt_branch
        ).to_dict(),
        "pytorch_audio": GitSourceInfo(
            commit=aud.commit, repo=aud.repo, branch=aud_branch
        ).to_dict(),
        "pytorch_vision": GitSourceInfo(
            commit=vis.commit, repo=vis.repo, branch=vis_branch
        ).to_dict(),
    }

    if triton_dir is not None:
        tri = git_head(triton_dir, label="triton")
        tri_branch = resolve_branch(
            inferred=git_branch_best_effort(triton_dir),
            provided=triton_git_ref,
        )
        sources["triton"] = GitSourceInfo(
            commit=tri.commit, repo=tri.repo, branch=tri_branch
        ).to_dict()

    if apex_dir is not None:
        ax = git_head(apex_dir, label="apex")
        ax_branch = resolve_branch(
            inferred=git_branch_best_effort(apex_dir),
            provided=apex_git_ref,
        )
        sources["apex"] = GitSourceInfo(
            commit=ax.commit, repo=ax.repo, branch=ax_branch
        ).to_dict()

    return sources


def build_manifest(
    *,
    sources: dict[str, dict[str, str]],
    therock_repo: str,
    therock_commit: str,
    therock_branch: str,
) -> dict[str, object]:
    # Flattened schema: top-level source keys, plus therock last.
    manifest: dict[str, object] = {}
    manifest.update(sources)
    manifest["therock"] = {
        "commit": therock_commit,
        "repo": therock_repo,
        "branch": therock_branch,
    }
    return manifest


def generate_manifest_dict(
    *,
    pytorch_dir: Path,
    pytorch_audio_dir: Path,
    pytorch_vision_dir: Path,
    triton_dir: Path | None,
    apex_dir: Path | None,
    pytorch_git_ref: str,
    pytorch_audio_git_ref: str | None,
    pytorch_vision_git_ref: str | None,
    triton_git_ref: str | None,
    apex_git_ref: str | None,
) -> dict[str, object]:
    """Generate the manifest dictionary"""
    sources = build_sources(
        pytorch_dir=pytorch_dir,
        pytorch_audio_dir=pytorch_audio_dir,
        pytorch_vision_dir=pytorch_vision_dir,
        triton_dir=triton_dir,
        apex_dir=apex_dir,
        pytorch_git_ref=pytorch_git_ref,
        pytorch_audio_git_ref=pytorch_audio_git_ref,
        pytorch_vision_git_ref=pytorch_vision_git_ref,
        triton_git_ref=triton_git_ref,
        apex_git_ref=apex_git_ref,
    )

    server_url = os.environ.get("GITHUB_SERVER_URL")
    repo = os.environ.get("GITHUB_REPOSITORY")
    sha = os.environ.get("GITHUB_SHA")
    ref = os.environ.get("GITHUB_REF")

    therock_repo = "unknown"
    if server_url and repo:
        therock_repo = f"{server_url}/{repo}.git"

    therock_commit = sha or "unknown"

    therock_branch = "unknown"
    if ref:
        if ref.startswith("refs/heads/"):
            therock_branch = ref[len("refs/heads/") :]
        else:
            # Could be refs/tags/<tag>, refs/pull/<id>/merge, or a SHA, etc.
            therock_branch = ref

    return build_manifest(
        sources=sources,
        therock_repo=therock_repo,
        therock_commit=therock_commit,
        therock_branch=therock_branch,
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate PyTorch manifest.")
    ap.add_argument(
        "--manifest-dir",
        type=Path,
        required=True,
        help="Output directory for the manifest JSON.",
    )
    ap.add_argument(
        "--python-version",
        required=True,
        help="Python version for manifest naming (e.g. 3.11 or py3.11).",
    )
    ap.add_argument(
        "--pytorch-git-ref",
        required=True,
        help="PyTorch ref for manifest naming (e.g. nightly or release/2.8).",
    )
    ap.add_argument(
        "--pytorch-audio-git-ref",
        help="Optional ref for pytorch_audio branch field (used if detached).",
    )
    ap.add_argument(
        "--pytorch-vision-git-ref",
        help="Optional ref for pytorch_vision branch field (used if detached).",
    )
    ap.add_argument(
        "--triton-git-ref",
        help="Optional ref for triton branch field (used if detached).",
    )
    ap.add_argument(
        "--apex-git-ref",
        help="Optional ref for apex branch field (used if detached).",
    )
    ap.add_argument("--pytorch-dir", type=Path, required=True)
    ap.add_argument("--pytorch-audio-dir", type=Path, required=True)
    ap.add_argument("--pytorch-vision-dir", type=Path, required=True)
    ap.add_argument(
        "--triton-dir",
        type=Path,
        help="Optional triton checkout (Linux only).",
    )
    ap.add_argument(
        "--apex-dir",
        type=Path,
        help="Optional apex checkout (Linux only).",
    )
    return ap.parse_args(argv)


def main(argv: list[str]) -> None:
    args = parse_args(argv)

    manifest_dir = args.manifest_dir.resolve()
    manifest_dir.mkdir(parents=True, exist_ok=True)

    name = manifest_filename(
        python_version=args.python_version,
        pytorch_git_ref=args.pytorch_git_ref,
    )
    out_path = manifest_dir / name

    manifest = generate_manifest_dict(
        pytorch_dir=args.pytorch_dir,
        pytorch_audio_dir=args.pytorch_audio_dir,
        pytorch_vision_dir=args.pytorch_vision_dir,
        triton_dir=args.triton_dir,
        apex_dir=args.apex_dir,
        pytorch_git_ref=args.pytorch_git_ref,
        pytorch_audio_git_ref=args.pytorch_audio_git_ref,
        pytorch_vision_git_ref=args.pytorch_vision_git_ref,
        triton_git_ref=args.triton_git_ref,
        apex_git_ref=args.apex_git_ref,
    )

    out_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    log(f"[pytorch-sources-manifest] wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
