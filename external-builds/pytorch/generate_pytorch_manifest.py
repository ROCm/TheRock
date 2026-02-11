#!/usr/bin/env python3
"""
Generate a manifest for PyTorch external builds.

Writes a JSON manifest containing:
  - pytorch/pytorch_audio/pytorch_vision(/triton): git commit + origin repo
  - therock: repo + commit + branch from GitHub Actions env (best-effort)

Filename format:
  therock-manifest_torch_py<python_version>_<release_track>.json
"""

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


@dataclass(frozen=True)
class GitSourceInfo:
    """Git commit and origin repo for a source checkout."""

    commit: str
    repo: str

    def to_dict(self) -> dict[str, str]:
        return {"commit": self.commit, "repo": self.repo}


def capture(args: list[str | Path], cwd: Path) -> str:
    args = [str(arg) for arg in args]
    print(f"++ Exec [{cwd}]$ {shlex.join(args)}")
    return (
        subprocess.check_output(
            args,
            cwd=str(cwd),
            stdin=subprocess.DEVNULL,
        )
        .decode()
        .strip()
    )


def git_head(dirpath: Path, *, label: str) -> GitSourceInfo:
    """Return commit + origin repo for a git checkout."""
    dirpath = dirpath.resolve()

    if not dirpath.exists():
        raise FileNotFoundError(
            f"{label}: directory does not exist: {dirpath}\n"
            "This indicates a misconfigured workflow or incomplete checkout."
        )

    if not (dirpath / ".git").exists():
        raise FileNotFoundError(
            f"{label}: not a git checkout (missing .git): {dirpath}\n"
            "Manifest generation requires git commit hash and origin repo."
        )

    commit = capture(["git", "rev-parse", "HEAD"], cwd=dirpath)
    repo = capture(["git", "remote", "get-url", "origin"], cwd=dirpath)
    return GitSourceInfo(commit=commit, repo=repo)


def normalize_release_track(pytorch_git_ref: str) -> str:
    """Normalize a git ref for filenames by replacing path separators.

    Examples:
      nightly                 -> nightly
      release/2.7             -> release-2.7
      users/alice/experiment  -> users-alice-experiment
    """
    return pytorch_git_ref.replace("/", "-")


def normalize_py(python_version: str) -> str:
    """Normalize python version for filenames: 'py3.11' -> '3.11'."""
    py = python_version.strip()
    if py.startswith("py"):
        py = py[2:]
    return py


def manifest_filename(*, python_version: str, pytorch_git_ref: str) -> str:
    py = normalize_py(python_version)
    track = normalize_release_track(pytorch_git_ref)
    return f"therock-manifest_torch_py{py}_{track}.json"


def build_sources(
    *,
    pytorch_dir: Path,
    pytorch_audio_dir: Path,
    pytorch_vision_dir: Path,
    triton_dir: Path | None,
) -> dict[str, dict[str, str]]:
    sources: dict[str, dict[str, str]] = {
        "pytorch": git_head(pytorch_dir, label="pytorch").to_dict(),
        "pytorch_audio": git_head(pytorch_audio_dir, label="pytorch_audio").to_dict(),
        "pytorch_vision": git_head(pytorch_vision_dir, label="pytorch_vision").to_dict(),
    }
    if triton_dir is not None:
        sources["triton"] = git_head(triton_dir, label="triton").to_dict()
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
        "repo": therock_repo,
        "commit": therock_commit,
        "branch": therock_branch,
    }
    return manifest


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
    ap.add_argument("--pytorch-dir", type=Path, required=True)
    ap.add_argument("--pytorch-audio-dir", type=Path, required=True)
    ap.add_argument("--pytorch-vision-dir", type=Path, required=True)
    ap.add_argument(
        "--triton-dir",
        type=Path,
        help="Optional triton checkout (Linux only).",
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

    sources = build_sources(
        pytorch_dir=args.pytorch_dir,
        pytorch_audio_dir=args.pytorch_audio_dir,
        pytorch_vision_dir=args.pytorch_vision_dir,
        triton_dir=args.triton_dir,
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

    manifest = build_manifest(
        sources=sources,
        therock_repo=therock_repo,
        therock_commit=therock_commit,
        therock_branch=therock_branch,
    )

    out_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    print(f"[pytorch-sources-manifest] wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
