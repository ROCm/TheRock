#!/usr/bin/env python3
"""
Generate a manifest for PyTorch external builds.

Writes a JSON manifest containing:
  - sources: git commit + remote for each provided source checkout
  - therock: repo/ref/commit from GitHub Actions env (or user-provided env)

Filename format
  therock-manifest_torch_py<python_version>_<release_track>.json
"""

import argparse
from dataclasses import dataclass
import json
import os
import subprocess
from pathlib import Path


@dataclass(frozen=True)
class GitSourceInfo:
    """Git commit and origin remote for a source checkout."""

    commit: str
    remote: str

    def to_dict(self) -> dict[str, str]:
        return {"commit": self.commit, "remote": self.remote}


def capture(cmd: list[str], *, cwd: Path) -> str:
    try:
        return subprocess.check_output(
            cmd,
            cwd=str(cwd),
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()
    except subprocess.CalledProcessError as e:
        output = (e.output or "").strip()
        raise RuntimeError(
            f"Command failed ({e.returncode}): {' '.join(cmd)}\n"
            + (f"Output:\n{output}" if output else "")
        ) from e


def git_head(dirpath: Path, *, label: str) -> GitSourceInfo:
    """Return commit + origin remote for a git checkout."""
    dirpath = dirpath.resolve()

    if not dirpath.exists():
        raise FileNotFoundError(
            f"{label}: directory does not exist: {dirpath}\n"
            "This indicates a misconfigured workflow or incomplete checkout."
        )

    if not (dirpath / ".git").exists():
        raise FileNotFoundError(
            f"{label}: not a git checkout (missing .git): {dirpath}\n"
            "Manifest generation requires git commit hash and origin remote."
        )

    commit = capture(["git", "rev-parse", "HEAD"], cwd=dirpath)
    remote = capture(["git", "remote", "get-url", "origin"], cwd=dirpath)
    return GitSourceInfo(commit=commit, remote=remote)


def normalize_release_track(pytorch_git_ref: str) -> str:
    if pytorch_git_ref == "nightly":
        return "nightly"
    if pytorch_git_ref.startswith("release/"):
        return pytorch_git_ref.replace("/", "-", 1)
    return pytorch_git_ref.replace("/", "-")


def normalize_py(python_version: str) -> str:
    py = python_version.strip()
    if py.startswith("py"):
        py = py[2:]
    return py


def require_env(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {name}\n"
            "Set it (or run under GitHub Actions) to populate the 'therock' block."
        )
    return value


def main() -> None:
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
        "--triton-dir", type=Path, help="Optional triton checkout (Linux only)."
    )

    args = ap.parse_args()

    manifest_dir = args.manifest_dir.resolve()
    manifest_dir.mkdir(parents=True, exist_ok=True)

    py = normalize_py(args.python_version)
    release_track = normalize_release_track(args.pytorch_git_ref)
    manifest_name = f"therock-manifest_torch_py{py}_{release_track}.json"
    out_path = manifest_dir / manifest_name

    sources = {
        "pytorch": git_head(args.pytorch_dir, label="pytorch").to_dict(),
        "pytorch_audio": git_head(
            args.pytorch_audio_dir, label="pytorch_audio"
        ).to_dict(),
        "pytorch_vision": git_head(
            args.pytorch_vision_dir, label="pytorch_vision"
        ).to_dict(),
    }

    if args.triton_dir is not None:
        sources["triton"] = git_head(args.triton_dir, label="triton").to_dict()

    server_url = require_env("GITHUB_SERVER_URL")
    repo = require_env("GITHUB_REPOSITORY")
    sha = require_env("GITHUB_SHA")
    ref = require_env("GITHUB_REF")

    manifest = {
        "sources": sources,
        "therock": {
            "repo": f"{server_url}/{repo}",
            "commit": sha,
            "ref": ref,
        },
    }

    out_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[pytorch-sources-manifest] wrote {out_path}")


if __name__ == "__main__":
    main()
