#!/usr/bin/env python3
"""
Generate a manifest for JAX external builds.

Writes a JSON manifest containing:
  - jax: git commit + origin repo (+ branch best-effort)
  - therock: repo + commit + branch from GitHub Actions env (best-effort)

Filename format:
  therock-manifest_jax_py<python_version>_<jax_track>.json
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
    branch: str | None = None

    def to_dict(self) -> dict[str, str]:
        d = {"commit": self.commit, "repo": self.repo}
        if self.branch is not None:
            d["branch"] = self.branch
        return d


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


def capture_optional(args: list[str | Path], cwd: Path) -> str | None:
    """Like capture(), but returns None on failure."""
    args = [str(arg) for arg in args]
    print(f"++ Exec [{cwd}]$ {shlex.join(args)}")
    try:
        out = (
            subprocess.check_output(
                args,
                cwd=str(cwd),
                stdin=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return out or None


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


def git_branch_best_effort(dirpath: Path) -> str | None:
    """Return current branch name if on a real branch; None if detached/unknown."""
    dirpath = dirpath.resolve()

    # Most reliable when on a branch; fails in detached HEAD.
    b = capture_optional(
        ["git", "symbolic-ref", "--quiet", "--short", "HEAD"], cwd=dirpath
    )
    if b and b != "HEAD":
        return b

    # Fallback. Returns empty on detached.
    b = capture_optional(["git", "branch", "--show-current"], cwd=dirpath)
    if b and b != "HEAD":
        return b

    return None


def resolve_branch(*, inferred: str | None, provided: str | None) -> str | None:
    """Choose inferred branch if available; else provided; else None."""
    if inferred:
        return inferred
    if provided:
        return provided
    return None


def normalize_release_track(track: str) -> str:
    """Normalize a git ref for filenames by replacing path separators.

    Examples:
      nightly                 -> nightly
      release/0.4.28          -> release-0.4.28
      users/alice/experiment  -> users-alice-experiment
    """
    return track.replace("/", "-")


def normalize_py(python_version: str) -> str:
    """Normalize python version for filenames: 'py3.11' -> '3.11'."""
    py = python_version.strip()
    if py.startswith("py"):
        py = py[2:]
    return py


def manifest_filename(*, python_version: str, jax_track: str) -> str:
    py = normalize_py(python_version)
    track = normalize_release_track(jax_track)
    return f"therock-manifest_jax_py{py}_{track}.json"


def build_sources(
    *, jax_dir: Path, jax_git_ref: str | None
) -> dict[str, dict[str, str]]:
    jax = git_head(jax_dir, label="jax")

    jax_branch = resolve_branch(
        inferred=git_branch_best_effort(jax_dir),
        provided=jax_git_ref,
    )

    return {
        "jax": GitSourceInfo(
            commit=jax.commit, repo=jax.repo, branch=jax_branch
        ).to_dict(),
    }


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
    jax_dir: Path,
    python_version: str,
    jax_track: str,
    jax_git_ref: str | None,
) -> tuple[str, dict[str, object]]:
    sources = build_sources(jax_dir=jax_dir, jax_git_ref=jax_git_ref)

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

    name = manifest_filename(
        python_version=python_version,
        jax_track=jax_track,
    )

    manifest = build_manifest(
        sources=sources,
        therock_repo=therock_repo,
        therock_commit=therock_commit,
        therock_branch=therock_branch,
    )
    return name, manifest


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate JAX manifest.")
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
        "--jax-track",
        required=True,
        help="JAX track/ref for manifest naming (e.g. nightly or release/0.4.28).",
    )
    ap.add_argument(
        "--jax-git-ref",
        help="Optional ref for jax branch field (used if detached).",
    )
    ap.add_argument("--jax-dir", type=Path, required=True)
    return ap.parse_args(argv)


def main(argv: list[str]) -> None:
    args = parse_args(argv)

    manifest_dir = args.manifest_dir.resolve()
    manifest_dir.mkdir(parents=True, exist_ok=True)

    name, manifest = generate_manifest_dict(
        jax_dir=args.jax_dir,
        python_version=args.python_version,
        jax_track=args.jax_track,
        jax_git_ref=args.jax_git_ref,
    )

    out_path = manifest_dir / name
    out_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )

    if not out_path.exists():
        raise RuntimeError(f"Failed to write manifest: {out_path}")
    if out_path.stat().st_size == 0:
        raise RuntimeError(f"Manifest is empty: {out_path}")

    print(f"[jax-sources-manifest] wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
