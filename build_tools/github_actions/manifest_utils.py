#!/usr/bin/env python3
"""
Helpers for manifest generation.
"""

from dataclasses import dataclass
from pathlib import Path
import shlex
import subprocess
import sys


def log(*args, **kwargs) -> None:
    """Consistent logging helper for manifest generation scripts."""
    print(*args, **kwargs)
    sys.stdout.flush()


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
    log(f"++ Exec [{cwd}]$ {shlex.join(args)}")
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
    log(f"++ Exec [{cwd}]$ {shlex.join(args)}")
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


def normalize_python_version_for_filename(python_version: str) -> str:
    """Normalize python version strings for filenames.

    Examples:
      "py3.12" -> "3.12"
      "3.12"   -> "3.12"
    """
    py = python_version.strip()
    if py.startswith("py"):
        py = py[2:]
    return py


def normalize_ref_for_filename(ref: str) -> str:
    """Normalize a git ref for filenames by replacing '/' with '-'.

    Examples:
      "nightly"                -> "nightly"
      "release/0.4.28"         -> "release-0.4.28"
      "users/alice/experiment" -> "users-alice-experiment"
    """
    return ref.replace("/", "-")
