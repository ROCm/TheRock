#!/usr/bin/env python3
import argparse
from datetime import datetime, UTC
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

# -----------------------
# git helpers
# -----------------------


def _run(cmd, cwd=None, check=True) -> str:
    if isinstance(cmd, str):
        cmd = shlex.split(cmd)
    res = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=False)
    if check and res.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{res.stderr}")
    return res.stdout.strip()


def git_root() -> Path:
    return Path(_run(["git", "rev-parse", "--show-toplevel"]))


def list_submodules_via_gitconfig(repo_dir: Path):
    """
    Enumerate submodules using: git config -f .gitmodules
    Returns [{name, path, url, branch?}]
    """
    raw = _run(
        [
            "git",
            "config",
            "-f",
            ".gitmodules",
            "--get-regexp",
            r"^submodule\..*\.path$",
        ],
        cwd=repo_dir,
        check=False,
    )
    if not raw:
        return []

    out = []
    for line in raw.splitlines():
        # line: "submodule.<name>.path <path>"
        key, path = line.split(None, 1)
        m = re.match(r"^submodule\.(?P<name>.+)\.path$", key)
        if not m:
            continue
        name = m.group("name")
        url = (
            _run(
                [
                    "git",
                    "config",
                    "-f",
                    ".gitmodules",
                    "--get",
                    f"submodule.{name}.url",
                ],
                cwd=repo_dir,
                check=False,
            ).strip()
            or None
        )
        branch = (
            _run(
                [
                    "git",
                    "config",
                    "-f",
                    ".gitmodules",
                    "--get",
                    f"submodule.{name}.branch",
                ],
                cwd=repo_dir,
                check=False,
            ).strip()
            or None
        )
        out.append({"name": name, "path": path.strip(), "url": url, "branch": branch})
    return out


def submodule_pin(repo_dir: Path, commit: str, sub_path: str):
    """
    Read the gitlink SHA for submodule `sub_path` at `commit`.
    Uses: git ls-tree <commit> -- <path>
    """
    out = _run(["git", "ls-tree", commit, "--", sub_path], cwd=repo_dir, check=False)
    if not out:
        return None
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[1] == "commit":
            return parts[2]
    return None


def patches_for_submodule_by_name(repo_dir: Path, sub_name: str):
    """
    Return repo-relative patch file paths under:
      patches/amd-mainline/<sub_name>/*.patch
    """
    base = repo_dir / "patches" / "amd-mainline" / sub_name
    if not base.exists():
        return []
    return [str(p.relative_to(repo_dir)) for p in sorted(base.glob("*.patch"))]


# ---------------
# Main
# ---------------


def main():
    ap = argparse.ArgumentParser(
        description="Generate submodule pin/patch manifest for TheRock."
    )
    ap.add_argument("-o", "--output", required=True, help="Output JSON path")
    ap.add_argument(
        "--commit", help="TheRock commit/ref to inspect (default: HEAD)", default="HEAD"
    )
    args = ap.parse_args()

    repo_root = git_root()
    os.chdir(repo_root)

    # Resolve commit + short SHA
    the_rock_commit = _run(["git", "rev-parse", args.commit])
    short_sha = _run(["git", "rev-parse", "--short", the_rock_commit])

    # Enumerate submodules via .gitmodules
    entries = list_submodules_via_gitconfig(repo_root)

    # Build rows with pins (from tree) and patch lists
    rows = []
    for e in sorted(entries, key=lambda x: x["path"] or ""):
        pin = submodule_pin(repo_root, the_rock_commit, e["path"])
        rows.append(
            {
                "submodule_name": e["name"],
                "submodule_path": e["path"],
                "submodule_url": e["url"],
                "pin_sha": pin,
                "patches": patches_for_submodule_by_name(repo_root, e["name"]),
            }
        )

    manifest = {
        "the_rock_commit": the_rock_commit,
        "submodules": rows,
    }

    # Decide output path
    out_path = Path(args.output)

    # Write JSON
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(str(out_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
