#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys


def _run(cmd, cwd=None, check=True) -> str:
    if isinstance(cmd, str):
        cmd = shlex.split(cmd)
    res = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=False)
    if check and res.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{res.stderr}")
    return res.stdout.strip()


def git_root() -> Path:
    """
    Determine the repo root strictly from this script's location:
      <repo>/build_tools/generate_therock_manifest.py  ->  <repo>
    """
    here = Path(__file__).resolve()
    repo_root = here.parents[1]  # .../build_tools -> repo root
    if not ((repo_root / ".git").exists() or (repo_root / ".gitmodules").exists()):
        raise RuntimeError(
            f"Could not locate repo root at {repo_root}. "
            "Expected this script to live under <repo>/build_tools/."
        )
    return repo_root


def list_submodules_from_gitmodules_at_commit(repo_dir: Path, commit: str = "HEAD"):
    """
    Read path/url/branch for all submodules from .gitmodules at a specific commit.

    Uses `git config --blob` to parse .gitmodules directly from git's object database.
    This approach lets git handle all config file parsing edge cases and works with any valid submodule names.

    Args:
        repo_dir: Path to the repository root.
        commit: The commit/ref to read .gitmodules from (default: HEAD).
                This is important when inspecting historical commits where submodules
                may have been added or removed since.

    Returns: [{name, path, url, branch}]

    Raises:
        RuntimeError: If git command fails for reasons other than missing .gitmodules.
    """
    cmd = [
        "git",
        "config",
        "--blob",
        f"{commit}:.gitmodules",
        "--get-regexp",
        r"^submodule\.",
    ]
    result = subprocess.run(
        cmd, cwd=repo_dir, text=True, capture_output=True, check=False
    )

    if result.returncode != 0:
        stderr = result.stderr.strip()
        # Exit code 1 with no stderr means no matches (no .gitmodules or no submodule entries)
        # Exit code 1 with stderr means an actual error (invalid commit, etc.)
        if stderr:
            raise RuntimeError(f"Git command failed: {' '.join(cmd)}\n{stderr}")
        return []

    submodules_by_name = {}

    # Output format: submodule.<name>.<key> <value>
    # Example: submodule.half.path base/half
    # The regex uses (.+) for name to handle submodule names containing dots.
    # The explicit \.(path|url|branch) ensures we match the last occurrence,
    # correctly extracting names like "foo.bar" from "submodule.foo.bar.path".
    pattern = re.compile(r"^submodule\.(.+)\.(path|url|branch)\s+(.+)$")

    for line in result.stdout.strip().splitlines():
        match = pattern.match(line)
        if match:
            name, key, value = match.groups()
            if name not in submodules_by_name:
                submodules_by_name[name] = {
                    "name": name,
                    "path": None,
                    "url": None,
                    "branch": None,
                }
            submodules_by_name[name][key] = value

    # Filter out entries without a path and sort by path
    results = [r for r in submodules_by_name.values() if r["path"]]
    results.sort(key=lambda r: r["path"])
    return results


def submodule_pin(repo_dir: Path, commit: str, sub_path: str):
    """
    Read the gitlink SHA for submodule `sub_path` at `commit`.
    Uses: git ls-tree <commit> -- <path>
    """
    out = _run(["git", "ls-tree", commit, "--", sub_path], cwd=repo_dir, check=False)
    if not out:
        return None
    # Iterate over matching entries
    for line in out.splitlines():
        # An example of ls-tree output:
        # "160000 commit d777ee5b682bfabe3d4cd436fd5c7f0e0b75300e  rocm-libraries"
        parts = line.split()
        # Skip malformed records that don't match the expected format
        if len(parts) >= 3 and parts[1] == "commit":
            # The pin comes after "commit"
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


def build_manifest_schema(repo_root: Path, the_rock_commit: str) -> dict:

    # Enumerate submodules from .gitmodules at the specified commit.
    entries = list_submodules_from_gitmodules_at_commit(repo_root, the_rock_commit)

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

    return {
        "the_rock_commit": the_rock_commit,
        "submodules": rows,
    }


def write_manifest_json(out_path: Path, manifest: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser(
        description="Generate submodule pin/patch manifest for TheRock."
    )
    # make --output optional with a default message and value of None
    ap.add_argument(
        "-o",
        "--output",
        help="Output JSON path (default: <repo>/therock_manifest.json)",
        default=None,
    )
    ap.add_argument(
        "--commit", help="TheRock commit/ref to inspect (default: HEAD)", default="HEAD"
    )
    args = ap.parse_args()

    repo_root = git_root()
    the_rock_commit = _run(["git", "rev-parse", args.commit], cwd=repo_root)

    manifest = build_manifest_schema(repo_root, the_rock_commit)

    # Decide output path
    # if not provided, write to repo_root / "therock_manifest.json"
    out_path = (
        Path(args.output) if args.output else (repo_root / "therock_manifest.json")
    )

    # Write JSON
    write_manifest_json(out_path, manifest)

    print(str(out_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
