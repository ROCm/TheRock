#!/usr/bin/env python3
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
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


def list_submodules_via_gitconfig(repo_dir: Path):
    """
    Read path/url/branch for all submodules from .gitmodules using a single git-config call.
    Returns: [{name, path, url, branch}]
    """
    gitconfig_output = _run(
        [
            "git",
            "config",
            "-f",
            ".gitmodules",
            "--get-regexp",
            r"^submodule\..*\.(path|url|branch)$",
        ],
        cwd=repo_dir,
        check=False,
    )
    if not gitconfig_output:
        return []

    submodules_by_name = (
        {}
    )  # name -> {"name": ..., "path": ..., "url": ..., "branch": ...}
    for line in gitconfig_output.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            full_key, raw_value = line.split(None, 1)
        except ValueError:
            continue

        m = re.match(r"^submodule\.(?P<name>.+)\.(?P<attr>path|url|branch)$", full_key)
        if not m:
            continue

        name = m.group("name")
        attr = m.group("attr")
        value = raw_value.strip()

        rec = submodules_by_name.setdefault(
            name, {"name": name, "path": None, "url": None, "branch": None}
        )
        rec[attr] = value

    results = [
        {"name": n, "path": r["path"], "url": r["url"], "branch": r["branch"]}
        for n, r in submodules_by_name.items()
        if r["path"]
    ]
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


def read_rocm_version(repo_root: Path) -> str | None:
    """
    Read ROCm version from <repo>/version.json with key 'rocm-version'.
    Returns None if file/key is missing or unreadable.
    """
    path = repo_root / "version.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        # get 'rocm-version'
        v = data.get("rocm-version")
        if isinstance(v, str):
            v = v.strip()
        return v or None
    except Exception:
        return None


def build_manifest_schema(repo_root: Path, the_rock_commit: str) -> dict:
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

    # Values pulled from env or system; only include if non-empty.
    artifact_group = os.getenv("ARTIFACT_GROUP")
    run_id = os.getenv("GITHUB_RUN_ID")
    job_id = os.getenv("GITHUB_JOB")
    # Prefer version.json; fall back to env ROCM_VERSION if provided.
    rocm_version = read_rocm_version(repo_root) or os.getenv("ROCM_VERSION")
    plat = (os.getenv("RUNNER_OS") or platform.system() or "").lower()
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    build_ctx = {}
    if run_id:
        build_ctx["run_id"] = run_id
    if plat:
        build_ctx["platform"] = plat
    if artifact_group:
        build_ctx["artifact_group"] = artifact_group
    if rocm_version:
        build_ctx["rocm_version"] = rocm_version
    if timestamp:
        build_ctx["timestamp"] = timestamp
    if job_id:
        build_ctx["job_id"] = job_id

    return {
        "the_rock_commit": the_rock_commit,
        "build_context": build_ctx,
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
