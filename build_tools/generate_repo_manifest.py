#!/usr/bin/env python3
"""
Generate TheRock build manifest (JSON Format).

Schema: TheRock-Manifest.v1
- TheRock: remote, commit, tree_state, describe, gitmodules_sha256
- Environment: generated_at_utc, ci_provider, run_id, runner_os, amdgpu_families, rocm_version
- submodules[]: path, url, branch, commit
"""

import argparse
from datetime import datetime, UTC
import hashlib
import json
import os
import shlex
import subprocess
import sys

SCHEMA_VERSION = "TheRock-Manifest.v1"


def _run(cmd, cwd=None, check=True):
    if isinstance(cmd, str):
        cmd = shlex.split(cmd)
    res = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=False)
    if check and res.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{res.stderr}")
    return res.stdout.strip()


def _try(cmd, cwd=None):
    try:
        return _run(cmd, cwd=cwd, check=True)
    except Exception:
        return None


def git_root():
    return _run(["git", "rev-parse", "--show-toplevel"])


def git_tree_state():
    # empty output => clean
    dirty = _run(["git", "status", "--porcelain"])
    return "clean" if dirty == "" else "dirty"


def git_remote_origin():
    return _try(["git", "config", "--get", "remote.origin.url"])


def file_sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def read_rocm_version_from_version_json():
    try:
        with open("version.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        # key used in CMake: rocm-version
        v = data.get("rocm-version")
        return str(v) if v is not None else None
    except Exception:
        return None


def parse_gitmodules():
    """
    Parse top-level .gitmodules into: { path: { 'url': str|None, 'branch': str|None } }
    """
    out = {}
    cur = None
    if not os.path.exists(".gitmodules"):
        return out
    with open(".gitmodules", "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("[submodule"):
                cur = None
                continue
            if s.startswith("path"):
                _, v = s.split("=", 1)
                cur = v.strip()
                out.setdefault(cur, {"url": None, "branch": None})
                continue
            if cur is None:
                continue
            if s.startswith("url"):
                _, v = s.split("=", 1)
                out[cur]["url"] = v.strip()
            elif s.startswith("branch"):
                _, v = s.split("=", 1)
                out[cur]["branch"] = v.strip()
    return out


def submodule_list_status():
    """
    Returns list of {path, commit or None} using `git submodule status --recursive`.
    """
    raw = _run(["git", "submodule", "status", "--recursive"])
    items = []
    for line in raw.splitlines():
        if not line:
            continue
        # First char: ' ' initialized, '-' not yet initialized, '+' out of date, 'U' conflict
        rest = line[1:].strip()
        parts = rest.split()
        if len(parts) < 2:
            continue
        commit_token = parts[0]
        path = parts[1]
        commit = None if commit_token == "-" else commit_token
        items.append({"path": path, "commit": commit})
    return items


def main():
    ap = argparse.ArgumentParser(description="Generate TheRock JSON manifest.")
    ap.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output file path. If omitted, uses TheRock-Manifest-<UTC>-<shortsha>.json",
    )
    args = ap.parse_args()

    # Ensure we run at repo root for consistent paths
    root = git_root()
    os.chdir(root)

    # --- TheRock section ---
    rock_commit = _run(["git", "rev-parse", "HEAD"])
    rock_short = _run(["git", "rev-parse", "--short", "HEAD"])
    repo = os.getenv("GITHUB_REPOSITORY")  # "ROCm/TheRock"
    if repo:
        # avoids URL masking in logs
        rock_remote = f"github.com/{repo}.git"
    else:
        # fallback if GITHUB_REPOSITORY not on GitHub Actions
        rock_remote = git_remote_origin()
    rock_state = git_tree_state()
    rock_desc = _try(["git", "describe", "--always", "--tags", "--dirty"])

    # .gitmodules hash only
    gm_path = ".gitmodules"
    gitmodules_sha = file_sha256(gm_path) if os.path.exists(gm_path) else None

    # --- Environment section ---
    now_utc = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    ci_provider = "github-actions" if os.getenv("GITHUB_ACTIONS") else None

    # Prefer GITHUB_RUN_ID; fallback to number(.attempt)
    run_id = os.getenv("GITHUB_RUN_ID")
    if not run_id:
        run_num = os.getenv("GITHUB_RUN_NUMBER")
        run_attempt = os.getenv("GITHUB_RUN_ATTEMPT")
        if run_num and run_attempt:
            run_id = f"{run_num}.{run_attempt}"
        elif run_num:
            run_id = run_num

    # Prefer ImageOS (hosted); fallback to RUNNER_OS (self-hosted)
    runner_os = os.getenv("ImageOS") or os.getenv("RUNNER_OS")

    amdgpu_families = os.getenv("AMDGPU_FAMILIES") or os.getenv(
        "THEROCK_AMDGPU_FAMILIES"
    )
    rocm_version = read_rocm_version_from_version_json()

    # --- Submodules section ---
    gm_map = parse_gitmodules()  # {path: {url, branch}}
    status_list = submodule_list_status()  # [{path, commit}]
    status_by_path = {r["path"]: r["commit"] for r in status_list}

    # Only include submodules declared in root .gitmodules
    submodules = []
    for path in sorted(gm_map.keys()):
        meta = gm_map[path]
        commit = status_by_path.get(path)  # None if uninitialized
        submodules.append(
            {
                "path": path,
                "url": meta.get("url"),
                "branch": meta.get("branch"),
                "commit": commit,
            }
        )

    # Build manifest dict
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "TheRock": {
            "remote": rock_remote,
            "commit": rock_commit,
            "tree_state": rock_state,
            "describe": rock_desc,
            "gitmodules_sha256": gitmodules_sha,
        },
        "Environment": {
            "generated_at_utc": now_utc,
            "ci_provider": ci_provider,
            "run_id": run_id,
            "runner_os": runner_os,
            "amdgpu_families": amdgpu_families,
            "rocm_version": rocm_version,
        },
        "submodules": submodules,
    }

    # Decide output path
    out_path = args.output
    if not out_path:
        safe_ts = now_utc.replace(":", "-")
        out_path = f"{SCHEMA_VERSION}-{safe_ts}-{rock_short}.json"

    # Write JSON
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
