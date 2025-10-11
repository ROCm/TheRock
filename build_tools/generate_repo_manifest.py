#!/usr/bin/env python3
"""
Generate TheRock build manifest (JSON Format).

Schema: therock-manifest.v1
- rock: remote, commit, tree_state, describe, gitmodules_sha256, gitmodules
- environment: generated_at_utc, ci_provider, run_id, runner_os, amdgpu_families, rocm_version
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


def parse_gitmodules():
    """
    Parse .gitmodules into a map: path -> {url, branch or None}.
    Keeps only fields needed for the manifest.
    """
    out = {}
    cur = None
    with open(".gitmodules", "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
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
    Returns list of {path, commit or None}.
    Uses `git submodule status --recursive`.
    """
    raw = _run(["git", "submodule", "status", "--recursive"])
    items = []
    for line in raw.splitlines():
        if not line:
            continue
        ch = line[0]  # ' ', '-', '+', 'U'
        rest = line[1:].strip()
        parts = rest.split()
        if len(parts) < 2:
            # defensive; skip malformed lines
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
        help="Output file path. If omitted, uses therock-manifest-<shortsha>-<UTC>.json",
    )
    args = ap.parse_args()

    # Ensure we run at repo root for consistent paths
    root = git_root()
    os.chdir(root)

    # --- rock ---
    rock_commit = _run(["git", "rev-parse", "HEAD"])
    rock_short = _run(["git", "rev-parse", "--short", "HEAD"])
    rock_remote = git_remote_origin()
    rock_state = git_tree_state()
    rock_desc = _try(["git", "describe", "--always", "--tags", "--dirty"])

    # .gitmodules (literal + sha256)
    gm_path = ".gitmodules"
    with open(gm_path, "r", encoding="utf-8") as f:
        gitmodules_text = f.read()
    gitmodules_sha = file_sha256(gm_path)

    # --- environment ---
    now_utc = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    ci_provider = "github-actions" if os.getenv("GITHUB_ACTIONS") else None
    run_id = os.getenv("RUN_ID")
    runner_os = os.getenv("RUNNER_OS")
    amdgpu_families = os.getenv("AMDGPU_FAMILIES") or os.getenv(
        "THEROCK_AMDGPU_FAMILIES"
    )
    rocm_version = os.getenv("ROCM_VERSION") or os.getenv("THEROCK_ROCM_VERSION")

    # --- submodules ---
    gm_map = parse_gitmodules()  # path -> {url, branch}
    status_list = submodule_list_status()  # [{path, commit}]
    # Merge, keep only required fields in required order
    submodules = []
    for rec in sorted(status_list, key=lambda r: r["path"]):
        path = rec["path"]
        commit = rec["commit"]
        meta = gm_map.get(path, {"url": None, "branch": None})
        submodules.append(
            {
                "path": path,
                "url": meta.get("url"),
                "branch": meta.get("branch"),
                "commit": commit,
            }
        )

    # Some submodules might exist in .gitmodules but not in status (e.g., not initialized)
    # Include them too with commit=null
    for path in sorted(set(gm_map.keys()) - {r["path"] for r in status_list}):
        meta = gm_map[path]
        submodules.append(
            {
                "path": path,
                "url": meta.get("url"),
                "branch": meta.get("branch"),
                "commit": None,
            }
        )
    # Re-sort after union
    submodules.sort(key=lambda r: r["path"])

    # Build manifest dict with keys in desired order
    manifest = {
        "schema_version": "therock-manifest.v1",
        "rock": {
            "remote": rock_remote,
            "commit": rock_commit,
            "tree_state": rock_state,
            "describe": rock_desc,
            "gitmodules_sha256": gitmodules_sha,
            "gitmodules": gitmodules_text,
        },
        "environment": {
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
        out_path = f"therock-manifest-{rock_short}-{safe_ts}.json"

    # Write JSON with stable indentation; ensure_ascii=False to preserve literal text
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(out_path)


if __name__ == "__main__":
    sys.exit(main())
