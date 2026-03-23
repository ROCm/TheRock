#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Bump iree-libs submodules (IREE and fusilli) in TheRock.

Reads fusilli's version.json to determine the IREE version tag, then updates:
  - iree-libs/iree  -> commit of the IREE tag from fusilli's version.json
  - iree-libs/fusilli -> HEAD of fusilli main

Designed to run inside a GitHub Actions workflow but also works locally.

Environment variable outputs (via GITHUB_ENV when running in CI):
    CURRENT_IREE_SHA      - current IREE submodule commit in TheRock
    LATEST_IREE_SHA       - target IREE commit (from fusilli's iree-version tag)
    LATEST_IREE_VERSION   - the iree-version string from fusilli's version.json (used in PR title)
    CURRENT_FUSILLI_SHA   - current fusilli submodule commit in TheRock
    LATEST_FUSILLI_SHA    - HEAD of fusilli main
"""

import json
import os
import shlex
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

THEROCK_DIR = Path(__file__).resolve().parent.parent.parent
FUSILLI_VERSION_JSON_URL = (
    "https://raw.githubusercontent.com/iree-org/fusilli/main/version.json"
)


def log(msg: str) -> None:
    print(msg)
    sys.stdout.flush()


def run_command(args: list[str], cwd: Path) -> str:
    """Run a command and return stripped stdout."""
    log(f"++ Exec [{cwd}]$ {shlex.join(args)}")
    result = subprocess.run(args, capture_output=True, text=True, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {shlex.join(args)}\n"
            f"stderr: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def set_github_env(key: str, value: str) -> None:
    """Set an environment variable in GITHUB_ENV if running in CI."""
    github_env = os.getenv("GITHUB_ENV")
    if github_env:
        with open(github_env, "a") as f:
            f.write(f"{key}={value}\n")
    # Also set in the current process so the workflow can use it immediately.
    os.environ[key] = value
    log(f"  {key}={value}")


def fetch_fusilli_version_json() -> dict[str, str]:
    """Fetch version.json from fusilli main branch."""
    log("Fetching fusilli version.json...")
    try:
        with urllib.request.urlopen(FUSILLI_VERSION_JSON_URL, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Failed to fetch fusilli version.json from {FUSILLI_VERSION_JSON_URL}"
        ) from e
    except json.JSONDecodeError as e:
        raise RuntimeError("fusilli version.json is not valid JSON") from e

    if "iree-version" not in data:
        raise ValueError("Missing 'iree-version' key in fusilli version.json")

    return data


def init_submodule(path: str) -> None:
    """Initialize a submodule with minimal depth."""
    run_command(
        ["git", "submodule", "update", "--init", "--depth", "1", "--", path],
        cwd=THEROCK_DIR,
    )


def resolve_iree_tag(iree_version: str) -> str:
    """Fetch an IREE version tag and resolve it to a commit SHA.

    Uses ``git rev-parse <tag>^{commit}`` to dereference the tag to the
    underlying commit. This "peel" syntax works for both lightweight tags
    (already a commit, no-op) and annotated tags (unwraps the tag object).
    """
    tag = f"iree-{iree_version}"
    iree_dir = THEROCK_DIR / "iree-libs" / "iree"
    log(f"Fetching IREE tag {tag}...")
    run_command(["git", "fetch", "--depth=1", "origin", "tag", tag], cwd=iree_dir)
    sha = run_command(["git", "rev-parse", f"{tag}^{{commit}}"], cwd=iree_dir)
    log(f"  IREE tag {tag}: {sha[:12]}")  # log short sha
    return sha


def get_fusilli_head() -> str:
    """Fetch and return the HEAD commit SHA of fusilli main branch."""
    fusilli_dir = THEROCK_DIR / "iree-libs" / "fusilli"
    log("Fetching fusilli main HEAD...")
    run_command(["git", "fetch", "--depth=1", "origin", "main"], cwd=fusilli_dir)
    sha = run_command(["git", "rev-parse", "origin/main"], cwd=fusilli_dir)
    log(f"  fusilli main HEAD: {sha[:12]}")  # log short sha
    return sha


def get_current_submodule_sha(path: str) -> str:
    """Get the current submodule commit SHA from the TheRock tree."""
    out = run_command(["git", "ls-tree", "HEAD", path], cwd=THEROCK_DIR)
    # Output format: "<mode> <type> <sha>\t<path>" (spaces then tab before path)
    sha = out.split()[2]
    return sha


def update_submodule(path: str, target_sha: str) -> None:
    """Checkout target_sha in an already-initialized submodule and stage it."""
    submodule_dir = THEROCK_DIR / path
    run_command(["git", "checkout", target_sha], cwd=submodule_dir)
    run_command(["git", "add", path], cwd=THEROCK_DIR)


def main() -> int:
    log("=" * 72)
    log("TheRock IREE+Fusilli Submodule Bump")
    log("=" * 72)

    # 1. Read current submodule SHAs (before init, from the tree).
    log("\nReading current submodule state...")
    current_iree_sha = get_current_submodule_sha("iree-libs/iree")
    current_fusilli_sha = get_current_submodule_sha("iree-libs/fusilli")
    log(f"  iree-libs/iree:    {current_iree_sha[:12]}")
    log(f"  iree-libs/fusilli: {current_fusilli_sha[:12]}")

    # 2. Initialize submodules so we can fetch tags and branches.
    log(f"\n{'─' * 72}")
    log("Initializing submodules...")
    init_submodule("iree-libs/iree")
    init_submodule("iree-libs/fusilli")

    # 3. Fetch fusilli's version.json to get the target IREE version.
    log(f"\n{'─' * 72}")
    version_data = fetch_fusilli_version_json()
    iree_version = version_data["iree-version"]
    log(f"Fusilli declares iree-version: {iree_version}")

    # 4. Resolve IREE tag to commit SHA.
    log(f"\n{'─' * 72}")
    latest_iree_sha = resolve_iree_tag(iree_version)

    # 5. Get fusilli HEAD.
    log(f"\n{'─' * 72}")
    latest_fusilli_sha = get_fusilli_head()

    # 6. Export to GITHUB_ENV.
    log(f"\n{'─' * 72}")
    log("Setting environment variables:")
    set_github_env("CURRENT_IREE_SHA", current_iree_sha)
    set_github_env("LATEST_IREE_SHA", latest_iree_sha)
    set_github_env("LATEST_IREE_VERSION", iree_version)
    set_github_env("CURRENT_FUSILLI_SHA", current_fusilli_sha)
    set_github_env("LATEST_FUSILLI_SHA", latest_fusilli_sha)

    # 7. Check if anything changed.
    iree_changed = current_iree_sha != latest_iree_sha
    fusilli_changed = current_fusilli_sha != latest_fusilli_sha

    if not iree_changed and not fusilli_changed:
        log(f"\n{'─' * 72}")
        log("Already up-to-date, no changes needed.")
        set_github_env("VERSIONS_CHANGED", "false")
        return 0

    set_github_env("VERSIONS_CHANGED", "true")

    # 8. Update submodules.
    log(f"\n{'─' * 72}")
    log("Updating submodules:")
    if iree_changed:
        log(
            f"  IREE: {current_iree_sha[:12]} -> {latest_iree_sha[:12]} (tag iree-{iree_version})"
        )
        update_submodule("iree-libs/iree", latest_iree_sha)
    if fusilli_changed:
        log(f"  Fusilli: {current_fusilli_sha[:12]} -> {latest_fusilli_sha[:12]}")
        update_submodule("iree-libs/fusilli", latest_fusilli_sha)

    log(f"\n{'─' * 72}")
    log("Submodules updated and staged.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
