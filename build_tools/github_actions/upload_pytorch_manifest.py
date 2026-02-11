#!/usr/bin/env python3
"""
Upload the generated PyTorch manifest JSON to S3.
"""

import argparse
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
from github_actions.github_actions_utils import retrieve_bucket_info


def log(*args):
    print(*args)
    sys.stdout.flush()


def run_command(cmd: list[str], cwd: Path):
    log(f"++ Exec [{cwd}]$ {shlex.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Upload a PyTorch manifest JSON to S3.")
    ap.add_argument(
        "--dist-dir",
        type=Path,
        required=True,
        help="Wheel dist dir (contains manifests/).",
    )
    ap.add_argument("--run-id", required=True, help="GitHub run id (used in prefix).")
    ap.add_argument(
        "--amdgpu-family",
        required=True,
        help="AMDGPU family (used in prefix).",
    )
    ap.add_argument(
        "--python-version",
        required=True,
        help="Python version (e.g. 3.11 or py3.11).",
    )
    ap.add_argument(
        "--pytorch-git-ref",
        required=True,
        help="PyTorch ref (e.g. nightly or release/2.8).",
    )
    args = ap.parse_args()

    platform_name = platform.system().lower()
    if platform_name not in {"linux", "windows"}:
        raise RuntimeError(f"Unsupported platform: {platform_name}")

    # Normalize Python version for filenames:
    #   "py3.11" -> "3.11"
    #   "3.11"   -> "3.11"
    py = args.python_version.strip()
    if py.startswith("py"):
        py = py[2:]

    # Normalize git ref for filenames by replacing path separators.
    # Examples:
    #   "nightly"               -> "nightly"
    #   "release/2.7"           -> "release-2.7"
    #   "users/alice/experiment"-> "users-alice-experiment"
    track = args.pytorch_git_ref.replace("/", "-")

    manifest_name = f"therock-manifest_torch_py{py}_{track}.json"
    manifest_path = (args.dist_dir / "manifests" / manifest_name).resolve()

    log(f"Manifest expected at: {manifest_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    external_repo_path, bucket = retrieve_bucket_info(workflow_run_id=args.run_id)

    bucket_uri = f"s3://{bucket}/{external_repo_path}{args.run_id}-{platform_name}"
    dest_uri = f"{bucket_uri}/manifests/{args.amdgpu_family}/{manifest_name}"

    run_command(["aws", "s3", "cp", str(manifest_path), dest_uri], cwd=Path.cwd())


if __name__ == "__main__":
    main()
