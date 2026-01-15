#!/usr/bin/env python3
"""
Upload the generated PyTorch manifest JSON to S3.

Intended for use from GitHub Actions workflows.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print("++", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def main() -> None:
    ap = argparse.ArgumentParser(description="Upload a PyTorch manifest JSON to S3.")
    ap.add_argument(
        "--dist-dir",
        type=Path,
        required=True,
        help="Wheel dist dir (contains manifests/).",
    )
    ap.add_argument("--bucket", required=True, help="S3 bucket name (no s3:// prefix).")
    ap.add_argument("--run-id", required=True, help="GitHub run id (used in prefix).")
    ap.add_argument(
        "--platform",
        required=True,
        choices=["linux", "windows"],
        help="Platform name for prefix.",
    )
    ap.add_argument(
        "--amdgpu-family", required=True, help="AMDGPU family (used in prefix)."
    )
    ap.add_argument(
        "--python-version", required=True, help="Python version (e.g. 3.11 or py3.11)."
    )
    ap.add_argument(
        "--pytorch-git-ref",
        required=True,
        help="PyTorch ref (e.g. nightly or release/2.8).",
    )
    args = ap.parse_args()

    py = args.python_version
    if py.startswith("py"):
        py = py[2:]

    if args.pytorch_git_ref == "nightly":
        track = "nightly"
    elif args.pytorch_git_ref.startswith("release/"):
        track = args.pytorch_git_ref.replace("/", "-", 1)
    else:
        track = args.pytorch_git_ref.replace("/", "-")

    manifest_name = f"therock-manifest_torch_py{py}_{track}.json"
    manifest_path = (args.dist_dir / "manifests" / manifest_name).resolve()

    print(f"Manifest expected at: {manifest_path}", flush=True)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    dest_uri = (
        f"s3://{args.bucket}/"
        f"{args.run_id}-{args.platform}/manifests/{args.amdgpu_family}/{manifest_name}"
    )
    _run(["aws", "s3", "cp", str(manifest_path), dest_uri])


if __name__ == "__main__":
    main()
