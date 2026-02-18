#!/usr/bin/env python3
"""
Upload the generated PyTorch manifest JSON to S3.

Upload layout:
  s3://{bucket}/{external_repo}{run_id}-{platform}/manifests/{amdgpu_family}/{manifest_name}
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
import platform
import shlex
import subprocess
import sys


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from github_actions.github_actions_utils import retrieve_bucket_info


PLATFORM = platform.system().lower()


def log(*args):
    print(*args)
    sys.stdout.flush()


def run_command(cmd: list[str], cwd: Path) -> None:
    log(f"++ Exec [{cwd}]$ {shlex.join(cmd)}")
    subprocess.run(cmd, check=True)


@dataclass(frozen=True)
class UploadPath:
    """Tracks upload paths and provides S3 URI computation."""

    bucket: str
    prefix: str  # e.g. "{external_repo}{run_id}-{platform}/manifests/gfx110X-all"

    @property
    def s3_uri(self) -> str:
        return f"s3://{self.bucket}/{self.prefix}"


def normalize_python_version_for_filename(python_version: str) -> str:
    """Normalize python version strings for filenames.

    Examples:
      "py3.11" -> "3.11"
      "3.11"   -> "3.11"
    """
    py = python_version.strip()
    if py.startswith("py"):
        py = py[2:]
    return py


def sanitize_ref_for_filename(pytorch_git_ref: str) -> str:
    """Sanitize a git ref for filenames by replacing '/' with '-'.

    Examples:
      "nightly"                -> "nightly"
      "release/2.7"            -> "release-2.7"
      "users/alice/experiment" -> "users-alice-experiment"
    """
    return pytorch_git_ref.replace("/", "-")


def build_upload_path_for_workflow_run(
    *,
    run_id: str,
    amdgpu_family: str,
    bucket_override: str | None,
) -> UploadPath:
    if bucket_override:
        external_repo = ""
        bucket = bucket_override
    else:
        # Prefer explicit run_id so retrieve_bucket_info can query the workflow run if needed.
        external_repo, bucket = retrieve_bucket_info(workflow_run_id=run_id)

    prefix = f"{external_repo}{run_id}-{PLATFORM}/manifests/{amdgpu_family}"
    return UploadPath(bucket=bucket, prefix=prefix)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload a PyTorch manifest JSON to S3."
    )
    parser.add_argument(
        "--dist-dir",
        type=Path,
        required=True,
        help="Wheel dist dir (contains manifests/).",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="Workflow run ID (e.g. 21440027240).",
    )
    parser.add_argument(
        "--amdgpu-family",
        type=str,
        required=True,
        help="AMDGPU family (e.g. gfx94X-dcgpu).",
    )
    parser.add_argument(
        "--python-version",
        type=str,
        required=True,
        help="Python version (e.g. 3.11 or py3.11).",
    )
    parser.add_argument(
        "--pytorch-git-ref",
        type=str,
        required=True,
        help="PyTorch ref (e.g. nightly, release/2.8, users/name/branch).",
    )
    parser.add_argument(
        "--bucket",
        type=str,
        default=None,
        help="Override S3 bucket (default: auto-select via retrieve_bucket_info).",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> None:
    args = parse_args(argv)

    py = normalize_python_version_for_filename(args.python_version)
    track = sanitize_ref_for_filename(args.pytorch_git_ref)

    manifest_name = f"therock-manifest_torch_py{py}_{track}.json"
    manifest_path = (args.dist_dir / "manifests" / manifest_name).resolve()

    log(f"Manifest expected at: {manifest_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    upload_path = build_upload_path_for_workflow_run(
        run_id=args.run_id,
        amdgpu_family=args.amdgpu_family,
        bucket_override=args.bucket,
    )
    dest_uri = f"{upload_path.s3_uri}/{manifest_name}"

    run_command(["aws", "s3", "cp", str(manifest_path), dest_uri], cwd=Path.cwd())


if __name__ == "__main__":
    main(sys.argv[1:])
