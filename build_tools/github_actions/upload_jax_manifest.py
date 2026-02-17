#!/usr/bin/env python3
"""
Upload the generated JAX manifest JSON to S3.

Upload layout (same style as torch manifests):
  s3://{bucket}/{external_repo}{run_id}-{platform}/manifests/{amdgpu_family}/{manifest_name}

The bucket and external_repo prefix are resolved via retrieve_bucket_info() unless
--bucket is provided.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
import platform
import shlex
import subprocess
import sys

# Import retrieve_bucket_info from build_tools/github_actions
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from github_actions.github_actions_utils import retrieve_bucket_info


PLATFORM = platform.system().lower()


def log(*args: object) -> None:
    print(*args)
    sys.stdout.flush()


def run_command(cmd: list[str], cwd: Path) -> None:
    log(f"++ Exec [{cwd}]$ {shlex.join(cmd)}")
    subprocess.run(cmd, check=True)


@dataclass(frozen=True)
class UploadPath:
    """Tracks upload paths and provides S3 URI computation."""

    bucket: str
    prefix: str  # e.g. "{external_repo}{run_id}-{platform}/manifests/gfx94X-dcgpu"

    @property
    def s3_uri(self) -> str:
        return f"s3://{self.bucket}/{self.prefix}"


def normalize_py(python_version: str) -> str:
    """Normalize python version strings for filenames.

    Examples:
      "py3.12" -> "3.12"
      "3.12"   -> "3.12"
    """
    py = python_version.strip()
    if py.startswith("py"):
        py = py[2:]
    return py


def sanitize_ref_for_filename(jax_track: str) -> str:
    """Sanitize a git ref for filenames by replacing '/' with '-'.

    Examples:
      "nightly"                -> "nightly"
      "release/0.4.28"         -> "release-0.4.28"
      "users/alice/experiment" -> "users-alice-experiment"
    """
    return jax_track.replace("/", "-")


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
        # retrieve_bucket_info() returns (external_repo_prefix, bucket_name)
        # It uses the run id to discover where artifacts should go.
        external_repo, bucket = retrieve_bucket_info(workflow_run_id=run_id)

    prefix = f"{external_repo}{run_id}-{PLATFORM}/manifests/{amdgpu_family}"
    return UploadPath(bucket=bucket, prefix=prefix)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload a JAX manifest JSON to S3.")
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
        help="Python version (e.g. 3.12 or py3.12).",
    )
    parser.add_argument(
        "--jax-track",
        type=str,
        required=True,
        help="JAX track used in manifest naming (e.g. nightly, release/0.4.28, rocm-jaxlib-v0.8.0-fixdevtar).",
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

    py = normalize_py(args.python_version)
    track = sanitize_ref_for_filename(args.jax_track)

    manifest_name = f"therock-manifest_jax_py{py}_{track}.json"
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

    log(f"Uploading to: {dest_uri}")
    run_command(["aws", "s3", "cp", str(manifest_path), dest_uri], cwd=Path.cwd())


if __name__ == "__main__":
    main(sys.argv[1:])
