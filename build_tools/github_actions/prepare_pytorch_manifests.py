#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Prepare one PyTorch source manifest for a build workflow.

The build workflow always consumes a manifest URL. This script either passes
through an existing URL or generates, uploads, and outputs one manifest for a
single PyTorch ref.
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

_BUILD_TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BUILD_TOOLS_DIR))

from _therock_utils.storage_backend import create_storage_backend
from _therock_utils.storage_location import StorageLocation
from _therock_utils.workflow_outputs import WorkflowOutputRoot
from github_actions.determine_version import derive_version_suffix
from github_actions.generate_pytorch_source_manifest import (
    default_projects_for_pytorch_ref,
    generate_manifest,
    manifest_filename,
    write_manifest_file,
)
from github_actions.github_actions_api import gha_append_step_summary, gha_set_output
from github_actions.manifest_utils import GitSourceInfo, detect_therock_source_info


@dataclass(frozen=True)
class UploadResult:
    manifest_url: str
    manifest_s3_uri: str
    manifest_dir_url: str
    manifest_dir_s3_uri: str


def _split_words(value: str) -> list[str]:
    return value.replace(";", " ").split() if value else []


def resolve_therock_source_info(
    *,
    therock_root: Path,
    therock_commit: str,
    therock_repo: str,
    therock_branch: str,
) -> GitSourceInfo:
    detected = detect_therock_source_info(therock_root)
    return GitSourceInfo(
        commit=therock_commit or detected.commit,
        repo=therock_repo or detected.repo,
        branch=therock_branch or detected.branch,
    )


def make_output_root(
    *,
    run_id: str,
    platform: str,
    release_type: str,
    bucket_override: str | None,
) -> WorkflowOutputRoot:
    if bucket_override:
        return WorkflowOutputRoot(
            bucket=bucket_override,
            external_repo="",
            run_id=run_id,
            platform=platform,
        )
    return WorkflowOutputRoot.from_workflow_run(
        run_id=run_id,
        platform=platform,
        release_type=release_type or None,
    )


def generate_manifest_file(
    *,
    manifest_dir: Path,
    pytorch_git_ref: str,
    rocm_version: str,
    version_suffix: str,
    platform: str,
    projects: list[str] | None,
    therock_info: GitSourceInfo,
) -> Path:
    manifest_projects = projects or default_projects_for_pytorch_ref(
        platform, pytorch_git_ref
    )
    manifest = generate_manifest(
        pytorch_git_ref=pytorch_git_ref,
        rocm_version=rocm_version,
        version_suffix=version_suffix,
        platform=platform,
        projects=manifest_projects,
        therock_commit=therock_info.commit,
        therock_repo=therock_info.repo,
        therock_branch=therock_info.branch or "",
    )
    manifest_path = manifest_dir / manifest_filename(
        platform=platform, pytorch_git_ref=pytorch_git_ref
    )
    write_manifest_file(manifest_path, manifest)
    return manifest_path


def upload_manifest_file(
    *,
    manifest_path: Path,
    run_id: str,
    platform: str,
    release_type: str,
    bucket: str | None = None,
    output_dir: Path | None = None,
    dry_run: bool = False,
) -> UploadResult:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    output_root = make_output_root(
        run_id=run_id,
        platform=platform,
        release_type=release_type,
        bucket_override=bucket,
    )
    manifest_dir = output_root.pytorch_manifest_dir()
    manifest_location = StorageLocation(
        manifest_dir.bucket,
        f"{manifest_dir.relative_path}/{manifest_path.name}",
    )

    backend = create_storage_backend(staging_dir=output_dir, dry_run=dry_run)
    backend.upload_file(manifest_path, manifest_location)
    return UploadResult(
        manifest_url=manifest_location.https_url,
        manifest_s3_uri=manifest_location.s3_uri,
        manifest_dir_url=manifest_dir.https_url,
        manifest_dir_s3_uri=manifest_dir.s3_uri,
    )


def append_manifest_summary(
    *, manifest_url: str, manifest_s3_uri: str = "", rocm_version: str = ""
) -> None:
    lines = [
        "## PyTorch Manifest",
        "",
        f"* Manifest: {manifest_url}",
    ]
    if manifest_s3_uri:
        lines.append(f"* S3 URI: `{manifest_s3_uri}`")
    if rocm_version:
        lines.append(f"* ROCm version: `{rocm_version}`")
    gha_append_step_summary("\n".join(lines) + "\n")


def emit_outputs(manifest_url: str, upload: UploadResult | None = None) -> None:
    outputs = {"manifest_url": manifest_url}
    if upload:
        outputs.update(
            {
                "manifest_s3_uri": upload.manifest_s3_uri,
                "manifest_dir_url": upload.manifest_dir_url,
                "manifest_dir_s3_uri": upload.manifest_dir_s3_uri,
            }
        )
    gha_set_output(outputs)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-url",
        default="",
        help="Already-uploaded manifest URL to pass through.",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("output/manifests"),
        help="Directory to write generated manifests.",
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="GitHub Actions run ID used for uploaded manifest layout.",
    )
    parser.add_argument(
        "--release-type",
        default="",
        help='Release type ("dev", "nightly", or "prerelease") for artifact bucket selection.',
    )
    parser.add_argument("--rocm-version", default="", help="ROCm package version.")
    parser.add_argument(
        "--version-suffix",
        default="",
        help="PyTorch package version suffix. Defaults to deriving from --rocm-version.",
    )
    parser.add_argument(
        "--platform",
        choices=["linux", "windows"],
        default="linux",
        help="Target platform.",
    )
    parser.add_argument(
        "--projects",
        default="",
        help=(
            "Semicolon- or space-separated manifest projects. Defaults to "
            "projects selected for the platform and PyTorch ref."
        ),
    )
    parser.add_argument(
        "--pytorch-git-ref",
        default="",
        help="PyTorch ref for the generated manifest.",
    )
    parser.add_argument("--therock-commit", default="", help="Override TheRock commit.")
    parser.add_argument("--therock-repo", default="", help="Override TheRock repo URL.")
    parser.add_argument("--therock-branch", default="", help="Override TheRock branch.")
    parser.add_argument("--bucket", default=None, help="Override artifact bucket.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output to local directory instead of S3 for testing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print upload plan without actually uploading.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> None:
    args = parse_args(argv)
    if args.manifest_url:
        emit_outputs(args.manifest_url)
        append_manifest_summary(manifest_url=args.manifest_url)
        return

    if not args.pytorch_git_ref:
        raise ValueError("--pytorch-git-ref is required without --manifest-url")
    if not args.rocm_version:
        raise ValueError("--rocm-version is required when generating a manifest")
    if not args.run_id:
        raise ValueError(
            "--run-id is required when generating and uploading a manifest"
        )

    version_suffix = args.version_suffix or derive_version_suffix(args.rocm_version)
    projects = _split_words(args.projects) or None
    therock_info = resolve_therock_source_info(
        therock_root=Path(__file__).resolve().parents[2],
        therock_commit=args.therock_commit,
        therock_repo=args.therock_repo,
        therock_branch=args.therock_branch,
    )
    manifest_path = generate_manifest_file(
        manifest_dir=args.manifest_dir,
        pytorch_git_ref=args.pytorch_git_ref,
        rocm_version=args.rocm_version,
        version_suffix=version_suffix,
        platform=args.platform,
        projects=projects,
        therock_info=therock_info,
    )
    upload = upload_manifest_file(
        manifest_path=manifest_path,
        run_id=args.run_id,
        platform=args.platform,
        release_type=args.release_type,
        bucket=args.bucket,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
    )
    append_manifest_summary(
        manifest_url=upload.manifest_url,
        manifest_s3_uri=upload.manifest_s3_uri,
        rocm_version=args.rocm_version,
    )
    emit_outputs(upload.manifest_url, upload)


if __name__ == "__main__":
    main(sys.argv[1:])
