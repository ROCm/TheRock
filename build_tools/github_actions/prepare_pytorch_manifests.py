#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Prepare one PyTorch source manifest for a build workflow.

The build workflow always consumes a manifest URL: an HTTP URL to a JSON source
manifest that pins the exact repository URLs, commits, branches, and expected
package versions for one PyTorch build. Passing this URL between jobs lets the
workflow freeze source discovery once, then have the build job check out exactly
the same sources even when floating refs such as ``nightly`` move later.

This script supports two input modes:

* Pass-through mode: ``--manifest-url`` is already known. The script emits that
  URL as a GitHub Actions output. It does not write a step summary because the
  caller that generated or selected the manifests owns that summary.
* Generation mode: ``--pytorch-git-ref``, ``--rocm-version``, and ``--run-id``
  identify the PyTorch ref, package version context, and artifact layout. The
  script invokes ``generate_pytorch_source_manifest.py`` to write one manifest
  under ``--manifest-dir``, uploads it to the workflow output location, emits
  the manifest URL output, and writes a step summary.

Optional inputs are either forwarded to the manifest generator (platform,
project list, version suffix, and TheRock source info overrides) or used by this
script for upload behavior (artifact bucket, local upload staging directory, and
dry-run behavior).
"""

import argparse
import subprocess
import sys
from pathlib import Path

_BUILD_TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BUILD_TOOLS_DIR))

from _therock_utils.storage_backend import create_storage_backend
from _therock_utils.storage_location import StorageLocation
from _therock_utils.workflow_outputs import WorkflowOutputRoot
from github_actions.generate_pytorch_source_manifest import manifest_filename
from github_actions.github_actions_api import gha_append_step_summary, gha_set_output


GENERATOR_SCRIPT = (
    Path(__file__).resolve().with_name("generate_pytorch_source_manifest.py")
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
    projects: str,
    therock_commit: str,
    therock_repo: str,
    therock_branch: str,
) -> Path:
    manifest_path = manifest_dir / manifest_filename(
        platform=platform, pytorch_git_ref=pytorch_git_ref
    )
    command = [
        sys.executable,
        str(GENERATOR_SCRIPT),
        "--rocm-version",
        rocm_version,
        "--platform",
        platform,
        "--pytorch-git-refs",
        pytorch_git_ref,
        "--output",
        str(manifest_path),
    ]
    if version_suffix:
        command.extend(["--version-suffix", version_suffix])
    if projects:
        command.extend(["--projects", projects])
    if therock_commit:
        command.extend(["--therock-commit", therock_commit])
    if therock_repo:
        command.extend(["--therock-repo", therock_repo])
    if therock_branch:
        command.extend(["--therock-branch", therock_branch])
    subprocess.check_call(command)
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
) -> str:
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
    return manifest_location.https_url


def append_manifest_summary(*, manifest_url: str, rocm_version: str = "") -> None:
    lines = [
        "## PyTorch Manifest",
        "",
        f"* Manifest: {manifest_url}",
    ]
    if rocm_version:
        lines.append(f"* ROCm version: `{rocm_version}`")
    gha_append_step_summary("\n".join(lines) + "\n")


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
    args = parser.parse_args(argv)

    # Without a manifest URL, this script generates and uploads one.
    if not args.manifest_url:
        if not args.pytorch_git_ref:
            parser.error("--pytorch-git-ref is required without --manifest-url")
        if not args.rocm_version:
            parser.error("--rocm-version is required when generating a manifest")
        if not args.run_id:
            parser.error(
                "--run-id is required when generating and uploading a manifest"
            )

    return args


def main(argv: list[str]) -> None:
    args = parse_args(argv)
    if args.manifest_url:
        # Pass-through mode: the caller already selected or generated the
        # manifest, so this job only forwards the URL to downstream jobs.
        gha_set_output({"manifest_url": args.manifest_url})
        return

    # Generation mode: invoke the manifest generator, upload its output, then
    # pass the uploaded URL to downstream jobs.
    manifest_path = generate_manifest_file(
        manifest_dir=args.manifest_dir,
        pytorch_git_ref=args.pytorch_git_ref,
        rocm_version=args.rocm_version,
        version_suffix=args.version_suffix,
        platform=args.platform,
        projects=args.projects,
        therock_commit=args.therock_commit,
        therock_repo=args.therock_repo,
        therock_branch=args.therock_branch,
    )
    manifest_url = upload_manifest_file(
        manifest_path=manifest_path,
        run_id=args.run_id,
        platform=args.platform,
        release_type=args.release_type,
        bucket=args.bucket,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
    )
    append_manifest_summary(
        manifest_url=manifest_url,
        rocm_version=args.rocm_version,
    )
    gha_set_output({"manifest_url": manifest_url})


if __name__ == "__main__":
    main(sys.argv[1:])
