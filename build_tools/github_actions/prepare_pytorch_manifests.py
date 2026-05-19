#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Prepare PyTorch manifests for build and test workflows.

This is the user-facing entry point for manifest preparation. It can:

* pass through an already-uploaded manifest URL;
* generate and upload one manifest for a single build cell; or
* generate/upload all release manifests and emit a build matrix with explicit
  manifest URLs.
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

_BUILD_TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BUILD_TOOLS_DIR))

from _therock_utils.storage_backend import create_storage_backend
from _therock_utils.workflow_outputs import WorkflowOutputRoot
from github_actions.determine_version import derive_version_suffix
from github_actions.generate_pytorch_manifest_upfront import (
    DEFAULT_PYTORCH_GIT_REFS,
    default_projects_for_pytorch_ref,
    generate_manifest,
)
from github_actions.github_actions_api import gha_append_step_summary, gha_set_output
from github_actions.manifest_utils import (
    GitSourceInfo,
    detect_therock_source_info,
    normalize_ref_for_filename,
)


DEFAULT_RELEASE_PYTHON_VERSIONS = ["3.10", "3.11", "3.12", "3.13", "3.14"]
DEFAULT_RELEASE_EXCLUDES = [("release/2.8", "3.14")]


@dataclass(frozen=True)
class UploadResult:
    count: int
    manifest_dir_url: str
    manifest_dir_s3_uri: str


def split_words(value: str) -> list[str]:
    return value.split() if value else []


def parse_excludes(values: list[str]) -> set[tuple[str, str]]:
    """Parse exclusions in '<pytorch_git_ref>|<python_version>' form."""
    excludes: set[tuple[str, str]] = set()
    for value in values:
        try:
            pytorch_ref, python_version = value.split("|", maxsplit=1)
        except ValueError as e:
            raise ValueError(
                f"Invalid exclusion {value!r}, expected '<pytorch_ref>|<python_version>'"
            ) from e
        excludes.add((pytorch_ref, python_version))
    return excludes


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


def manifest_filename(*, platform: str, pytorch_git_ref: str) -> str:
    ref = normalize_ref_for_filename(pytorch_git_ref)
    return f"therock-manifest_torch_{platform}_{ref}.json"


def write_manifest_file(path: Path, manifest: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Failed to write manifest: {path}")


def generate_manifest_files(
    *,
    manifest_dir: Path,
    pytorch_git_refs: list[str],
    rocm_version: str,
    version_suffix: str,
    platform: str,
    projects: list[str] | None,
    therock_info: GitSourceInfo,
) -> dict[str, Path]:
    """Generate manifest files and return pytorch_git_ref -> path."""
    outputs: dict[str, Path] = {}
    for pytorch_git_ref in pytorch_git_refs:
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
        out_path = manifest_dir / manifest_filename(
            platform=platform, pytorch_git_ref=pytorch_git_ref
        )
        write_manifest_file(out_path, manifest)
        outputs[pytorch_git_ref] = out_path
    return outputs


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


def upload_manifest_directory(
    *,
    manifest_dir: Path,
    run_id: str,
    platform: str,
    release_type: str,
    bucket: str | None = None,
    output_dir: Path | None = None,
    dry_run: bool = False,
    amdgpu_family: str = "",
) -> UploadResult:
    """Upload manifests and return the destination directory metadata."""
    if not manifest_dir.is_dir():
        raise FileNotFoundError(f"Manifest directory not found: {manifest_dir}")

    output_root = make_output_root(
        run_id=run_id,
        platform=platform,
        release_type=release_type,
        bucket_override=bucket,
    )
    dest = output_root.pytorch_manifest_dir(amdgpu_family)
    backend = create_storage_backend(staging_dir=output_dir, dry_run=dry_run)
    count = backend.upload_directory(manifest_dir, dest, include=["*.json"])
    if count == 0:
        raise FileNotFoundError(f"No JSON files found in {manifest_dir}")

    gha_append_step_summary(f"PyTorch manifests: {dest.https_url}/index.html\n")
    return UploadResult(
        count=count,
        manifest_dir_url=dest.https_url,
        manifest_dir_s3_uri=dest.s3_uri,
    )


def read_manifest_pytorch_ref(manifest_path: Path) -> str:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pytorch = manifest.get("pytorch")
    if not isinstance(pytorch, dict):
        raise ValueError(f"{manifest_path}: missing pytorch manifest entry")
    pytorch_ref = pytorch.get("branch")
    if not isinstance(pytorch_ref, str) or not pytorch_ref:
        raise ValueError(f"{manifest_path}: missing pytorch branch")
    return pytorch_ref


def collect_manifest_urls(
    *, manifest_dir: Path, manifest_dir_url: str
) -> dict[str, str]:
    """Return pytorch_git_ref -> explicit uploaded manifest URL."""
    if not manifest_dir.is_dir():
        raise FileNotFoundError(f"Manifest directory not found: {manifest_dir}")

    base_url = manifest_dir_url.rstrip("/")
    manifest_urls: dict[str, str] = {}
    for manifest_path in sorted(manifest_dir.glob("*.json")):
        pytorch_ref = read_manifest_pytorch_ref(manifest_path)
        if pytorch_ref in manifest_urls:
            raise ValueError(f"Duplicate manifest for PyTorch ref {pytorch_ref!r}")
        manifest_urls[pytorch_ref] = f"{base_url}/{manifest_path.name}"

    if not manifest_urls:
        raise FileNotFoundError(f"No JSON manifests found in {manifest_dir}")
    return manifest_urls


def build_matrix(
    *,
    manifest_urls: dict[str, str],
    python_versions: list[str],
    pytorch_git_refs: list[str],
    excludes: set[tuple[str, str]],
) -> dict[str, list[dict[str, str]]]:
    missing_refs = [ref for ref in pytorch_git_refs if ref not in manifest_urls]
    if missing_refs:
        raise ValueError(f"Missing manifests for PyTorch refs: {missing_refs}")

    include = []
    for pytorch_ref in pytorch_git_refs:
        for python_version in python_versions:
            if (pytorch_ref, python_version) in excludes:
                continue
            include.append(
                {
                    "python_version": python_version,
                    "pytorch_git_ref": pytorch_ref,
                    "manifest_url": manifest_urls[pytorch_ref],
                }
            )

    if not include:
        raise ValueError("Generated an empty PyTorch manifest matrix")
    return {"include": include}


def emit_single_outputs(manifest_url: str, upload: UploadResult | None = None) -> None:
    outputs: dict[str, str] = {"manifest_url": manifest_url}
    if upload:
        outputs["manifest_dir_url"] = upload.manifest_dir_url
        outputs["manifest_dir_s3_uri"] = upload.manifest_dir_s3_uri
    gha_set_output(outputs)


def emit_matrix_outputs(matrix: dict[str, object], upload: UploadResult) -> None:
    gha_set_output(
        {
            "matrix": json.dumps(matrix),
            "manifest_dir_url": upload.manifest_dir_url,
            "manifest_dir_s3_uri": upload.manifest_dir_s3_uri,
        }
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-mode",
        choices=["single", "matrix"],
        default="single",
        help="Emit either one manifest_url or a build matrix.",
    )
    parser.add_argument(
        "--matrix-preset",
        choices=["none", "linux-release"],
        default="none",
        help="Populate release defaults for refs, Python versions, and excludes.",
    )
    parser.add_argument(
        "--manifest-url",
        default="",
        help="Already-uploaded manifest URL to pass through in single mode.",
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
            "Space-separated manifest projects. Defaults to projects selected "
            "for each platform and PyTorch ref."
        ),
    )
    parser.add_argument(
        "--pytorch-git-refs",
        default="",
        help="Space-separated PyTorch refs.",
    )
    parser.add_argument(
        "--python-versions",
        default="",
        help="Space-separated Python versions for matrix mode.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Exclude one matrix cell in '<pytorch_ref>|<python_version>' form.",
    )
    parser.add_argument(
        "--no-default-excludes",
        action="store_true",
        help="Do not add matrix-preset exclusions.",
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
        if args.output_mode != "single":
            raise ValueError(
                "--manifest-url can only be used with --output-mode=single"
            )
        emit_single_outputs(args.manifest_url)
        return

    if not args.rocm_version:
        raise ValueError("--rocm-version is required when generating manifests")
    if not args.run_id:
        raise ValueError("--run-id is required when generating and uploading manifests")

    pytorch_refs = split_words(args.pytorch_git_refs)
    python_versions = split_words(args.python_versions)
    excludes = parse_excludes(args.exclude)
    if args.matrix_preset == "linux-release":
        if not pytorch_refs:
            pytorch_refs = DEFAULT_PYTORCH_GIT_REFS
        if not python_versions:
            python_versions = DEFAULT_RELEASE_PYTHON_VERSIONS
        if not args.no_default_excludes:
            excludes.update(DEFAULT_RELEASE_EXCLUDES)
    if not pytorch_refs:
        raise ValueError("--pytorch-git-refs is required without a matrix preset")
    if args.output_mode == "single" and len(pytorch_refs) != 1:
        raise ValueError("--output-mode=single requires exactly one PyTorch ref")
    if args.output_mode == "matrix" and not python_versions:
        raise ValueError("--python-versions is required for matrix mode")

    version_suffix = args.version_suffix or derive_version_suffix(args.rocm_version)
    projects = split_words(args.projects) or None
    therock_info = resolve_therock_source_info(
        therock_root=Path(__file__).resolve().parents[2],
        therock_commit=args.therock_commit,
        therock_repo=args.therock_repo,
        therock_branch=args.therock_branch,
    )
    generated = generate_manifest_files(
        manifest_dir=args.manifest_dir,
        pytorch_git_refs=pytorch_refs,
        rocm_version=args.rocm_version,
        version_suffix=version_suffix,
        platform=args.platform,
        projects=projects,
        therock_info=therock_info,
    )
    upload = upload_manifest_directory(
        manifest_dir=args.manifest_dir,
        run_id=args.run_id,
        platform=args.platform,
        release_type=args.release_type,
        bucket=args.bucket,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
    )
    manifest_urls = collect_manifest_urls(
        manifest_dir=args.manifest_dir,
        manifest_dir_url=upload.manifest_dir_url,
    )

    if args.output_mode == "single":
        pytorch_ref = next(iter(generated))
        emit_single_outputs(manifest_urls[pytorch_ref], upload)
        return

    matrix = build_matrix(
        manifest_urls=manifest_urls,
        python_versions=python_versions,
        pytorch_git_refs=pytorch_refs,
        excludes=excludes,
    )
    emit_matrix_outputs(matrix, upload)


if __name__ == "__main__":
    main(sys.argv[1:])
