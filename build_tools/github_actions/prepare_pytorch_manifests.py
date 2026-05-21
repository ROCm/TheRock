#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Prepare PyTorch manifests for build and test workflows.

This is the user-facing entry point for manifest preparation. It can:

* pass through an already-uploaded manifest URL;
* generate and upload one manifest for a single build cell; or
* consume an already-uploaded manifest directory and emit a build matrix; or
* generate/upload all release manifests and emit a build matrix with explicit
  manifest URLs.
"""

import argparse
import json
import sys
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

_BUILD_TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BUILD_TOOLS_DIR))

from _therock_utils.storage_backend import create_storage_backend
from _therock_utils.storage_location import StorageLocation
from _therock_utils.workflow_outputs import WorkflowOutputRoot
from github_actions.determine_version import derive_version_suffix
from github_actions.generate_pytorch_manifest_upfront import (
    DEFAULT_PYTORCH_GIT_REFS,
    default_projects_for_pytorch_ref,
    generate_manifest,
    manifest_filename,
    write_manifest_file,
)
from github_actions.github_actions_api import gha_append_step_summary, gha_set_output
from github_actions.manifest_utils import (
    GitSourceInfo,
    detect_therock_source_info,
)


DEFAULT_RELEASE_PYTHON_VERSIONS = ["3.10", "3.11", "3.12", "3.13", "3.14"]
DEFAULT_RELEASE_EXCLUDES = [("release/2.8", "3.14")]


@dataclass(frozen=True)
class UploadResult:
    count: int
    manifest_dir_url: str
    manifest_dir_s3_uri: str


def _split_words(value: str) -> list[str]:
    return value.replace(";", " ").split() if value else []


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

    return UploadResult(
        count=count,
        manifest_dir_url=dest.https_url,
        manifest_dir_s3_uri=dest.s3_uri,
    )


def read_manifest_pytorch_ref(manifest_path: Path) -> str:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return read_manifest_data_pytorch_ref(manifest, str(manifest_path))


def read_manifest_data_pytorch_ref(
    manifest: dict[str, object], source_name: str
) -> str:
    pytorch = manifest.get("pytorch")
    if not isinstance(pytorch, dict):
        raise ValueError(f"{source_name}: missing pytorch manifest entry")
    pytorch_ref = pytorch.get("branch")
    if not isinstance(pytorch_ref, str) or not pytorch_ref:
        raise ValueError(f"{source_name}: missing pytorch branch")
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


def storage_location_from_url(url: str) -> StorageLocation:
    """Parse an S3 URI or bucket-style S3 HTTPS URL into a StorageLocation."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme == "s3":
        bucket = parsed.netloc
        relative_path = parsed.path.lstrip("/")
    elif parsed.scheme in ("http", "https"):
        suffix = ".s3.amazonaws.com"
        if not parsed.netloc.endswith(suffix):
            raise ValueError(f"Unsupported manifest directory URL: {url}")
        bucket = parsed.netloc[: -len(suffix)]
        relative_path = urllib.parse.unquote(parsed.path.lstrip("/"))
    else:
        raise ValueError(f"Unsupported manifest directory URL: {url}")

    if relative_path.endswith("/index.html"):
        relative_path = relative_path.removesuffix("/index.html")
    relative_path = relative_path.rstrip("/")
    if not bucket or not relative_path:
        raise ValueError(f"Unsupported manifest directory URL: {url}")
    return StorageLocation(bucket=bucket, relative_path=relative_path)


def read_manifest_location(
    location: StorageLocation, *, output_dir: Path | None
) -> dict[str, object]:
    """Read a manifest from either local test storage or S3."""
    if output_dir:
        path = location.local_path(output_dir)
        if not path.is_file():
            raise FileNotFoundError(f"Manifest not found: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    # boto3 is only needed for this consume mode; most local unit tests should
    # continue to run without importing it.
    import boto3

    response = boto3.client("s3").get_object(
        Bucket=location.bucket, Key=location.relative_path
    )
    body = response["Body"].read().decode("utf-8")
    return json.loads(body)


def collect_manifest_urls_from_storage(
    *, manifest_dir_url: str, output_dir: Path | None = None
) -> tuple[dict[str, str], UploadResult]:
    """Return pytorch_git_ref -> URL for manifests already in storage."""
    manifest_dir = storage_location_from_url(manifest_dir_url)
    backend = create_storage_backend(staging_dir=output_dir)
    locations = backend.list_files(manifest_dir, include=["*.json"])
    if not locations:
        raise FileNotFoundError(f"No JSON manifests found at {manifest_dir.s3_uri}")

    manifest_urls: dict[str, str] = {}
    for location in sorted(locations, key=lambda loc: loc.relative_path):
        manifest = read_manifest_location(location, output_dir=output_dir)
        pytorch_ref = read_manifest_data_pytorch_ref(manifest, location.s3_uri)
        if pytorch_ref in manifest_urls:
            raise ValueError(f"Duplicate manifest for PyTorch ref {pytorch_ref!r}")
        manifest_urls[pytorch_ref] = location.https_url

    return manifest_urls, UploadResult(
        count=len(manifest_urls),
        manifest_dir_url=manifest_dir.https_url,
        manifest_dir_s3_uri=manifest_dir.s3_uri,
    )


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


def append_manifest_summary(
    *,
    manifest_urls: dict[str, str],
    upload: UploadResult,
    pytorch_git_refs: list[str],
    matrix: dict[str, list[dict[str, str]]] | None = None,
    rocm_version: str = "",
) -> None:
    """Write a GitHub Actions summary for uploaded manifests."""
    base_url = upload.manifest_dir_url.rstrip("/")
    lines = [
        "## PyTorch Manifests",
        "",
        f"* Index: {base_url}/index.html",
        f"* S3 URI: `{upload.manifest_dir_s3_uri}`",
    ]

    if rocm_version:
        lines.append(f"* ROCm version: `{rocm_version}`")

    lines.extend(
        [
            "",
            "### Manifest Files",
            "",
            "| PyTorch ref | Manifest |",
            "| --- | --- |",
        ]
    )
    for pytorch_ref in pytorch_git_refs:
        manifest_url = manifest_urls[pytorch_ref]
        manifest_name = manifest_url.rsplit("/", maxsplit=1)[-1]
        lines.append(f"| `{pytorch_ref}` | [{manifest_name}]({manifest_url}) |")

    if matrix:
        lines.extend(
            [
                "",
                "### Build Matrix",
                "",
                "| PyTorch ref | Python version |",
                "| --- | --- |",
            ]
        )
        for entry in matrix["include"]:
            lines.append(
                f"| `{entry['pytorch_git_ref']}` | `{entry['python_version']}` |"
            )

    gha_append_step_summary("\n".join(lines) + "\n")


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
        "--manifest-dir-url",
        default="",
        help=(
            "Existing S3 manifest directory URL or s3:// URI to consume in "
            "matrix mode instead of generating manifests."
        ),
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
            "projects selected for each platform and PyTorch ref."
        ),
    )
    parser.add_argument(
        "--pytorch-git-refs",
        default="",
        help="Semicolon- or space-separated PyTorch refs.",
    )
    parser.add_argument(
        "--python-versions",
        default="",
        help="Semicolon- or space-separated Python versions for matrix mode.",
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
        if args.manifest_dir_url:
            raise ValueError("--manifest-url and --manifest-dir-url are exclusive")
        emit_single_outputs(args.manifest_url)
        return

    pytorch_refs = _split_words(args.pytorch_git_refs)
    python_versions = _split_words(args.python_versions)
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

    if args.manifest_dir_url:
        if args.output_mode != "matrix":
            raise ValueError(
                "--manifest-dir-url can only be used with --output-mode=matrix"
            )
        manifest_urls, upload = collect_manifest_urls_from_storage(
            manifest_dir_url=args.manifest_dir_url,
            output_dir=args.output_dir,
        )
        matrix = build_matrix(
            manifest_urls=manifest_urls,
            python_versions=python_versions,
            pytorch_git_refs=pytorch_refs,
            excludes=excludes,
        )
        append_manifest_summary(
            manifest_urls=manifest_urls,
            upload=upload,
            pytorch_git_refs=pytorch_refs,
            matrix=matrix,
            rocm_version=args.rocm_version,
        )
        emit_matrix_outputs(matrix, upload)
        return

    if not args.rocm_version:
        raise ValueError("--rocm-version is required when generating manifests")
    if not args.run_id:
        raise ValueError("--run-id is required when generating and uploading manifests")

    version_suffix = args.version_suffix or derive_version_suffix(args.rocm_version)
    projects = _split_words(args.projects) or None
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
        append_manifest_summary(
            manifest_urls=manifest_urls,
            upload=upload,
            pytorch_git_refs=pytorch_refs,
            rocm_version=args.rocm_version,
        )
        emit_single_outputs(manifest_urls[pytorch_ref], upload)
        return

    matrix = build_matrix(
        manifest_urls=manifest_urls,
        python_versions=python_versions,
        pytorch_git_refs=pytorch_refs,
        excludes=excludes,
    )
    append_manifest_summary(
        manifest_urls=manifest_urls,
        upload=upload,
        pytorch_git_refs=pytorch_refs,
        matrix=matrix,
        rocm_version=args.rocm_version,
    )
    emit_matrix_outputs(matrix, upload)


if __name__ == "__main__":
    main(sys.argv[1:])
