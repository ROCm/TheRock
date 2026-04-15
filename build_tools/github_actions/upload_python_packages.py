#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
This script uploads built Python packages (wheels, sdists) along with an index
page to S3 or a local directory for testing. Once packages are uploaded, they
can be downloaded directly or via `pip install --find-links {url}` by
developers, users, and test workflows.

Usage:
  upload_python_packages.py
    --input-packages-dir PACKAGES_DIR
    --run-id RUN_ID
    [--artifact-group ARTIFACT_GROUP]  # Required for single-arch; omit with --multiarch
    [--multiarch]              # Multi-arch mode: omits artifact_group from upload path
    [--output-dir OUTPUT_DIR]  # Local output instead of S3
    [--bucket BUCKET]          # Override bucket selection (defaults to auto-select)
    [--dry-run]                # Print what would happen without taking action

Modes:
  1. S3 upload (default): Uploads to an AWS S3 bucket
  2. Local output: With --output-dir, copies files to local directory
  3. Dry run: With --dry-run, prints plan without uploading or copying

Output Layout:
  {bucket}/{external_repo}{run_id}-{platform}/python/{artifact_group}/
    *.whl, *.tar.gz   # Wheel and sdist files
    index.html        # File listing for pip --find-links

  For multi-arch builds (--multiarch), artifact_group is omitted:
  {bucket}/{external_repo}{run_id}-{platform}/python/
    {family}/         # Per-family subdirectory (gfx94X-dcgpu, etc.)
      index.html
    *.whl, *.tar.gz

Installation:
  pip install rocm[libraries,devel] --pre \\
    --find-links=https://{bucket}.s3.amazonaws.com/{path}/index.html
"""

import argparse
from pathlib import Path
import platform
import shlex
import subprocess
import sys

_BUILD_TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BUILD_TOOLS_DIR))
sys.path.insert(0, str(_BUILD_TOOLS_DIR / "packaging" / "python"))
from _therock_utils.workflow_outputs import WorkflowOutputRoot
from _therock_utils.storage_location import StorageLocation
from _therock_utils.storage_backend import StorageBackend, create_storage_backend
from generate_local_index import generate_flat_index, generate_multiarch_indexes
from github_actions_api import (
    gha_append_step_summary,
    gha_set_output,
)

THEROCK_DIR = Path(__file__).resolve().parent.parent.parent
PLATFORM = platform.system().lower()
LINE_CONTINUATION_CHAR = "^" if PLATFORM == "windows" else "\\"


def log(*args):
    print(*args)
    sys.stdout.flush()


def run_command(cmd: list[str], cwd: Path = Path.cwd()):
    log(f"++ Exec [{cwd}]$ {shlex.join(cmd)}")
    subprocess.run(cmd, check=True)


def _make_output_root(
    run_id: str, bucket_override: str | None = None
) -> WorkflowOutputRoot:
    if bucket_override:
        return WorkflowOutputRoot(
            bucket=bucket_override, external_repo="", run_id=run_id, platform=PLATFORM
        )
    return WorkflowOutputRoot.from_workflow_run(run_id=run_id, platform=PLATFORM)


def generate_index(dist_dir: Path, multiarch: bool = False, dry_run: bool = False):
    """Generates an index.html file listing packages for pip --find-links.

    Args:
        dist_dir: Directory containing packages to index
        multiarch: If True, generate per-family indexes for multi-arch builds
        dry_run: If True, print command without executing
    """
    if dry_run:
        log(f"[DRY RUN] Would generate index in {dist_dir} (multiarch={multiarch})")
        return

    if multiarch:
        has_subdirs = any(d.is_dir() for d in dist_dir.iterdir())
        if has_subdirs:
            log("[INFO] Multi-arch legacy mode: generating per-family indexes")
            generate_multiarch_indexes(dist_dir)
        else:
            log("[INFO] kpack-split flat mode: generating single top-level index")
            generate_flat_index(dist_dir)
    else:
        # Single-arch mode: use existing indexer.py for top-level
        log("[INFO] Single-arch mode: using indexer.py")
        indexer_script = THEROCK_DIR / "third-party" / "indexer" / "indexer.py"
        if not indexer_script.is_file():
            raise FileNotFoundError(f"Indexer script not found: {indexer_script}")

        cmd = [
            sys.executable,
            str(indexer_script),
            str(dist_dir),
            "--filter",
            "*.whl",
            "*.tar.gz",
        ]

        run_command(cmd)


def find_package_files(dist_dir: Path) -> list[Path]:
    """Finds all wheel, sdist, and index files in the dist directory."""
    files = []
    for pattern in ["*.whl", "*.tar.gz", "index.html"]:
        files.extend(dist_dir.glob(pattern))

    return sorted(files)


def upload_packages(
    dist_dir: Path, packages_loc: StorageLocation, backend: StorageBackend
):
    """Upload package files using the provided backend."""
    package_files = find_package_files(dist_dir)
    if not package_files:
        raise FileNotFoundError(f"No package files found in {dist_dir}")

    log(f"[INFO] Found {len(package_files)} top-level package files in {dist_dir}:")
    for f in package_files:
        log(f"  - {f.relative_to(dist_dir)}")

    # Log all files that will actually be uploaded (including subdirectories).
    all_files = sorted(
        f for f in dist_dir.rglob("*") if f.is_file() and not f.is_symlink()
    )
    log(f"[INFO] Uploading {len(all_files)} total files to {packages_loc.s3_uri}:")
    for f in all_files:
        log(f"  {f.relative_to(dist_dir).as_posix()}")

    count = backend.upload_directory(dist_dir, packages_loc)
    log(f"[INFO] Uploaded {count} files")


def write_gha_upload_summary(
    packages_loc: StorageLocation, families: list[str] | None = None
):
    """Write GitHub Actions summary with pip install instructions.

    Args:
        packages_loc: Storage location for packages
        families: Controls the summary format:
            - None: single-arch mode — single index URL
            - [] (empty list): kpack-split flat mode — single index URL covering
              all per-target wheels
            - [...] non-empty: legacy per-family mode — per-family install links
    """
    if families:
        # Legacy per-family multi-arch
        base_url = packages_loc.https_url
        family_links = "\n".join(
            f"- [{family}]({base_url}/{family}/index.html)" for family in families
        )
        family_installs = "\n\n".join(
            f"```bash\npip install rocm[libraries,devel] --pre {LINE_CONTINUATION_CHAR}\n"
            f"    --find-links={base_url}/{family}/index.html\n```"
            for family in families
        )
        install_instructions_markdown = f"""ROCm Python packages (multi-arch build)

Per-family indexes:
{family_links}

{family_installs}
"""
    elif families is not None:
        # kpack-split flat build — single index covers all per-target wheels
        index_url = f"{packages_loc.https_url}/index.html"
        install_instructions_markdown = f"""[ROCm Python packages (kpack-split)]({index_url})
Replace `<YOUR_TARGET>` with your GPU target (e.g. `gfx942`, `gfx1201`):
```bash
pip install rocm[libraries,devel,device-<YOUR_TARGET>] --pre {LINE_CONTINUATION_CHAR}
    --find-links={index_url}
```
"""
    else:
        # Single-arch: traditional index URL
        index_url = f"{packages_loc.https_url}/index.html"
        install_instructions_markdown = f"""[ROCm Python packages]({index_url})
```bash
pip install rocm[libraries,devel] --pre {LINE_CONTINUATION_CHAR}
    --find-links={index_url}
```
"""
    gha_append_step_summary(install_instructions_markdown)


def run(args: argparse.Namespace):
    packages_dir = args.input_packages_dir.resolve()
    if not packages_dir.is_dir():
        raise FileNotFoundError(f"Packages root directory not found: {packages_dir}")

    dist_dir = packages_dir / "dist"
    if not dist_dir.is_dir():
        raise FileNotFoundError(f"Packages dist/ subdirectory not found: {dist_dir}")

    log(f"[INFO] Packages directory: {packages_dir}")
    log(f"[INFO] Dist subdirectory : {dist_dir}")
    log(f"[INFO] Artifact group    : {args.artifact_group}")
    log(f"[INFO] Run ID            : {args.run_id}")
    log(f"[INFO] Platform          : {PLATFORM}")
    if args.dry_run:
        log(f"[INFO] Mode              : DRY RUN")
    elif args.output_dir:
        log(f"[INFO] Mode              : Local output to {args.output_dir}")
    else:
        log(f"[INFO] Mode              : S3 upload")

    log("")
    log("Generating index.html")
    log("---------------------")
    log(f"[INFO] Multi-arch indexing: {args.multiarch}")
    generate_index(dist_dir, multiarch=args.multiarch, dry_run=args.dry_run)

    output_root = _make_output_root(args.run_id, bucket_override=args.bucket)
    # Multi-arch builds don't need an artifact_group subdirectory: the run_id
    # already uniquely identifies the build, and there is only one Python build
    # job per multi-arch run.
    packages_loc = output_root.python_packages(
        "" if args.multiarch else args.artifact_group
    )
    backend = create_storage_backend(staging_dir=args.output_dir, dry_run=args.dry_run)

    log("")
    log("Uploading packages")
    log("------------------")
    upload_packages(dist_dir=dist_dir, packages_loc=packages_loc, backend=backend)

    if not args.output_dir:
        if args.multiarch:
            # Detect flat (kpack-split) vs legacy per-family layout from dist/.
            family_subdirs = sorted(d.name for d in dist_dir.iterdir() if d.is_dir())
            if family_subdirs:
                # Legacy per-family: return base URL; downstream appends /{family}/index.html
                index_url = packages_loc.https_url
                kpack_split = "false"
                log(
                    f"[INFO] Multi-arch legacy URL (tests append /{{family}}/index.html): {index_url}"
                )
            else:
                # kpack-split flat: all wheels at top level; return full /index.html URL
                index_url = f"{packages_loc.https_url}/index.html"
                kpack_split = "true"
                log(f"[INFO] kpack-split flat URL: {index_url}")
        else:
            family_subdirs = None
            index_url = f"{packages_loc.https_url}/index.html"
            kpack_split = "false"
            log(f"[INFO] Single-arch index URL: {index_url}")

        log("Set github actions output")
        log("-------------------------")
        gha_set_output(
            {"package_find_links_url": index_url, "kpack_split": kpack_split}
        )

        log("Write github actions build summary")
        log("----------------------------------")
        # families: None=single-arch, []=kpack-split flat, [...]=legacy per-family
        families = family_subdirs if args.multiarch else None
        write_gha_upload_summary(packages_loc, families=families)

    log("")
    log("[INFO] Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Upload Python packages to S3 or a local directory"
    )
    parser.add_argument(
        "--input-packages-dir",
        type=Path,
        required=True,
        help="Directory containing built packages (with dist/ subdirectory)",
    )
    parser.add_argument(
        "--artifact-group",
        type=str,
        default="",
        help="Artifact group (e.g., gfx94X-dcgpu). Omit for multi-arch builds.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="Workflow run ID (e.g. 21440027240)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output to local directory instead of S3",
    )
    parser.add_argument(
        "--bucket",
        type=str,
        default=None,
        help="Override S3 bucket (default: auto-select from workflow run)",
    )
    parser.add_argument(
        "--multiarch",
        action="store_true",
        help="Multi-arch mode: generate per-family indexes (required for multi-arch builds with family-specific wheels)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without uploading or copying",
    )

    args = parser.parse_args()

    if args.output_dir and args.bucket:
        parser.error("--output-dir and --bucket are mutually exclusive")
    if not args.multiarch and not args.artifact_group:
        parser.error("--artifact-group is required for single-arch builds")

    run(args)


if __name__ == "__main__":
    main()
