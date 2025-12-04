#!/usr/bin/env python
"""Stage-aware artifact manager for multi-stage CI/CD pipeline.

This CLI tool manages artifacts between build stages, supporting both local
directories (for prototyping) and S3 (for CI/CD). Operations are parallelized
for performance.

Usage:
    # Fetch inbound artifacts for a stage (downloads and extracts in parallel)
    python stage_artifact_manager.py fetch \
        --stage math-libs \
        --amdgpu-families gfx94X-dcgpu \
        --run-id 12345 \
        --output-dir build/

    # Push produced artifacts after building (compresses and uploads in parallel)
    python stage_artifact_manager.py push \
        --stage math-libs \
        --amdgpu-families gfx94X-dcgpu \
        --run-id 12345 \
        --build-dir build/

    # List what artifacts a stage needs/produces
    python stage_artifact_manager.py info \
        --stage math-libs \
        --amdgpu-families gfx94X-dcgpu

Environment variables:
    THEROCK_LOCAL_STAGING_DIR: Use local directory instead of S3
    THEROCK_RUN_ID: Override run ID
    THEROCK_PLATFORM: Override platform (default: current platform)
"""

import argparse
import concurrent.futures
from dataclasses import dataclass
import os
import platform as platform_module
import shutil
import sys
import tarfile
import threading
import time
from pathlib import Path
from typing import List, Optional, Set

from _therock_utils.build_topology import BuildTopology
from _therock_utils.artifact_backend import (
    ArtifactBackend,
    create_backend_from_env,
)
from _therock_utils.artifacts import ArtifactName, ArtifactPopulator


def log(msg: str):
    """Print message and flush."""
    print(msg, flush=True)


def get_topology() -> BuildTopology:
    """Load the BUILD_TOPOLOGY.toml from the repository root."""
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    topology_path = repo_root / "BUILD_TOPOLOGY.toml"
    if not topology_path.exists():
        raise FileNotFoundError(f"BUILD_TOPOLOGY.toml not found at {topology_path}")
    return BuildTopology(str(topology_path))


# =============================================================================
# Fetch (Download + Extract) with Parallel Processing
# =============================================================================


@dataclass
class DownloadRequest:
    """Request to download an artifact."""

    artifact_key: str
    dest_path: Path
    backend: ArtifactBackend


@dataclass
class ExtractRequest:
    """Request to extract a downloaded artifact."""

    archive_path: Path
    output_dir: Path
    delete_archive: bool
    flatten: bool
    bootstrap: bool = False
    # Shared state for parallel bootstrap extraction
    cleaned_paths: Optional[set] = None
    cleaned_paths_lock: Optional[threading.Lock] = None


def download_artifact(request: DownloadRequest) -> Optional[Path]:
    """Download a single artifact with retry logic."""
    MAX_RETRIES = 3
    BASE_DELAY_SECONDS = 2

    for attempt in range(MAX_RETRIES):
        try:
            log(f"  ++ Downloading {request.artifact_key}")
            request.dest_path.parent.mkdir(parents=True, exist_ok=True)
            request.backend.download_artifact(request.artifact_key, request.dest_path)
            return request.dest_path
        except FileNotFoundError:
            # Artifact doesn't exist - not an error, just skip
            return None
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                delay = BASE_DELAY_SECONDS * (2**attempt)
                log(
                    f"  ++ Retry {attempt + 1}/{MAX_RETRIES} for {request.artifact_key}: {e}"
                )
                time.sleep(delay)
            else:
                log(f"  !! Failed to download {request.artifact_key}: {e}")
                return None
    return None


class BootstrappingPopulator(ArtifactPopulator):
    """ArtifactPopulator that creates .prebuilt markers for bootstrapping.

    When used with parallel extraction, pass shared state (cleaned_paths set
    and lock) to avoid race conditions where multiple threads try to
    clean/populate the same directory paths.
    """

    def __init__(
        self,
        output_path: Path,
        verbose: bool = False,
        cleaned_paths: Optional[set] = None,
        cleaned_paths_lock: Optional[threading.Lock] = None,
    ):
        super().__init__(output_path=output_path, verbose=verbose, flatten=False)
        self.created_markers: list[Path] = []
        self._cleaned_paths = cleaned_paths if cleaned_paths is not None else set()
        self._lock = (
            cleaned_paths_lock if cleaned_paths_lock is not None else threading.Lock()
        )

    def on_first_relpath(self, relpath: str):
        if not relpath:
            return  # Skip empty relpaths

        full_path = self.output_path / relpath

        with self._lock:
            if relpath in self._cleaned_paths:
                return  # Already cleaned by another thread
            self._cleaned_paths.add(relpath)

            # Do cleanup while holding lock to prevent race with extraction
            if full_path.exists():
                shutil.rmtree(full_path)
            # Write the ".prebuilt" marker file
            prebuilt_path = full_path.with_name(full_path.name + ".prebuilt")
            prebuilt_path.parent.mkdir(parents=True, exist_ok=True)
            prebuilt_path.touch()
            self.created_markers.append(prebuilt_path)


def extract_artifact(request: ExtractRequest) -> Optional[Path]:
    """Extract a single artifact archive."""
    try:
        archive_path = request.archive_path
        artifact_name, *_ = archive_path.name.partition(".")

        if request.bootstrap:
            # Bootstrap mode: use ArtifactPopulator with prebuilt markers
            output_dir = request.output_dir
            log(f"  ++ Bootstrapping {archive_path.name}")
            populator = BootstrappingPopulator(
                output_path=output_dir,
                verbose=False,
                cleaned_paths=request.cleaned_paths,
                cleaned_paths_lock=request.cleaned_paths_lock,
            )
            populator(archive_path)
        elif request.flatten:
            output_dir = request.output_dir
            log(f"  ++ Flattening {archive_path.name}")
            flattener = ArtifactPopulator(
                output_path=output_dir, verbose=False, flatten=True
            )
            flattener(archive_path)
        else:
            output_dir = request.output_dir / artifact_name
            if output_dir.exists():
                shutil.rmtree(output_dir)
            log(f"  ++ Extracting {archive_path.name}")
            with tarfile.TarFile.open(archive_path, mode="r:xz") as tf:
                tf.extractall(output_dir, filter="tar")

        if request.delete_archive:
            archive_path.unlink()

        return output_dir
    except Exception as e:
        log(f"  !! Failed to extract {request.archive_path.name}: {e}")
        return None


def do_fetch(args: argparse.Namespace):
    """Fetch inbound artifacts for a stage with parallel download and extract."""
    topology = get_topology()

    # Validate stage
    if args.stage not in topology.build_stages:
        log(f"ERROR: Stage '{args.stage}' not found")
        log(f"Available stages: {', '.join(topology.build_stages.keys())}")
        sys.exit(1)

    # Get inbound artifacts for this stage
    inbound = topology.get_inbound_artifacts(args.stage)
    if not inbound:
        log(f"Stage '{args.stage}' has no inbound artifacts")
        return

    log(
        f"Stage '{args.stage}' needs {len(inbound)} artifacts: {', '.join(sorted(inbound))}"
    )

    # Determine target families
    target_families = ["generic"]
    if args.amdgpu_families:
        target_families.extend(args.amdgpu_families.split(","))

    # Create backend
    backend = create_backend_from_env(
        run_id=args.run_id,
        platform=args.platform,
    )
    log(f"Using backend: {backend.base_uri}")

    # Get list of available artifacts
    available = set(backend.list_artifacts())
    log(f"Found {len(available)} artifacts in backend")

    # Build download requests
    output_dir = Path(args.output_dir)
    download_dir = output_dir / ".download_cache"
    download_dir.mkdir(parents=True, exist_ok=True)

    components = ["lib", "run", "dev", "dbg", "doc", "test"]
    download_requests = []

    for artifact_name in sorted(inbound):
        for tf in target_families:
            for comp in components:
                filename = f"{artifact_name}_{comp}_{tf}.tar.xz"
                if filename in available:
                    download_requests.append(
                        DownloadRequest(
                            artifact_key=filename,
                            dest_path=download_dir / filename,
                            backend=backend,
                        )
                    )

    if not download_requests:
        log("No matching artifacts found to download")
        return

    log(f"\nDownloading {len(download_requests)} artifacts...")

    # Parallel download and extract pipeline
    downloaded_count = 0
    extracted_count = 0

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.download_concurrency
    ) as download_executor:
        download_futures = [
            download_executor.submit(download_artifact, req)
            for req in download_requests
        ]

        if args.no_extract:
            # Just wait for downloads
            for future in concurrent.futures.as_completed(download_futures):
                result = future.result()
                if result:
                    downloaded_count += 1
        else:
            # Pipeline: extract as downloads complete
            # For bootstrap mode, create shared state to coordinate parallel
            # extractions that may write to overlapping paths
            bootstrap_cleaned_paths: set = set()
            bootstrap_lock = threading.Lock()

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=args.extract_concurrency
            ) as extract_executor:
                extract_futures = []

                for download_future in concurrent.futures.as_completed(
                    download_futures
                ):
                    downloaded_path = download_future.result()
                    if not downloaded_path or not downloaded_path.exists():
                        continue
                    downloaded_count += 1
                    extract_futures.append(
                        extract_executor.submit(
                            extract_artifact,
                            ExtractRequest(
                                archive_path=downloaded_path,
                                output_dir=(
                                    output_dir / "artifacts"
                                    if not args.bootstrap
                                    else output_dir
                                ),
                                # Don't delete during parallel extraction - cleanup
                                # happens after all extractions complete
                                delete_archive=False,
                                flatten=False,
                                bootstrap=args.bootstrap,
                                cleaned_paths=(
                                    bootstrap_cleaned_paths if args.bootstrap else None
                                ),
                                cleaned_paths_lock=(
                                    bootstrap_lock if args.bootstrap else None
                                ),
                            ),
                        )
                    )

                for future in concurrent.futures.as_completed(extract_futures):
                    result = future.result()
                    if result:
                        extracted_count += 1

    log(f"\nDownloaded {downloaded_count} artifacts, extracted {extracted_count}")

    # Cleanup download cache
    if download_dir.exists() and not args.no_extract:
        shutil.rmtree(download_dir)


# =============================================================================
# Push (Compress + Upload) with Parallel Processing
# =============================================================================


@dataclass
class CompressRequest:
    """Request to compress an artifact directory."""

    source_dir: Path
    archive_path: Path
    compression_level: int = 6


@dataclass
class UploadRequest:
    """Request to upload an artifact."""

    source_path: Path
    artifact_key: str
    backend: ArtifactBackend


def compress_artifact(request: CompressRequest) -> Optional[Path]:
    """Compress a single artifact directory using fileset_tool.py artifact-archive."""
    try:
        log(f"  ++ Compressing {request.source_dir.name}")
        request.archive_path.parent.mkdir(parents=True, exist_ok=True)

        # Use fileset_tool.py artifact-archive for proper archive creation
        import subprocess

        script_dir = Path(__file__).parent
        fileset_tool = script_dir / "fileset_tool.py"

        cmd = [
            sys.executable,
            str(fileset_tool),
            "artifact-archive",
            "-o",
            str(request.archive_path),
            "--compression-level",
            str(request.compression_level),
            "--hash-file",
            str(request.archive_path) + ".sha256sum",
            str(request.source_dir),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"fileset_tool.py artifact-archive failed (returncode={result.returncode}): {result.stderr}"
            )

        return request.archive_path
    except Exception as e:
        log(f"  !! Failed to compress {request.source_dir.name}: {e}")
        return None


def upload_artifact(request: UploadRequest) -> bool:
    """Upload a single artifact with retry logic."""
    MAX_RETRIES = 3
    BASE_DELAY_SECONDS = 2

    for attempt in range(MAX_RETRIES):
        try:
            log(f"  ++ Uploading {request.artifact_key}")
            request.backend.upload_artifact(request.source_path, request.artifact_key)

            # Also upload sha256sum if it exists
            sha_path = request.source_path.with_suffix(
                request.source_path.suffix + ".sha256sum"
            )
            if sha_path.exists():
                request.backend.upload_artifact(
                    sha_path, f"{request.artifact_key}.sha256sum"
                )

            return True
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                delay = BASE_DELAY_SECONDS * (2**attempt)
                log(
                    f"  ++ Retry {attempt + 1}/{MAX_RETRIES} for {request.artifact_key}: {e}"
                )
                time.sleep(delay)
            else:
                log(f"  !! Failed to upload {request.artifact_key}: {e}")
                return False
    return False


def do_push(args: argparse.Namespace):
    """Push produced artifacts after building with parallel compress and upload."""
    topology = get_topology()

    # Validate stage
    if args.stage not in topology.build_stages:
        log(f"ERROR: Stage '{args.stage}' not found")
        log(f"Available stages: {', '.join(topology.build_stages.keys())}")
        sys.exit(1)

    # Get produced artifacts for this stage
    produced = topology.get_produced_artifacts(args.stage)
    if not produced:
        log(f"Stage '{args.stage}' produces no artifacts")
        return

    log(
        f"Stage '{args.stage}' produces {len(produced)} artifacts: {', '.join(sorted(produced))}"
    )

    # Create backend
    backend = create_backend_from_env(
        run_id=args.run_id,
        platform=args.platform,
    )
    log(f"Using backend: {backend.base_uri}")

    # Find artifact directories in build directory
    build_dir = Path(args.build_dir)
    artifacts_dir = build_dir / "artifacts"
    if not artifacts_dir.exists():
        log(f"ERROR: Artifacts directory not found: {artifacts_dir}")
        sys.exit(1)

    # Determine target families
    target_families = ["generic"]
    if args.amdgpu_families:
        target_families.extend(args.amdgpu_families.split(","))

    # Find artifact directories to compress and upload
    # Check for both pre-compressed archives and exploded directories
    upload_dir = build_dir / ".upload_cache"
    upload_dir.mkdir(parents=True, exist_ok=True)

    compress_requests = []
    direct_upload_requests = []

    for item in artifacts_dir.iterdir():
        if item.is_dir():
            # Exploded artifact directory - needs compression
            an = ArtifactName.from_path(item)
            if not an:
                continue
            if an.name not in produced:
                continue
            if an.target_family not in target_families:
                continue

            archive_name = f"{item.name}.tar.xz"
            # Use low compression for arch-specific artifacts since device
            # code is already compressed
            compression = 1 if an.target_family != "generic" else args.compression_level
            compress_requests.append(
                CompressRequest(
                    source_dir=item,
                    archive_path=upload_dir / archive_name,
                    compression_level=compression,
                )
            )
        elif item.suffix == ".xz" and item.name.endswith(".tar.xz"):
            # Pre-compressed archive - direct upload
            an = ArtifactName.from_path(item)
            if not an:
                continue
            if an.name not in produced:
                continue
            if an.target_family not in target_families:
                continue

            direct_upload_requests.append(
                UploadRequest(
                    source_path=item,
                    artifact_key=item.name,
                    backend=backend,
                )
            )

    total_artifacts = len(compress_requests) + len(direct_upload_requests)
    if total_artifacts == 0:
        log("No matching artifacts found to push")
        return

    log(
        f"\nProcessing {total_artifacts} artifacts ({len(compress_requests)} to compress, {len(direct_upload_requests)} pre-compressed)..."
    )

    compressed_count = 0
    uploaded_count = 0

    # Parallel compress and upload pipeline
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.compress_concurrency
    ) as compress_executor:
        compress_futures = [
            compress_executor.submit(compress_artifact, req)
            for req in compress_requests
        ]

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.upload_concurrency
        ) as upload_executor:
            upload_futures = []

            # Start uploading pre-compressed archives immediately
            for req in direct_upload_requests:
                upload_futures.append(upload_executor.submit(upload_artifact, req))

            # As compression completes, start uploads
            for compress_future in concurrent.futures.as_completed(compress_futures):
                archive_path = compress_future.result()
                if not archive_path or not archive_path.exists():
                    continue
                compressed_count += 1
                upload_futures.append(
                    upload_executor.submit(
                        upload_artifact,
                        UploadRequest(
                            source_path=archive_path,
                            artifact_key=archive_path.name,
                            backend=backend,
                        ),
                    )
                )

            # Wait for all uploads
            for future in concurrent.futures.as_completed(upload_futures):
                if future.result():
                    uploaded_count += 1

    log(f"\nCompressed {compressed_count} artifacts, uploaded {uploaded_count}")

    # Cleanup upload cache
    if upload_dir.exists():
        shutil.rmtree(upload_dir)


# =============================================================================
# Info Commands
# =============================================================================


def do_info(args: argparse.Namespace):
    """Show information about stage artifact requirements."""
    topology = get_topology()

    stage = topology.build_stages.get(args.stage)
    if not stage:
        log(f"ERROR: Stage '{args.stage}' not found")
        log(f"Available stages: {', '.join(topology.build_stages.keys())}")
        sys.exit(1)

    log(f"Stage: {stage.name}")
    log(f"Type: {stage.type}")
    log(f"Description: {stage.description}")
    log(f"Artifact groups: {', '.join(stage.artifact_groups)}")

    # Inbound artifacts
    inbound = topology.get_inbound_artifacts(args.stage)
    log(f"\nInbound artifacts ({len(inbound)}):")
    for name in sorted(inbound):
        artifact = topology.artifacts.get(name)
        if artifact:
            log(f"  - {name} ({artifact.type})")
        else:
            log(f"  - {name}")

    # Produced artifacts
    produced = topology.get_produced_artifacts(args.stage)
    log(f"\nProduced artifacts ({len(produced)}):")
    for name in sorted(produced):
        artifact = topology.artifacts.get(name)
        if artifact:
            log(f"  - {name} ({artifact.type})")
        else:
            log(f"  - {name}")

    # Show target families if provided
    if args.amdgpu_families:
        families = args.amdgpu_families.split(",")
        target_families = ["generic"] + families
        log(f"\nTarget families: {', '.join(target_families)}")


def do_list_stages(args: argparse.Namespace):
    """List all build stages."""
    topology = get_topology()

    log("Build stages:")
    for stage in topology.get_build_stages():
        inbound = topology.get_inbound_artifacts(stage.name)
        produced = topology.get_produced_artifacts(stage.name)
        log(f"\n  {stage.name} ({stage.type})")
        log(f"    {stage.description}")
        log(f"    Groups: {', '.join(stage.artifact_groups)}")
        log(f"    Inbound: {len(inbound)} artifacts")
        log(f"    Produces: {len(produced)} artifacts")


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Stage-aware artifact manager for multi-stage CI/CD pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Common arguments
    parser.add_argument(
        "--run-id",
        type=str,
        default=os.getenv("THEROCK_RUN_ID", os.getenv("GITHUB_RUN_ID", "local")),
        help="Run ID for artifact storage (default: from env or 'local')",
    )
    parser.add_argument(
        "--platform",
        type=str,
        default=os.getenv("THEROCK_PLATFORM", platform_module.system().lower()),
        help="Platform name (default: current platform)",
    )
    parser.add_argument(
        "--local-staging-dir",
        type=Path,
        default=os.getenv("THEROCK_LOCAL_STAGING_DIR"),
        help="Local staging directory (sets THEROCK_LOCAL_STAGING_DIR)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # fetch command
    fetch_parser = subparsers.add_parser(
        "fetch", help="Fetch inbound artifacts for a stage"
    )
    fetch_parser.add_argument(
        "--stage", type=str, required=True, help="Build stage name"
    )
    fetch_parser.add_argument(
        "--amdgpu-families",
        type=str,
        help="Comma-separated GPU families (e.g., gfx94X-dcgpu,gfx110X-all)",
    )
    fetch_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for fetched artifacts",
    )
    fetch_parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Bootstrap build directory (flatten artifacts and create prebuilt markers)",
    )
    fetch_parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Download only, do not extract",
    )
    fetch_parser.add_argument(
        "--download-concurrency",
        type=int,
        default=10,
        help="Number of concurrent downloads (default: 10)",
    )
    fetch_parser.add_argument(
        "--extract-concurrency",
        type=int,
        default=None,
        help="Number of concurrent extractions (default: CPU count)",
    )
    fetch_parser.set_defaults(func=do_fetch)

    # push command
    push_parser = subparsers.add_parser(
        "push", help="Push produced artifacts after building"
    )
    push_parser.add_argument(
        "--stage", type=str, required=True, help="Build stage name"
    )
    push_parser.add_argument(
        "--amdgpu-families",
        type=str,
        help="Comma-separated GPU families (e.g., gfx94X-dcgpu,gfx110X-all)",
    )
    push_parser.add_argument(
        "--build-dir",
        type=Path,
        required=True,
        help="Build directory containing artifacts/",
    )
    push_parser.add_argument(
        "--compression-level",
        type=int,
        default=6,
        choices=range(0, 10),
        help="XZ compression level 0-9 (default: 6)",
    )
    push_parser.add_argument(
        "--compress-concurrency",
        type=int,
        default=None,
        help="Number of concurrent compressions (default: CPU count)",
    )
    push_parser.add_argument(
        "--upload-concurrency",
        type=int,
        default=10,
        help="Number of concurrent uploads (default: 10)",
    )
    push_parser.set_defaults(func=do_push)

    # info command
    info_parser = subparsers.add_parser(
        "info", help="Show information about stage artifact requirements"
    )
    info_parser.add_argument(
        "--stage", type=str, required=True, help="Build stage name"
    )
    info_parser.add_argument(
        "--amdgpu-families",
        type=str,
        help="Comma-separated GPU families to show file lists for",
    )
    info_parser.set_defaults(func=do_info)

    # list-stages command
    list_parser = subparsers.add_parser("list-stages", help="List all build stages")
    list_parser.set_defaults(func=do_list_stages)

    args = parser.parse_args()

    # Set environment variable if --local-staging-dir provided
    if args.local_staging_dir:
        os.environ["THEROCK_LOCAL_STAGING_DIR"] = str(args.local_staging_dir)

    args.func(args)


if __name__ == "__main__":
    main()
