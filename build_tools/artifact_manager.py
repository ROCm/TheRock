#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

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

    # Fetch and flatten artifacts into single directory structure
    python stage_artifact_manager.py fetch \
        --stage math-libs \
        --amdgpu-families gfx94X-dcgpu \
        --run-id 12345 \
        --output-dir build/ \
        --flatten

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

from _therock_utils.cmake_amdgpu_targets import (
    build_family_to_targets,
    parse_amdgpu_targets_cmake,
)
from _therock_utils.build_topology import BuildTopology
from _therock_utils.artifact_backend import (
    ArtifactBackend,
    ARTIFACT_EXTENSIONS,
    LocalDirectoryBackend,
    S3Backend,
    create_backend_from_env,
)
from _therock_utils.artifacts import ArtifactName, ArtifactPopulator
from _therock_utils.workflow_outputs import WorkflowOutputRoot

# Component types that artifacts are split into
ARTIFACT_COMPONENTS = ["lib", "run", "dev", "dbg", "doc", "test"]

# Lazy-loaded cache for the family -> targets mapping parsed from CMake.
_family_to_targets_cache: Optional[dict[str, list[str]]] = None


def _get_family_to_targets() -> dict[str, list[str]]:
    """Return the family -> gfx targets mapping, parsed once from CMake."""
    global _family_to_targets_cache
    if _family_to_targets_cache is None:
        cmake_path = (
            Path(__file__).parent.parent / "cmake" / "therock_amdgpu_targets.cmake"
        )
        infos = parse_amdgpu_targets_cmake(cmake_path)
        _family_to_targets_cache = build_family_to_targets(infos)
    return _family_to_targets_cache


def log(msg: str):
    """Print message and flush."""
    print(msg, flush=True)


def _delay_for_retry(seconds: float):
    """Sleep for retry delay. Mockable for testing."""
    time.sleep(seconds)


def _get_pyzstd():
    """Lazy import pyzstd with helpful error message."""
    try:
        import pyzstd

        return pyzstd
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "pyzstd is required for zstd artifact decompression. "
            "Install it with: pip install pyzstd"
        )


def _open_archive_for_read(path: Path) -> tarfile.TarFile:
    """Open a tar archive for reading, auto-detecting compression type."""
    if path.name.endswith(".tar.zst"):
        pyzstd = _get_pyzstd()
        zstd_file = pyzstd.ZstdFile(path, mode="rb")
        return tarfile.TarFile(fileobj=zstd_file, mode="r")
    elif path.name.endswith(".tar.xz"):
        return tarfile.TarFile.open(path, mode="r:xz")
    else:
        raise ValueError(f"Unknown archive format: {path}")


def get_default_topology_path() -> Path:
    """Get the default BUILD_TOPOLOGY.toml path from the repository root."""
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    return repo_root / "BUILD_TOPOLOGY.toml"


def get_topology(topology_path: Optional[Path] = None) -> BuildTopology:
    """Load the BUILD_TOPOLOGY.toml.

    Args:
        topology_path: Path to topology file. If None, uses default location.
    """
    if topology_path is None:
        topology_path = get_default_topology_path()
    if not topology_path.exists():
        raise FileNotFoundError(f"BUILD_TOPOLOGY.toml not found at {topology_path}")
    return BuildTopology(str(topology_path))


# =============================================================================
# Shared Helpers
# =============================================================================


def parse_target_families(args: argparse.Namespace) -> List[str]:
    """Parse target families from argparse args.

    Returns a list starting with "generic", extended with any families and
    individual targets from the args.
    """
    output_families = ["generic"]
    if args.generic_only:
        log("Using generic (host) artifacts only")
    else:
        if args.amdgpu_families:
            input_families = args.amdgpu_families.split(";")
            output_families.extend(input_families)
            if args.expand_family_to_targets:
                family_map = _get_family_to_targets()
                for input_family in input_families:
                    for target in family_map.get(input_family, []):
                        if target not in output_families:
                            output_families.append(target)
        if args.amdgpu_targets:
            output_families.extend(
                t.strip() for t in args.amdgpu_targets.split(",") if t.strip()
            )
    return output_families


def find_available_artifacts(
    artifact_names: Set[str],
    target_families: List[str],
    available: Set[str],
) -> List[str]:
    """Find which artifacts exist in the available set.

    Iterates artifact_names × target_families × components × extensions,
    returning filenames that are present in `available`. Prefers .tar.zst
    over .tar.xz when both exist.
    """
    matched = []
    for artifact_name in sorted(artifact_names):
        for tf in target_families:
            for comp in ARTIFACT_COMPONENTS:
                for ext in ARTIFACT_EXTENSIONS:
                    filename = f"{artifact_name}_{comp}_{tf}{ext}"
                    if filename in available:
                        matched.append(filename)
                        break  # Found this artifact, don't check other extensions
    return matched


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
    """Download a single artifact with retry logic.

    Skips downloading if the file already exists in the download cache.
    """
    # TODO: Validate cached file integrity (e.g. compare size against S3
    # object metadata). A partial download from a killed process would pass
    # this check. Consider adding a --no-cache flag to bypass.
    if request.dest_path.exists():
        log(f"  == Cached {request.artifact_key}")
        return request.dest_path

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
                _delay_for_retry(delay)
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
        self.created_markers: List[Path] = []
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
            log(f"  ++ Flattening {archive_path.name} to {output_dir}")
            flattener = ArtifactPopulator(
                output_path=output_dir, verbose=False, flatten=True
            )
            flattener(archive_path)
        else:
            output_dir = request.output_dir / artifact_name
            if output_dir.exists():
                shutil.rmtree(output_dir)
            log(f"  ++ Extracting {archive_path.name}")
            with _open_archive_for_read(archive_path) as tf:
                tf.extractall(output_dir, filter="tar")

        if request.delete_archive:
            archive_path.unlink()

        return output_dir
    except Exception as e:
        log(f"  !! Failed to extract {request.archive_path.name}: {e}")
        return None


def do_fetch(args: argparse.Namespace):
    """Fetch inbound artifacts for a stage with parallel download and extract."""
    topology = get_topology(args.topology)

    # Determine which artifacts to fetch
    if args.stage == "all":
        # Fetch all artifacts in the topology
        inbound = set(topology.artifacts.keys())
        log(f"Fetching all {len(inbound)} artifacts")
    else:
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

    target_families = parse_target_families(args)

    # Create backend
    backend = create_backend_from_env(
        run_id=args.run_id,
        github_repository=args.run_github_repo,
        platform=args.platform,
    )
    log(f"Using backend: {backend.base_uri}")

    # Get list of available artifacts
    available = set(backend.list_artifacts())
    log(f"Found {len(available)} artifacts in backend")

    # Build download requests
    output_dir = Path(args.output_dir)
    shared_cache = args.download_cache_dir is not None
    download_dir = (
        args.download_cache_dir if shared_cache else output_dir / ".download_cache"
    )
    download_dir.mkdir(parents=True, exist_ok=True)

    matched_filenames = find_available_artifacts(inbound, target_families, available)

    download_requests = [
        DownloadRequest(
            artifact_key=filename,
            dest_path=download_dir / filename,
            backend=backend,
        )
        for filename in matched_filenames
    ]

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
                                    if not args.bootstrap and not args.flatten
                                    else output_dir
                                ),
                                # Don't delete during parallel extraction - cleanup
                                # happens after all extractions complete
                                delete_archive=False,
                                flatten=args.flatten,
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

    # Cleanup download cache (skip when using a shared cache dir)
    if download_dir.exists() and not args.no_extract and not shared_cache:
        shutil.rmtree(download_dir)

    # Fail if any downloads failed
    total_requested = len(download_requests)
    if downloaded_count < total_requested:
        log(
            f"ERROR: Only downloaded {downloaded_count}/{total_requested} artifacts - "
            f"{total_requested - downloaded_count} failed"
        )
        sys.exit(1)

    # Fail if any extractions failed (when extraction was requested)
    if not args.no_extract and extracted_count < downloaded_count:
        log(
            f"ERROR: Only extracted {extracted_count}/{downloaded_count} artifacts - "
            f"{downloaded_count - extracted_count} failed"
        )
        sys.exit(1)


# =============================================================================
# Push (Compress + Upload) with Parallel Processing
# =============================================================================


@dataclass
class CompressRequest:
    """Request to compress an artifact directory."""

    source_dir: Path
    archive_path: Path
    compression_type: str = "zstd"
    compression_level: Optional[int] = (
        None  # None = use algorithm default (3 for zstd, 6 for xz)
    )


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
            "--compression-type",
            request.compression_type,
        ]
        if request.compression_level is not None:
            cmd.extend(["--compression-level", str(request.compression_level)])
        cmd.extend(
            [
                "--hash-file",
                str(request.archive_path) + ".sha256sum",
                str(request.source_dir),
            ]
        )

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
                _delay_for_retry(delay)
            else:
                log(f"  !! Failed to upload {request.artifact_key}: {e}")
                return False
    return False


def do_push(args: argparse.Namespace):
    """Push produced artifacts after building with parallel compress and upload."""
    topology = get_topology(args.topology)

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
        github_repository=args.run_github_repo,
        platform=args.platform,
    )
    log(f"Using backend: {backend.base_uri}")

    # Find artifact directories in build directory
    build_dir = Path(args.build_dir)
    artifacts_dir = build_dir / "artifacts"
    if not artifacts_dir.exists():
        log(f"ERROR: Artifacts directory not found: {artifacts_dir}")
        sys.exit(1)

    # Find artifact directories to compress and upload
    # Check for both pre-compressed archives and exploded directories
    # Note: We push all artifacts produced by this stage regardless of target family.
    # The build system already determined what to build; filtering here is redundant
    # and breaks with kpack splitting (which produces individual arch artifacts).
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

            ext = ".tar.zst" if args.compression_type == "zstd" else ".tar.xz"
            archive_name = f"{item.name}{ext}"
            compress_requests.append(
                CompressRequest(
                    source_dir=item,
                    archive_path=upload_dir / archive_name,
                    compression_type=args.compression_type,
                    compression_level=args.compression_level,
                )
            )
        elif (item.suffix == ".xz" and item.name.endswith(".tar.xz")) or (
            item.suffix == ".zst" and item.name.endswith(".tar.zst")
        ):
            # Pre-compressed archive - direct upload
            an = ArtifactName.from_path(item)
            if not an:
                continue
            if an.name not in produced:
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

    # Fail if any artifacts failed to upload
    if uploaded_count < total_artifacts:
        log(
            f"ERROR: Only uploaded {uploaded_count}/{total_artifacts} artifacts - "
            f"{total_artifacts - uploaded_count} failed"
        )
        sys.exit(1)


# =============================================================================
# Copy (Server-side S3 copy between run IDs)
# =============================================================================


@dataclass
class CopyRequest:
    """Request to copy a single artifact between backends."""

    artifact_key: str
    source_backend: ArtifactBackend
    dest_backend: ArtifactBackend


def copy_single_artifact(request: CopyRequest) -> bool:
    """Copy a single artifact with retry logic."""
    MAX_RETRIES = 3
    BASE_DELAY_SECONDS = 2

    for attempt in range(MAX_RETRIES):
        try:
            log(f"  ++ Copying {request.artifact_key}")
            request.dest_backend.copy_artifact(
                request.artifact_key, request.source_backend
            )
            return True
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                delay = BASE_DELAY_SECONDS * (2**attempt)
                log(
                    f"  ++ Retry {attempt + 1}/{MAX_RETRIES} for {request.artifact_key}: {e}"
                )
                _delay_for_retry(delay)
            else:
                log(f"  !! Failed to copy {request.artifact_key}: {e}")
                return False
    return False


def _create_source_backend(
    source_run_id: str, platform: str, local_staging_dir: Optional[Path] = None
) -> ArtifactBackend:
    """Create a backend for the source run ID.

    For S3, uses WorkflowOutputRoot.from_workflow_run(lookup_workflow_run=True)
    to resolve the correct bucket (which may differ from the current run's bucket).

    For local backends, creates a LocalDirectoryBackend in the same staging dir.
    """
    if local_staging_dir or os.getenv("THEROCK_LOCAL_STAGING_DIR"):
        staging = local_staging_dir or Path(os.environ["THEROCK_LOCAL_STAGING_DIR"])
        output_root = WorkflowOutputRoot.for_local(
            run_id=source_run_id, platform=platform
        )
        return LocalDirectoryBackend(
            staging_dir=staging,
            output_root=output_root,
        )

    output_root = WorkflowOutputRoot.from_workflow_run(
        run_id=source_run_id, platform=platform, lookup_workflow_run=True
    )
    return S3Backend(output_root=output_root)


def do_copy(args: argparse.Namespace):
    """Copy produced artifacts for one or more stages from one run to another."""
    topology = get_topology(args.topology)

    # Parse and validate stages (comma-separated). Unlike fetch/push which
    # operate on a single stage, copy accepts multiple stages at once so that
    # a single setup job can copy all prebuilt stages in one invocation.
    stage_names = [s.strip() for s in args.stage.split(",") if s.strip()]
    available_stages = topology.build_stages.keys()
    for stage_name in stage_names:
        if stage_name not in available_stages:
            log(f"ERROR: Stage '{stage_name}' not found")
            log(f"Available stages: {', '.join(available_stages)}")
            sys.exit(1)

    # Union produced artifacts across all specified stages
    produced: Set[str] = set()
    for stage_name in stage_names:
        stage_produced = topology.get_produced_artifacts(stage_name)
        log(
            f"Stage '{stage_name}' produces {len(stage_produced)} artifacts: {', '.join(sorted(stage_produced))}"
        )
        produced.update(stage_produced)

    if not produced:
        log("Specified stages produce no artifacts")
        return

    target_families = parse_target_families(args)

    # Create source and dest backends
    source_backend = _create_source_backend(
        source_run_id=args.source_run_id,
        platform=args.platform,
        local_staging_dir=args.local_staging_dir,
    )
    dest_backend = create_backend_from_env(
        run_id=args.run_id,
        platform=args.platform,
    )

    log(f"Source: {source_backend.base_uri}")
    log(f"Dest:   {dest_backend.base_uri}")

    # List available artifacts in source
    available = set(source_backend.list_artifacts())
    log(f"Found {len(available)} artifacts in source")

    # Build copy requests from matched artifacts
    matched_filenames = find_available_artifacts(produced, target_families, available)
    copy_requests = [
        CopyRequest(
            artifact_key=filename,
            source_backend=source_backend,
            dest_backend=dest_backend,
        )
        for filename in matched_filenames
    ]

    if not copy_requests:
        log("No matching artifacts found to copy")
        return

    if args.dry_run:
        log(f"\nDry run: would copy {len(copy_requests)} artifacts:")
        for req in copy_requests:
            log(f"  {req.artifact_key}")
        return

    log(f"\nCopying {len(copy_requests)} artifacts...")

    failed_artifacts = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.concurrency
    ) as executor:
        futures = {
            executor.submit(copy_single_artifact, req): req for req in copy_requests
        }
        for future in concurrent.futures.as_completed(futures):
            if not future.result():
                failed_artifacts.append(futures[future].artifact_key)

    copied_count = len(copy_requests) - len(failed_artifacts)
    log(f"\nCopied {copied_count}/{len(copy_requests)} artifacts")

    if failed_artifacts:
        log(f"ERROR: {len(failed_artifacts)} artifacts failed to copy:")
        for name in sorted(failed_artifacts):
            log(f"  - {name}")
        sys.exit(1)


# =============================================================================
# Info Commands
# =============================================================================


def do_info(args: argparse.Namespace):
    """Show information about stage artifact requirements."""
    topology = get_topology(args.topology)

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
        families = args.amdgpu_families.split(";")
        target_families = ["generic"] + families
        log(f"\nTarget families: {', '.join(target_families)}")


def do_list_stages(args: argparse.Namespace):
    """List all build stages."""
    topology = get_topology(args.topology)

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


def _add_common_args(parser: argparse.ArgumentParser):
    """Add common arguments shared by all subcommands."""
    parser.add_argument(
        "--topology",
        type=Path,
        default=None,
        help="Path to BUILD_TOPOLOGY.toml (default: auto-detect from repo root)",
    )


def _add_target_args(parser: argparse.ArgumentParser):
    """Add target family/GPU arguments to a subparser."""
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument(
        "--amdgpu-families",
        type=str,
        help="Semicolon-separated GPU families (e.g., gfx94X-dcgpu;gfx1100)",
    )
    target_group.add_argument(
        "--generic-only",
        action="store_true",
        help="Only use generic (host) artifacts, skip device-specific artifacts",
    )
    parser.add_argument(
        "--amdgpu-targets",
        type=str,
        default="",
        help="Comma-separated individual GPU targets for split artifacts (e.g. 'gfx942')",
    )
    parser.add_argument(
        "--expand-family-to-targets",
        action="store_true",
        help=(
            "Expand each --amdgpu-families entry to its constituent gfx targets "
            "using cmake/therock_amdgpu_targets.cmake. Use when fetching kpack-split "
            "artifacts that are named per target rather than per family. "
            "Safe to pass against non-kpack-split buckets: the family name is still "
            "matched and the extra per-target lookups simply find nothing."
        ),
    )


def _add_backend_args(parser: argparse.ArgumentParser):
    """Add common backend-related arguments to a subparser."""
    _add_common_args(parser)
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
        "--run-github-repo",
        type=str,
        default=None,
        help="GitHub repository for --run-id in 'owner/repo' format (e.g. 'ROCm/TheRock'). "
        "Defaults to GITHUB_REPOSITORY env var or 'ROCm/TheRock'",
    )
    parser.add_argument(
        "--local-staging-dir",
        type=Path,
        default=os.getenv("THEROCK_LOCAL_STAGING_DIR"),
        help="Local staging directory (sets THEROCK_LOCAL_STAGING_DIR)",
    )


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(
        description="Stage-aware artifact manager for multi-stage CI/CD pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # fetch command
    fetch_parser = subparsers.add_parser(
        "fetch", help="Fetch inbound artifacts for a stage"
    )
    _add_backend_args(fetch_parser)
    fetch_parser.add_argument(
        "--stage",
        type=str,
        default="all",
        help="Build stage name (default: 'all' fetches all artifacts)",
    )
    _add_target_args(fetch_parser)
    fetch_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for fetched artifacts",
    )
    fetch_extract_group = fetch_parser.add_mutually_exclusive_group()
    fetch_extract_group.add_argument(
        "--bootstrap",
        action="store_true",
        help="Bootstrap build directory (flatten artifacts and create prebuilt markers)",
    )
    fetch_extract_group.add_argument(
        "--flatten",
        action="store_true",
        help="Flatten artifacts into a single directory structure (merge all artifacts)",
    )
    fetch_parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Download only, do not extract",
    )
    fetch_parser.add_argument(
        "--download-cache-dir",
        type=Path,
        default=None,
        help="Shared download cache directory. When set, downloaded archives "
        "are kept after extraction so subsequent fetches can reuse them. "
        "Defaults to OUTPUT_DIR/.download_cache (cleaned up after extraction).",
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
        help="Number of concurrent extractions (default: auto)",
    )
    fetch_parser.set_defaults(func=do_fetch)

    # push command
    push_parser = subparsers.add_parser(
        "push", help="Push produced artifacts after building"
    )
    _add_backend_args(push_parser)
    push_parser.add_argument(
        "--stage", type=str, required=True, help="Build stage name"
    )
    push_parser.add_argument(
        "--build-dir",
        type=Path,
        required=True,
        help="Build directory containing artifacts/",
    )
    push_parser.add_argument(
        "--compression-type",
        type=str,
        default="zstd",
        choices=["zstd", "xz"],
        help="Compression algorithm (default: zstd)",
    )
    push_parser.add_argument(
        "--compression-level",
        type=int,
        default=None,
        help="Compression level (default: 3 for zstd, 6 for xz)",
    )
    push_parser.add_argument(
        "--compress-concurrency",
        type=int,
        default=None,
        help="Number of concurrent compressions (default: auto)",
    )
    push_parser.add_argument(
        "--upload-concurrency",
        type=int,
        default=10,
        help="Number of concurrent uploads (default: 10)",
    )
    push_parser.set_defaults(func=do_push)

    # copy command
    copy_parser = subparsers.add_parser(
        "copy",
        help="Copy produced artifacts for a stage from one run to another",
    )
    _add_backend_args(copy_parser)
    copy_parser.add_argument(
        "--source-run-id",
        type=str,
        required=True,
        help="Run ID to copy artifacts from (bucket resolved via GitHub API)",
    )
    copy_parser.add_argument(
        "--stage",
        type=str,
        required=True,
        help="Build stage name(s), comma-separated (e.g., 'foundation,compiler-runtime')",
    )
    _add_target_args(copy_parser)
    copy_parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Number of concurrent copy operations (default: 10)",
    )
    copy_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be copied without actually copying",
    )
    copy_parser.set_defaults(func=do_copy)

    # info command
    info_parser = subparsers.add_parser(
        "info", help="Show information about stage artifact requirements"
    )
    _add_common_args(info_parser)
    info_parser.add_argument(
        "--stage", type=str, required=True, help="Build stage name"
    )
    info_parser.add_argument(
        "--amdgpu-families",
        type=str,
        help="Semicolon-separated GPU families to show file lists for",
    )
    info_parser.set_defaults(func=do_info)

    # list-stages command
    list_parser = subparsers.add_parser("list-stages", help="List all build stages")
    _add_common_args(list_parser)
    list_parser.set_defaults(func=do_list_stages)

    args = parser.parse_args(argv)

    # Set environment variable if --local-staging-dir provided (only on fetch/push)
    local_staging_dir = getattr(args, "local_staging_dir", None)
    if local_staging_dir:
        os.environ["THEROCK_LOCAL_STAGING_DIR"] = str(local_staging_dir)

    args.func(args)


if __name__ == "__main__":
    main()
