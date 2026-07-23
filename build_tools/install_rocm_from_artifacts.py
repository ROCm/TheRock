#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""install_rocm_from_artifacts.py

This script helps CI workflows, developers and testing suites easily install
TheRock to their environment using artifacts. It installs TheRock to an output
directory from one of these sources:

  - GitHub CI workflow run
  - Release tag
  - An existing installation of TheRock

Usage:
python build_tools/install_rocm_from_artifacts.py
    (--artifact-group ARTIFACT_GROUP | --amdgpu_family AMDGPU_FAMILY)
    [--output-dir OUTPUT_DIR]
    (--run-id RUN_ID | --release RELEASE | --latest-release | --input-dir INPUT_DIR)
    [--dry-run]
    [--run-github-repo RUN_GITHUB_REPO]
    [--aqlprofile | --no-aqlprofile]
    [--blas | --no-blas]
    [--debug-tools | --no-debug-tools]
    [--fft | --no-fft]
    [--hipdnn | --no-hipdnn]
    [--hipdnn-integration-tests | --no-hipdnn-integration-tests]
    [--hipdnn-samples | --no-hipdnn-samples]
    [--hipfile | --no-hipfile]
    [--miopen | --no-miopen]
    [--miopenprovider | --no-miopenprovider]
    [--hipblasltprovider | --no-hipblasltprovider]
    [--hipkernelprovider | --no-hipkernelprovider]
    [--prim | --no-prim]
    [--rand | --no-rand]
    [--rccl | --no-rccl]
    [--rocdecode | --no-rocdecode]
    [--rocjpeg | --no-rocjpeg]
    [--rocjitsu | --no-rocjitsu]
    [--mirage | --no-mirage]
    [--rocprofiler-compute | --no-rocprofiler-compute]
    [--rocprofiler-sdk | --no-rocprofiler-sdk ]
    [--rocprofiler-systems | --no-rocprofiler-systems]
    [--rocprofiler-systems-examples | --no-rocprofiler-systems-examples]
    [--rocrtst | --no-rocrtst]
    [--rocalution | --no-rocalution]
    [--rocwmma | --no-rocwmma]
    [--hiptensor | --no-hiptensor]
    [--libhipcxx | --no-libhipcxx]
    [--tests | --no-tests]
    [--base-only]

Examples:
- Downloads and unpacks the gfx94X S3 artifacts from GitHub CI workflow run 14474448215
  (from https://github.com/ROCm/TheRock/actions/runs/14474448215) to the
  default output directory `therock-build`:
    ```
    python build_tools/install_rocm_from_artifacts.py \
        --run-id 14474448215 \
        --amdgpu-family gfx94X-dcgpu \
        --tests
    ```
- Downloads and unpacks the version `6.4.0rc20250416` gfx110X artifacts from
  the multi-arch nightly tarball index to the specified output directory `build`:
    ```
    python build_tools/install_rocm_from_artifacts.py \
        --release 6.4.0rc20250416 \
        --amdgpu-family gfx110X-all \
        --output-dir build
    ```
- Downloads and unpacks the version `6.4.0.dev0+8f6cdfc0d95845f4ca5a46de59d58894972a29a9`
  gfx120X artifacts from release tag `dev-tarball` to the default output directory `therock-build`:
    ```
    python build_tools/install_rocm_from_artifacts.py \
        --release 6.4.0.dev0+8f6cdfc0d95845f4ca5a46de59d58894972a29a9 \
        --amdgpu-family gfx120X-all
    ```
- Downloads and unpacks the gfx94X S3 artifacts from GitHub CI workflow run 19644138192
  (from https://github.com/ROCm/rocm-libraries/actions/runs/19644138192) in the `ROCm/rocm-libraries` repository to the
  default output directory `therock-build`:
    ```
    python build_tools/install_rocm_from_artifacts.py \
        --run-id 19644138192 \
        --amdgpu-family gfx94X-dcgpu \
        --tests \
        --run-github-repo ROCm/rocm-libraries
    ```
- Downloads and unpacks the latest nightly release for gfx110X:
    ```
    python build_tools/install_rocm_from_artifacts.py \
        --latest-release \
        --amdgpu-family gfx110X-all
    ```
- Shows what would be downloaded without actually downloading (works with any mode):
    ```
    python build_tools/install_rocm_from_artifacts.py \
        --latest-release \
        --amdgpu-family gfx110X-all \
        --dry-run

    python build_tools/install_rocm_from_artifacts.py \
        --release 7.11.0a20260119 \
        --amdgpu-family gfx110X-all \
        --dry-run
    ```
You can select your AMD GPU family from therock_amdgpu_targets.cmake.

By default for CI workflow retrieval, all artifacts (excluding test artifacts)
will be downloaded. For specific artifacts, pass in the flag such as `--rand`
(RAND artifacts) For test artifacts, pass in the flag `--tests` (test artifacts).
For base artifacts only, pass in the flag `--base-only`

Note that the ARTIFACT_GROUP controls which sub-directory of the run contains
the artifacts. If not specified, it defaults to the AMDGPU_FAMILY, which was
the historic interpretation.

Note: the script will overwrite the output directory argument. If no argument
is passed, it will overwrite the default "therock-build" directory.
"""

import argparse
import json
from datetime import datetime
from html.parser import HTMLParser
from fetch_artifacts import main as fetch_artifacts_main
from _therock_utils.cmake_amdgpu_targets import amdgpu_family_map, expand_families
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tarfile
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.parse import quote, unquote, urljoin, urlparse
from urllib.request import urlopen

PLATFORM = platform.system().lower()
NIGHTLY_TARBALL_INDEX_URL = "https://rocm.nightlies.amd.com/tarball-multi-arch/"
DEV_TARBALL_INDEX_URL = "https://rocm.devreleases.amd.com/tarball-multi-arch/"


def parse_nightly_version(version: str) -> Optional[datetime]:
    """
    Parse nightly version like '7.11.0a20251124' to extract date.
    Returns datetime for sorting, None if not parseable.
    """
    match = re.search(r"(\d+)\.(\d+)\.(\d+)(a|rc)(\d{4})(\d{2})(\d{2})", version)
    if match:
        year, month, day = int(match.group(5)), int(match.group(6)), int(match.group(7))
        return datetime(year, month, day)
    return None


def extract_version_from_asset_name(
    asset_name: str, artifact_group: str, platform_str: str
) -> Optional[str]:
    """
    Extract version string from asset name.
    E.g., 'therock-dist-linux-gfx110X-all-7.11.0a20251124.tar.gz' -> '7.11.0a20251124'
    """
    prefix = f"therock-dist-{platform_str}-{artifact_group}-"
    suffix = ".tar.gz"
    if asset_name.startswith(prefix) and asset_name.endswith(suffix):
        version = asset_name[len(prefix) : -len(suffix)]
        if version.startswith("tests-"):
            return None
        return version
    return None


class _TarballIndexParser(HTMLParser):
    """Collect tarball asset names from the multi-arch directory listing."""

    def __init__(self) -> None:
        super().__init__()
        self.asset_last_modified: dict[str, datetime] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        if tag != "a":
            return

        href = dict(attrs).get("href")
        if href is not None:
            asset_name = _tarball_asset_name(href)
            if asset_name:
                self.asset_last_modified.setdefault(asset_name, datetime.min)


def _tarball_asset_name(value: str) -> Optional[str]:
    """Extract a tarball asset name from an index link or file entry."""
    asset_name = unquote(urlparse(value).path).rsplit("/", maxsplit=1)[-1]
    return asset_name if asset_name.endswith(".tar.gz") else None


def _extract_multiarch_tarball_assets(index_html: str) -> dict[str, datetime]:
    """Extract tarball names and modification times from the multi-arch index."""
    parser = _TarballIndexParser()
    parser.feed(index_html)
    parser.close()

    files_marker = re.search(r"\bconst\s+files\s*=", index_html)
    if files_marker is None:
        return parser.asset_last_modified

    try:
        file_entries, _ = json.JSONDecoder().raw_decode(
            index_html[files_marker.end() :].lstrip()
        )
    except json.JSONDecodeError as ex:
        raise ValueError("multi-arch tarball index contains invalid file data") from ex

    if not isinstance(file_entries, list):
        raise ValueError("multi-arch tarball index file data is not a list")

    for entry in file_entries:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        mtime = entry.get("mtime")
        if isinstance(name, str):
            asset_name = _tarball_asset_name(name)
            if asset_name:
                if isinstance(mtime, (int, float)):
                    parser.asset_last_modified[asset_name] = datetime.fromtimestamp(
                        mtime
                    )
                else:
                    parser.asset_last_modified.setdefault(asset_name, datetime.min)

    return parser.asset_last_modified


def _fetch_multiarch_tarball_assets() -> dict[str, datetime]:
    """Fetch tarball names and modification times from the nightly index."""
    try:
        with urlopen(NIGHTLY_TARBALL_INDEX_URL) as response:
            index_html = response.read().decode("utf-8")
        return _extract_multiarch_tarball_assets(index_html)
    except (HTTPError, URLError, OSError, UnicodeDecodeError, ValueError) as ex:
        log(f"ERROR: Failed to fetch multi-arch tarball index: {ex}")
        raise


def _fetch_multiarch_tarball_asset_names() -> set[str]:
    """Fetch tarball asset names from the multi-arch nightly tarball index."""
    return set(_fetch_multiarch_tarball_assets())


def _tarball_url(tarball_index_url: str, asset_name: str) -> str:
    """Return an asset URL within a multi-arch tarball feed."""
    return urljoin(tarball_index_url, quote(asset_name))


def list_available_nightly_gpu_families(platform_str: str = PLATFORM) -> set[str]:
    """
    Query the multi-arch tarball index to find GPU families with nightly releases.
    Useful for error messages when an invalid GPU family is specified.
    """
    prefix = f"therock-dist-{platform_str}-"
    asset_pattern = re.compile(
        rf"^{re.escape(prefix)}(.+)-\d+\.\d+\.\d+(?:(?:a|rc)\d{{8}})?\.tar\.gz$"
    )
    families: set[str] = set()

    for asset_name in _fetch_multiarch_tarball_asset_names():
        if "-tests-" in asset_name:
            continue
        match = asset_pattern.match(asset_name)
        if match:
            families.add(match.group(1))

    return families


def _fetch_and_sort_nightly_releases(
    artifact_group: str,
    platform_str: str = PLATFORM,
) -> list[dict]:
    """
    Fetch and sort nightly releases from the multi-arch tarball index.

    Returns:
        List of dicts with keys: version, asset_name, last_modified, parsed_date
        Sorted by recency (newest first).
    """
    releases: list[dict] = []

    for asset_name, last_modified in _fetch_multiarch_tarball_assets().items():
        version = extract_version_from_asset_name(
            asset_name, artifact_group, platform_str
        )
        if version:
            releases.append(
                {
                    "version": version,
                    "asset_name": asset_name,
                    "last_modified": last_modified,
                    "parsed_date": parse_nightly_version(version),
                }
            )

    releases.sort(
        key=lambda x: (
            x["parsed_date"] if x["parsed_date"] else datetime.min,
            x["last_modified"],
        ),
        reverse=True,
    )
    return releases


def discover_latest_release(
    artifact_group: str,
    platform_str: str = PLATFORM,
) -> Optional[tuple[str, str]]:
    """
    Query the multi-arch tarball index for the latest nightly release.

    Returns:
        Tuple of (version_string, full_asset_name) or None if not found.
    """
    releases = _fetch_and_sort_nightly_releases(artifact_group, platform_str)
    if not releases:
        return None
    return (releases[0]["version"], releases[0]["asset_name"])


def log(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


def _untar_files(output_dir: Path, destination: Path):
    """
    Retrieves all tar files in the output_dir, then extracts all files to the output_dir
    """
    log(f"Extracting {destination.name} to {str(output_dir)}")
    with tarfile.open(destination) as extracted_tar_file:
        extracted_tar_file.extractall(output_dir)
    destination.unlink()


def _create_output_directory(output_dir: Path):
    """
    If the output directory already exists, delete it and its contents.
    Then, create the output directory.
    """
    log(f"Creating output directory '{output_dir.resolve()}'")
    if output_dir.is_dir():
        log(
            f"Directory '{output_dir}' already exists, removing existing directory and files"
        )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    log(f"Created output directory '{output_dir.resolve()}'")


def _retrieve_multiarch_tarball(
    asset_name: str,
    output_dir: Path,
    tarball_index_url: str = NIGHTLY_TARBALL_INDEX_URL,
) -> None:
    """Download a multi-arch tarball to disk, then extract it."""
    tarball_url = _tarball_url(tarball_index_url, asset_name)
    destination = output_dir / asset_name
    log(f"Downloading {tarball_url}")

    try:
        with urlopen(tarball_url) as response, open(destination, "wb") as file:
            shutil.copyfileobj(response, file)
    except (HTTPError, URLError, OSError) as ex:
        log(f"ERROR: Failed to download {tarball_url}: {ex}")
        raise

    _untar_files(output_dir, destination)


def retrieve_artifacts_by_run_id(args):
    """
    If the user requested TheRock artifacts by CI (run ID), this function will retrieve those assets
    """
    run_id = args.run_id
    log(f"Retrieving artifacts for run ID {run_id}")
    argv = [
        "--run-id",
        run_id,
        "--artifact-group",
        args.artifact_group,
        "--output-dir",
        str(args.output_dir),
        "--flatten",
    ]
    if args.amdgpu_targets:
        argv.extend(["--amdgpu-targets", args.amdgpu_targets])
    else:
        # Auto-derive gfx targets from the family so a family-only install also
        # fetches per-target (kpack-split) shards. In split runs the per-target
        # shard carries data the family/'generic' shards lack (e.g. MIOpen
        # tuning DBs under share/miopen/db); without these targets the fetch
        # matches only the family literal and 'generic', silently dropping them.
        derived_targets = expand_families(
            [args.artifact_group], amdgpu_family_map(), strict=False
        )
        if derived_targets:
            log(
                f"Auto-deriving --amdgpu-targets from family "
                f"'{args.artifact_group}': {','.join(derived_targets)}"
            )
            argv.extend(["--amdgpu-targets", ",".join(derived_targets)])
    if args.dry_run:
        argv.append("--dry-run")
    if args.run_github_repo:
        argv.extend(["--run-github-repo", args.run_github_repo])

    # These artifacts are the "base" requirements for running tests.
    base_artifact_patterns = [
        "core-hipinfo_run",
        "core-runtime_run",
        "core-runtime_lib",
        "sysdeps_lib",
        "base_run",
        "base_lib",
        "amd-llvm_run",
        "amd-llvm_lib",
        "core-amdsmi_run",
        "core-amdsmi_lib",
        "core-hip_lib",
        "core-hip_dev",
        "core-kpack_lib",
        "core-ocl_lib",
        "core-ocl_dev",
        "rocprofiler-sdk_lib",
        "host-suite-sparse_lib",
    ]

    if args.base_only:
        argv.extend(base_artifact_patterns)
    elif any(
        [
            args.aqlprofile,
            args.blas,
            args.debug_tools,
            args.fft,
            args.hipdnn,
            args.hipdnn_integration_tests,
            args.hipdnn_samples,
            args.hipfile,
            args.miopen,
            args.miopenprovider,
            args.hiptensor,
            args.hipblasltprovider,
            args.hipkernelprovider,
            args.prim,
            args.mpi,
            args.rand,
            args.rccl,
            args.rocdecode,
            args.rocjpeg,
            args.rocjitsu,
            args.mirage,
            args.rocprofiler_compute,
            args.rocprofiler_sdk,
            args.rocprofiler_systems,
            args.rocprofiler_systems_examples,
            args.rocrtst,
            args.rocalution,
            args.rocwmma,
            args.libhipcxx,
        ]
    ):
        argv.extend(base_artifact_patterns)

        extra_artifacts = []
        if args.aqlprofile:
            extra_artifacts.append("aqlprofile")
        if args.blas:
            extra_artifacts.append("blas")
        if args.debug_tools:
            extra_artifacts.append("amd-dbgapi")
            extra_artifacts.append("rocgdb")
            extra_artifacts.append("rocr-debug-agent")
            extra_artifacts.append("rocr-debug-agent-tests")
            # Contains the rocgdb executable.
            argv.append("rocgdb_run")

            # Libraries rocgdb depends on.
            extra_artifacts.append("gmp")
            extra_artifacts.append("mpfr")
            extra_artifacts.append("expat")
            extra_artifacts.append("ncurses")
            # rocgdb tests require amd-llvm_dev for compiler headers/tools.
            argv.append("amd-llvm_dev")
        if args.fft:
            extra_artifacts.append("fft")
            extra_artifacts.append("fftw3")
        if args.hipdnn:
            extra_artifacts.append("hipdnn")
        if args.hipdnn_integration_tests:
            extra_artifacts.append("hipdnn-integration-tests")
            # The main test binary `hipdnn_integration_tests` is in the artifact's
            # _run component (per ml-libs/artifact-hipdnn-integration-tests.toml).
            # Provider cross-provider integration suites (e.g. miopenprovider's
            # external-integration-check) invoke it with --test-article and
            # --test-engine; without _run, ctest finds the entry but errors with
            # "Unable to find executable: ../hipdnn_integration_tests".
            argv.append("hipdnn-integration-tests_run")
        if args.hipdnn_samples:
            extra_artifacts.append("hipdnn-samples")
        if args.hipfile:
            extra_artifacts.append("hipfile")
            extra_artifacts.append("sysdeps-util-linux")
        if args.miopen:
            extra_artifacts.append("miopen")
            # Contains bin/MIOpenDriver executable for tests.
            argv.append("miopen_run")
            # Also need these for runtime kernel compilation (rocrand includes).
            argv.append("rand_dev")
        if args.miopenprovider:
            extra_artifacts.append("miopenprovider")
        if args.hipkernelprovider:
            extra_artifacts.append("hipkernelprovider")
        if args.hiptensor:
            extra_artifacts.append("hiptensor")
        if args.rocdecode:
            extra_artifacts.append("sysdeps-amd-mesa")
            extra_artifacts.append("rocdecode")
            argv.append("rocdecode_dev")
            argv.append("rocdecode_test")
            argv.append("base_dev")
            argv.append("amd-llvm_dev")
        if args.mpi:
            extra_artifacts.append("openmpi")
            # Ensure binaries like mpiexec are installed
            argv.append("openmpi_run")
            # Optional but useful (headers, dev libs)
            argv.append("openmpi_dev")
        if args.rocjpeg:
            extra_artifacts.append("sysdeps-amd-mesa")
            extra_artifacts.append("rocjpeg")
            argv.append("rocjpeg_dev")
            argv.append("rocjpeg_test")
            argv.append("base_dev")
            argv.append("amd-llvm_dev")
        if args.rocjitsu:
            extra_artifacts.append("rocjitsu")
            argv.append("rocjitsu_run")
        if args.mirage:
            extra_artifacts.append("mirage")
            argv.append("mirage_run")
        if args.hipblasltprovider:
            extra_artifacts.append("hipblasltprovider")
        if args.prim:
            extra_artifacts.append("prim")
        if args.rand:
            extra_artifacts.append("rand")
        if args.rccl:
            extra_artifacts.append("rccl")
        if args.rocprofiler_sdk:
            extra_artifacts.append("rocprofiler-sdk")
            extra_artifacts.append("aqlprofile")
            # Contains rocprofiler-sdk-rocpd
            argv.append("rocprofiler-sdk_run")
        if args.rocprofiler_compute:
            extra_artifacts.append("rocprofiler-compute")
            # Contains the rocprof-compute CLI executable.
            argv.append("rocprofiler-compute_run")
        if args.rocprofiler_systems:
            extra_artifacts.append("rocprofiler-systems")
            # Contains executables (rocprof-sys-run, rocprof-sys-instrument, etc.)
            argv.append("rocprofiler-systems_run")
            if args.tests:
                # Tests need version.h for rocprofiler-sdk version detection.
                argv.append("rocprofiler-sdk_dev")
        if args.rocprofiler_systems_examples:
            # Only a _test artifact is produced
            argv.append("rocprofiler-systems-examples_test")
        if args.rocrtst:
            extra_artifacts.append("rocrtst")
            # rocrtst depends on sysdeps-hwloc (which depends on sysdeps-libpciaccess)
            extra_artifacts.append("sysdeps-hwloc")
            extra_artifacts.append("sysdeps-libpciaccess")
        if args.rocalution:
            extra_artifacts.append("rocalution")
            argv.append("rocalution_dev")
        if args.rocwmma:
            extra_artifacts.append("rocwmma")
            argv.append("rocwmma_dev")
        if args.libhipcxx:
            extra_artifacts.append("libhipcxx")
            argv.append("amd-llvm_dev")
            argv.append("amd-llvm_lib")
            argv.append("base_dev_generic")

        # Fetch _lib (always) and _test (when --tests) for each artifact.
        # Some projects have self-contained _test archives (just test
        # binaries), while others may also need executables or data from
        # _run. Add those explicitly above via argv.append("<name>_run").
        extra_artifact_patterns = [f"{a}_lib" for a in extra_artifacts]
        if args.tests:
            extra_artifact_patterns.extend([f"{a}_test" for a in extra_artifacts])

        argv.extend(extra_artifact_patterns)
    else:
        # No include (or exclude) patterns, so all artifacts will be fetched.
        pass

    log(f"\nCalling fetch_artifacts_main with args:\n  {' '.join(argv)}\n")
    fetch_artifacts_main(argv)

    log(f"Retrieved artifacts for run ID {run_id}")


def retrieve_artifacts_by_release(args):
    """
    If the user requested TheRock artifacts by release version, this function will retrieve those assets
    """
    output_dir = args.output_dir
    artifact_group = args.artifact_group
    # Determine if version is nightly or dev.
    nightly_regex_expression = (
        "(\\d+\\.)?(\\d+\\.)?(\\*|\\d+)(a|rc)(\\d{4})(\\d{2})(\\d{2})"
    )
    dev_regex_expression = "(\\d+\\.)?(\\d+\\.)?(\\*|\\d+).dev0+"
    nightly_release = re.search(nightly_regex_expression, args.release) != None
    dev_release = re.search(dev_regex_expression, args.release) != None
    if not nightly_release and not dev_release:
        log("This script requires a nightly or dev release version.")
        log("Please retrieve the correct release version from:")
        log(
            "\t - https://rocm.nightlies.amd.com/tarball-multi-arch/ (nightly examples: 6.4.0rc20250416, 7.10.0a20251024)"
        )
        log(
            "\t - https://rocm.devreleases.amd.com/tarball-multi-arch/ (dev-tarball example: 6.4.0.dev0+8f6cdfc0d95845f4ca5a46de59d58894972a29a9)"
        )
        log("Exiting...")
        return

    release_version = args.release
    asset_name = f"therock-dist-{PLATFORM}-{artifact_group}-{release_version}.tar.gz"
    tarball_index_url = (
        NIGHTLY_TARBALL_INDEX_URL if nightly_release else DEV_TARBALL_INDEX_URL
    )
    release_kind = "nightly" if nightly_release else "dev"
    tarball_url = _tarball_url(tarball_index_url, asset_name)

    log(
        f"Retrieving {release_kind} artifacts from multi-arch tarball feed "
        f"{tarball_index_url}"
    )
    if args.dry_run:
        log(
            f"[DRY RUN] Would download: {tarball_url} "
            f"(asset {asset_name}, version {release_version})"
        )
        return

    _retrieve_multiarch_tarball(asset_name, output_dir, tarball_index_url)


def retrieve_artifacts_by_input_dir(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    log(f"Retrieving artifacts from input dir {input_dir}")

    if args.dry_run:
        log(f"[DRY RUN] Would rsync from {input_dir} to {output_dir}")
        return

    # Check to make sure rsync exists
    if not shutil.which("rsync"):
        log("Error: rsync command not found.")
        if platform.system() == "Windows":
            log("Please install rsync via MSYS2 or WSL to your Windows system")
        return

    cmd = [
        "rsync",
        "-azP",  # archive, compress and progress indicator
        input_dir,
        output_dir,
    ]
    try:
        subprocess.run(cmd, check=True)
        log(f"Retrieved artifacts from input dir {input_dir} to {output_dir}")
    except Exception as ex:
        # rsync is not available
        log(f"Error when running [{cmd}]")
        log(str(ex))


def retrieve_artifacts_by_latest_release(args):
    """
    Find and retrieve the latest nightly release from the multi-arch tarball index.
    """
    log(f"Finding latest nightly release for {args.artifact_group}...")

    result = discover_latest_release(artifact_group=args.artifact_group)

    if result is None:
        log(f"ERROR: No nightly release found for '{args.artifact_group}'")
        log("")
        log("Available GPU families in the multi-arch tarball index:")
        available = list_available_nightly_gpu_families()
        for family in sorted(available):
            log(f"  - {family}")
        sys.exit(1)

    version, asset_name = result
    log(f"Found latest release: {version}")

    if args.dry_run:
        log(f"[DRY RUN] Would download: {asset_name} (version {version})")
        return

    _retrieve_multiarch_tarball(asset_name, args.output_dir)


def run(args):
    log("### Installing TheRock using artifacts ###")

    # Skip directory creation for dry-run
    if not args.dry_run:
        _create_output_directory(args.output_dir)

    if args.run_id:
        retrieve_artifacts_by_run_id(args)
    elif args.release:
        retrieve_artifacts_by_release(args)
    elif args.latest_release:
        retrieve_artifacts_by_latest_release(args)

    if args.input_dir:
        retrieve_artifacts_by_input_dir(args)


def main(argv):
    parser = argparse.ArgumentParser(prog="provision")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="./therock-build",
        help="Path of the output directory for TheRock",
    )

    artifact_group_parser = parser.add_mutually_exclusive_group(required=True)
    artifact_group_parser.add_argument(
        "--artifact-group",
        dest="artifact_group",
        type=str,
        help="Explicit artifact group to install",
    )
    artifact_group_parser.add_argument(
        "--amdgpu-family",
        dest="artifact_group",
        type=str,
        help="AMD GPU family to install (please refer to this: https://github.com/ROCm/TheRock/blob/59c324a759e8ccdfe5a56e0ebe72a13ffbc04c1f/cmake/therock_amdgpu_targets.cmake#L44-L81 for family choices)",
    )

    # This mutually exclusive group will ensure that only one argument is present
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-id", type=str, help="GitHub run ID of TheRock to install")

    group.add_argument(
        "--release",
        type=str,
        help="Release version of TheRock to install, from the multi-arch nightly tarball index (X.Y.ZrcYYYYMMDD) or dev-tarball (X.Y.Z.dev0+{hash})",
    )

    group.add_argument(
        "--latest-release",
        action="store_true",
        help="Install the latest nightly release (built daily from main branch)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded/copied without actually doing it",
    )

    artifacts_group = parser.add_argument_group("artifacts_group")
    artifacts_group.add_argument(
        "--aqlprofile",
        default=False,
        help="Include 'aqlprofile' artifacts",
        action=argparse.BooleanOptionalAction,
    )

    artifacts_group.add_argument(
        "--blas",
        default=False,
        help="Include 'blas' artifacts",
        action=argparse.BooleanOptionalAction,
    )

    artifacts_group.add_argument(
        "--debug-tools",
        default=False,
        help="Include ROCm debugging tools (amd-dbgapi, rocgdb and rocr_debug_agent) artifacts",
        action=argparse.BooleanOptionalAction,
    )

    artifacts_group.add_argument(
        "--fft",
        default=False,
        help="Include 'fft' artifacts",
        action=argparse.BooleanOptionalAction,
    )

    artifacts_group.add_argument(
        "--hipdnn",
        default=False,
        help="Include 'hipdnn' artifacts",
        action=argparse.BooleanOptionalAction,
    )

    artifacts_group.add_argument(
        "--hipdnn-integration-tests",
        default=False,
        help="Include 'hipdnn-integration-tests' artifacts",
        action=argparse.BooleanOptionalAction,
    )

    artifacts_group.add_argument(
        "--hipdnn-samples",
        default=False,
        help="Include 'hipdnn-samples' artifacts",
        action=argparse.BooleanOptionalAction,
    )

    artifacts_group.add_argument(
        "--hipfile",
        default=False,
        help="Include 'hipfile' artifacts",
        action=argparse.BooleanOptionalAction,
    )

    artifacts_group.add_argument(
        "--miopen",
        default=False,
        help="Include 'miopen' artifacts",
        action=argparse.BooleanOptionalAction,
    )

    artifacts_group.add_argument(
        "--miopenprovider",
        default=False,
        help="Include 'miopenprovider' artifacts",
        action=argparse.BooleanOptionalAction,
    )

    artifacts_group.add_argument(
        "--hipkernelprovider",
        default=False,
        help="Include 'hipkernelprovider' artifacts",
        action=argparse.BooleanOptionalAction,
    )

    artifacts_group.add_argument(
        "--hiptensor",
        default=False,
        help="Include 'hiptensor' artifacts",
        action=argparse.BooleanOptionalAction,
    )

    artifacts_group.add_argument(
        "--rocdecode",
        default=False,
        help="Include 'rocdecode' artifacts",
        action=argparse.BooleanOptionalAction,
    )

    artifacts_group.add_argument(
        "--rocjpeg",
        default=False,
        help="Include 'rocjpeg' artifacts",
        action=argparse.BooleanOptionalAction,
    )

    artifacts_group.add_argument(
        "--rocjitsu",
        default=False,
        help="Include 'rocjitsu' artifacts",
        action=argparse.BooleanOptionalAction,
    )

    artifacts_group.add_argument(
        "--mirage",
        default=False,
        help="Include 'mirage' artifacts",
        action=argparse.BooleanOptionalAction,
    )

    artifacts_group.add_argument(
        "--hipblasltprovider",
        default=False,
        help="Include 'hipblasltprovider' artifacts",
        action=argparse.BooleanOptionalAction,
    )

    artifacts_group.add_argument(
        "--prim",
        default=False,
        help="Include 'prim' artifacts",
        action=argparse.BooleanOptionalAction,
    )

    artifacts_group.add_argument(
        "--rand",
        default=False,
        help="Include 'rand' artifacts",
        action=argparse.BooleanOptionalAction,
    )

    artifacts_group.add_argument(
        "--rccl",
        default=False,
        help="Include 'rccl' artifacts",
        action=argparse.BooleanOptionalAction,
    )

    artifacts_group.add_argument(
        "--mpi",
        default=False,
        help="Include OpenMPI (vendored by TheRock build)",
        action=argparse.BooleanOptionalAction,
    )

    artifacts_group.add_argument(
        "--rocprofiler-compute",
        default=False,
        help="Include 'rocprofiler-compute' artifacts",
        action=argparse.BooleanOptionalAction,
    )

    artifacts_group.add_argument(
        "--rocprofiler-sdk",
        default=False,
        help="Include 'rocprofiler-sdk' artifacts",
        action=argparse.BooleanOptionalAction,
    )

    artifacts_group.add_argument(
        "--rocprofiler-systems",
        default=False,
        help="Include 'rocprofiler-systems' artifacts",
        action=argparse.BooleanOptionalAction,
    )

    artifacts_group.add_argument(
        "--rocprofiler-systems-examples",
        default=False,
        help="Include 'rocprofiler-systems-examples' artifacts",
        action=argparse.BooleanOptionalAction,
    )

    artifacts_group.add_argument(
        "--rocrtst",
        default=False,
        help="Include 'rocrtst' artifacts",
        action=argparse.BooleanOptionalAction,
    )

    artifacts_group.add_argument(
        "--rocalution",
        default=False,
        help="Include 'rocalution' artifacts",
        action=argparse.BooleanOptionalAction,
    )

    artifacts_group.add_argument(
        "--rocwmma",
        default=False,
        help="Include 'rocwmma' artifacts",
        action=argparse.BooleanOptionalAction,
    )

    artifacts_group.add_argument(
        "--libhipcxx",
        default=False,
        help="Include 'libhipcxx' artifacts",
        action=argparse.BooleanOptionalAction,
    )

    artifacts_group.add_argument(
        "--tests",
        default=False,
        help="Include all test artifacts for enabled libraries",
        action=argparse.BooleanOptionalAction,
    )

    artifacts_group.add_argument(
        "--base-only", help="Include only base artifacts", action="store_true"
    )

    group.add_argument(
        "--input-dir",
        type=str,
        help="Pass in an existing directory of TheRock to provision and test",
    )

    parser.add_argument(
        "--amdgpu-targets",
        type=str,
        default="",
        help="Comma-separated individual GPU targets for fetching split artifacts (e.g. 'gfx942')",
    )

    parser.add_argument(
        "--run-github-repo",
        type=str,
        help="GitHub repository for --run-id in 'owner/repo' format (e.g. 'ROCm/TheRock'). Defaults to GITHUB_REPOSITORY env var or 'ROCm/TheRock'",
    )

    args = parser.parse_args(argv)

    if not args.artifact_group:
        raise argparse.ArgumentTypeError(
            "Either --amdgpu-family or --artifact-group must be specified"
        )

    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
