#!/usr/bin/env python
"""install_rocm_from_artifacts.py

This script helps CI workflows, developers and testing suites easily install
TheRock to their environment using artifacts. It installs TheRock to an output
directory from one of these sources:

  - GitHub CI workflow run
  - Release tag
  - An existing installation of TheRock

Usage:
python build_tools/install_rocm_from_artifacts.py
    [--output-dir OUTPUT_DIR]
    [--amdgpu-family AMDGPU_FAMILY]
    (--run-id RUN_ID | --release RELEASE | --input-dir INPUT_DIR)
    [--blas | --no-blas] [--fft | --no-fft] [--hipdnn | --no-hipdnn] [--miopen | --no-miopen] [--prim | --no-prim]
    [--rand | --no-rand] [--rccl | --no-rccl] [--tests | --no-tests] [--base-only]

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
  release tag `nightly-tarball` to the specified output directory `build`:
    ```
    python build_tools/install_rocm_from_artifacts.py \
        --release 6.4.0rc20250416 \
        --amdgpu-family gfx110X-dgpu \
        --output-dir build
    ```
- Downloads and unpacks the version `6.4.0.dev0+8f6cdfc0d95845f4ca5a46de59d58894972a29a9`
  gfx120X artifacts from release tag `dev-tarball` to the default output directory `therock-build`:
    ```
    python build_tools/install_rocm_from_artifacts.py \
        --release 6.4.0.dev0+8f6cdfc0d95845f4ca5a46de59d58894972a29a9 \
        --amdgpu-family gfx120X-all
    ```

You can select your AMD GPU family from therock_amdgpu_targets.cmake.

By default for CI workflow retrieval, all artifacts (excluding test artifacts)
will be downloaded. For specific artifacts, pass in the flag such as `--rand`
(RAND artifacts) For test artifacts, pass in the flag `--tests` (test artifacts).
For base artifacts only, pass in the flag `--base-only`

Note: the script will overwrite the output directory argument. If no argument
is passed, it will overwrite the default "therock-build" directory.
"""

import argparse
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from fetch_artifacts import main as fetch_artifacts_main
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tarfile

PLATFORM = platform.system().lower()
s3_client = boto3.client(
    "s3",
    verify=False,
    config=Config(max_pool_connections=100, signature_version=UNSIGNED),
)


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


def _retrieve_s3_release_assets(
    release_bucket, amdgpu_family, release_version, output_dir
):
    """
    Makes an API call to retrieve the release's assets, then retrieves the asset matching the amdgpu family
    """
    asset_name = f"therock-dist-{PLATFORM}-{amdgpu_family}-{release_version}.tar.gz"
    destination = output_dir / asset_name

    with open(destination, "wb") as f:
        s3_client.download_fileobj(release_bucket, asset_name, f)

    # After downloading the asset, untar-ing the file
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
        "--target",
        args.amdgpu_family,
        "--output-dir",
        str(args.output_dir),
        "--flatten",
    ]

    # These artifacts are the "base" requirements for running tests.
    base_artifact_patterns = [
        "core-runtime_run",
        "core-runtime_lib",
        "sysdeps_lib",
        "base_run",
        "base_lib",
        "amd-llvm_run",
        "amd-llvm_lib",
        "core-hip_lib",
        "core-hip_dev",
        "rocprofiler-sdk_lib",
        "host-suite-sparse_lib",
    ]

    if args.base_only:
        argv.extend(base_artifact_patterns)
    elif any(
        [args.blas, args.fft, args.hipdnn, args.miopen, args.prim, args.rand, args.rccl]
    ):
        argv.extend(base_artifact_patterns)

        extra_artifacts = []
        if args.blas:
            extra_artifacts.append("blas")
        if args.fft:
            extra_artifacts.append("fft")
        if args.hipdnn:
            extra_artifacts.append("hipdnn")
        if args.miopen:
            extra_artifacts.append("miopen")
        if args.prim:
            extra_artifacts.append("prim")
        if args.rand:
            extra_artifacts.append("rand")
        if args.rccl:
            extra_artifacts.append("rccl")

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
    amdgpu_family = args.amdgpu_family
    # Determine if version is nightly-tarball or dev-tarball
    nightly_regex_expression = (
        "(\\d+\\.)?(\\d+\\.)?(\\*|\\d+)rc(\\d{4})(\\d{2})(\\d{2})"
    )
    dev_regex_expression = "(\\d+\\.)?(\\d+\\.)?(\\*|\\d+).dev0+"
    nightly_release = re.search(nightly_regex_expression, args.release) != None
    dev_release = re.search(dev_regex_expression, args.release) != None
    if not nightly_release and not dev_release:
        log("This script requires a nightly-tarball or dev-tarball version.")
        log("Please retrieve the correct release version from:")
        log(
            "\t - https://therock-nightly-tarball.s3.amazonaws.com/ (nightly-tarball example: 6.4.0rc20250416)"
        )
        log(
            "\t - https://therock-dev-tarball.s3.amazonaws.com/ (dev-tarball example: 6.4.0.dev0+8f6cdfc0d95845f4ca5a46de59d58894972a29a9)"
        )
        log("Exiting...")
        return

    release_bucket = (
        "therock-nightly-tarball" if nightly_release else "therock-dev-tarball"
    )
    release_version = args.release

    log(f"Retrieving artifacts from release bucket {release_bucket}")
    _retrieve_s3_release_assets(
        release_bucket, amdgpu_family, release_version, output_dir
    )


def retrieve_artifacts_by_input_dir(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    log(f"Retrieving artifacts from input dir {input_dir}")

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


def run(args):
    log("### Installing TheRock using artifacts ###")
    _create_output_directory(args.output_dir)
    if args.run_id:
        retrieve_artifacts_by_run_id(args)
    elif args.release:
        retrieve_artifacts_by_release(args)

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

    parser.add_argument(
        "--amdgpu-family",
        type=str,
        required=True,
        default="gfx94X-dcgpu",
        help="AMD GPU family to install (please refer to this: https://github.com/ROCm/TheRock/blob/59c324a759e8ccdfe5a56e0ebe72a13ffbc04c1f/cmake/therock_amdgpu_targets.cmake#L44-L81 for family choices)",
    )

    # This mutually exclusive group will ensure that only one argument is present
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-id", type=str, help="GitHub run ID of TheRock to install")

    group.add_argument(
        "--release",
        type=str,
        help="Release version of TheRock to install, from the nightly-tarball (X.Y.ZrcYYYYMMDD) or dev-tarball (X.Y.Z.dev0+{hash})",
    )

    artifacts_group = parser.add_argument_group("artifacts_group")
    artifacts_group.add_argument(
        "--blas",
        default=False,
        help="Include 'blas' artifacts",
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
        "--miopen",
        default=False,
        help="Include 'miopen' artifacts",
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

    args = parser.parse_args(argv)

    if not args.amdgpu_family:
        raise argparse.ArgumentTypeError(
            "AMD GPU family argument is required and cannot be empty"
        )

    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
