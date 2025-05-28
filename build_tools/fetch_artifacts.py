#!/usr/bin/env python
# This script provides a somewhat dynamic way to
# retrieve artifacts from s3

# NOTE: This script currently only retrieves the requested artifacts,
# but those artifacts may not have all required dependencies.

import argparse
import os
import platform
import shutil
import subprocess
import sys

GENERIC_VARIANT = "generic"
PLATFORM = platform.system().lower()

REPO = os.getenv("REPO", "ROCm/TheRock")
OWNER, REPO_NAME = REPO.split("/")
EXTERNAL_REPO = "" if REPO_NAME == "TheRock" and OWNER == "ROCm" else f"{OWNER}-{REPO}/"
BUCKET = "therock-artifacts" if REPO_NAME == "TheRock" and OWNER == "ROCm" else 'therock-artifacts-external'


def log(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


def check_aws_cli_available():
    if not shutil.which("aws"):
        log("[ERROR] AWS CLI not found in PATH.")
        sys.exit(1)


def s3_bucket_exists(run_id):
    check_aws_cli_available()
    cmd = [
        "aws",
        "s3",
        "ls",
        f"s3://{BUCKET}/{EXTERNAL_REPO}{run_id}-{PLATFORM}",
        "--no-sign-request",
    ]
    process = subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL)
    return process.returncode == 0


def s3_exec(variant, package, run_id, build_dir):
    check_aws_cli_available()
    cmd = [
        "aws",
        "s3",
        "cp",
        f"s3://{BUCKET}/{EXTERNAL_REPO}{run_id}-{PLATFORM}/{package}_{variant}.tar.xz",
        str(build_dir),
        "--no-sign-request",
    ]
    log(f"++ Exec [{cmd}]")
    try:
        subprocess.run(cmd, check=True)
    except Exception as ex:
        log(f"Exception when executing [{cmd}]")
        log(str(ex))


def retrieve_base_artifacts(args, run_id, build_dir):
    base_artifacts = [
        "core-runtime_run",
        "core-runtime_lib",
        "sysdeps_lib",
        "base_lib",
        "amd-llvm_run",
        "amd-llvm_lib",
        "core-hip_lib",
        "core-hip_dev",
        "rocprofiler-sdk_lib",
        "host-suite-sparse_lib",
    ]
    if args.blas:
        base_artifacts.append("host-blas_lib")

    for base_artifact in base_artifacts:
        s3_exec(GENERIC_VARIANT, base_artifact, run_id, build_dir)


def retrieve_enabled_artifacts(args, target, run_id, build_dir):
    artifact_paths = []
    all_artifacts = ["blas", "fft", "miopen", "prim", "rand"]
    # RCCL is disabled for Windows
    if PLATFORM != "windows":
        all_artifacts.append("rccl")

    if args.blas:
        artifact_paths.append("blas")
    if args.fft:
        artifact_paths.append("fft")
    if args.miopen:
        artifact_paths.append("miopen")
    if args.prim:
        artifact_paths.append("prim")
    if args.rand:
        artifact_paths.append("rand")
    if args.rccl and PLATFORM != "windows":
        artifact_paths.append("rccl")

    enabled_artifacts = []

    # In the case that no library arguments were passed and base_only args is false, we install all artifacts
    if not artifact_paths and not args.base_only:
        artifact_paths = all_artifacts

    for base_path in artifact_paths:
        enabled_artifacts.append(f"{base_path}_lib")
        if args.tests:
            enabled_artifacts.append(f"{base_path}_test")

    for enabled_artifact in enabled_artifacts:
        s3_exec(f"{target}", enabled_artifact, run_id, build_dir)


def run(args):
    run_id = args.run_id
    target = args.target
    build_dir = args.build_dir
    if not s3_bucket_exists(run_id):
        print(f"S3 artifacts for {run_id} does not exist. Exiting...")
        return
    retrieve_base_artifacts(args, run_id, build_dir)
    if not args.base_only:
        retrieve_enabled_artifacts(args, target, run_id, build_dir)


def main(argv):
    parser = argparse.ArgumentParser(prog="fetch_artifacts")
    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="GitHub run ID to retrieve artifacts from",
    )

    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Target variant for specific GPU target",
    )

    parser.add_argument(
        "--build-dir",
        type=str,
        default="build/artifacts",
        help="Path to the artifact build directory",
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

    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
