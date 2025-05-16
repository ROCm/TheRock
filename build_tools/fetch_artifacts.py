#!/usr/bin/env python
# This script provides a somewhat dynamic way to
# retrieve artifacts from s3

# NOTE: This script currently only retrieves the requested artifacts,
# but those artifacts may not have all required dependencies.

import argparse
import concurrent.futures
import platform
import shlex
import subprocess
import sys


class ArtifactNotFoundException(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


GENERIC_VARIANT = "generic"
PLATFORM = platform.system().lower()

# TODO(geomin12): switch out logging library
def log(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


def s3_bucket_exists(run_id):
    """Checks that the AWS S3 bucket exists."""
    cmd = [
        "aws",
        "s3",
        "ls",
        f"s3://therock-artifacts/{run_id}-{PLATFORM}",
        "--no-sign-request",
    ]
    process = subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL)
    return process.returncode == 0


def s3_commands_for_artifacts(artifacts, run_id, build_dir, variant):
    """Collects AWS S3 copy commands to execute later in parallel."""
    cmds = []
    for artifact in artifacts:
        cmds.append(
            [
                "aws",
                "s3",
                "cp",
                f"s3://therock-artifacts/{run_id}/{artifact}_{variant}.tar.xz",
                build_dir,
                "--no-sign-request",
            ]
        )
    return cmds


def subprocess_run(cmd):
    """Runs a command via subprocess run."""
    try:
        log(f"++ Exec '{shlex.join(cmd)}'")
        process = subprocess.run(cmd, capture_output=True)
        if process.returncode == 1:
            # Fetching is done at a best effort. If an artifact is not found, it doesn't trigger any errors
            if "(404)" in str(process.stderr):
                raise ArtifactNotFoundException(
                    f"Artifact not found for '{shlex.join(cmd)}'"
                )
            else:
                raise Exception(process.stderr)
        log(f"++ Exec complete '{shlex.join(cmd)}'")
    except ArtifactNotFoundException as ex:
        log(str(ex))
    except Exception as ex:
        log(str(ex))
        sys.exit(1)


def parallel_exec_commands(cmds):
    """Runs parallelized subprocess commands using a thread pool executor"""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(subprocess_run, cmd) for cmd in cmds]
        for future in concurrent.futures.as_completed(futures):
            future.result(timeout=60)


def retrieve_base_artifacts(args, run_id, build_dir):
    """Retrieves TheRock base artifacts using AWS S3 copy."""
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

    cmds = s3_commands_for_artifacts(base_artifacts, run_id, build_dir, GENERIC_VARIANT)
    parallel_exec_commands(cmds)


def retrieve_enabled_artifacts(args, target, run_id, build_dir):
    """Retrieves TheRock artifacts using AWS S3 copy, based on the enabled arguments.

    If no artifacts have been collected, we assume that we want to install all artifacts
    If `args.tests` have been enabled, we also collect test artifacts
    """
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

    cmds = s3_commands_for_artifacts(enabled_artifacts, run_id, build_dir, target)
    parallel_exec_commands(cmds)


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
