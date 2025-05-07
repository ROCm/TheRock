#!/usr/bin/env python
# This script provides a somewhat dynamic way to
# retrieve artifacts from s3

# NOTE: This script currently only retrieves the requested artifacts,
# but those artifacts may not have all required dependencies.

import argparse
import platform
import subprocess
import sys

GENERIC_VARIANT = "generic"
PLATFORM = platform.system()


def log(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


def s3_bucket_exists(run_id):
    cmd = [
        "aws",
        "s3",
        "ls",
        f"s3://therock-artifacts/{run_id}",
        "--no-sign-request",
    ]
    process = subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL)
    return process.returncode == 0


def s3_cmd(variant, package, run_id, build_dir):
    cmd = [
        "aws",
        "s3",
        "cp",
        f"s3://therock-artifacts/{run_id}/{package}_{variant}.tar.xz",
        build_dir,
        "--no-sign-request",
    ]
    log(f"++ Exec [{cmd}]")
    return " ".join(cmd)


def s3_parallel_exec(cmds):
    procs = [subprocess.Popen(cmd, check=False) for cmd in cmds]
    for p in procs:
        p.wait()


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
    
    cmds = []
    for base_artifact in base_artifacts:
        cmds.append(s3_cmd(GENERIC_VARIANT, base_artifact, run_id, build_dir))
    s3_parallel_exec(cmds)
    


def retrieve_enabled_artifacts(args, target, run_id, build_dir):
    artifact_paths = []
    all_artifacts = ["blas", "fft", "miopen", "prim", "rand"]
    # RCCL is disabled for Windows
    if PLATFORM != "Windows":
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
    if args.rccl and PLATFORM != "Windows":
        artifact_paths.append("rccl")

    enabled_artifacts = []

    # In the case that no library arguments were passed and base_only args is false, we install all artifacts
    if not artifact_paths and not args.base_only:
        artifact_paths = all_artifacts

    for base_path in artifact_paths:
        enabled_artifacts.append(f"{base_path}_lib")
        if args.tests:
            enabled_artifacts.append(f"{base_path}_test")

    cmds = []
    for enabled_artifact in enabled_artifacts:
        cmds.append(s3_cmd(f"{target}", enabled_artifact, run_id, build_dir))
    s3_parallel_exec(cmds)


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
