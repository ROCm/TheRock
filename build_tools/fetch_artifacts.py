#!/usr/bin/env python
# This script provides a somewhat dynamic way to
# retrieve artifacts from s3

# NOTE: This script currently only retrieves the requested artifacts,
# but those artifacts may not have all required dependencies.

import argparse
import subprocess
import sys

GENERIC_VARIANT = "generic"


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


def s3_exec(variant, package, run_id, build_dir):
    cmd = [
        "aws",
        "s3",
        "cp",
        f"s3://therock-artifacts/{run_id}/{package}_{variant}.tar.xz",
        build_dir,
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
    if args.all or args.blas:
        base_artifacts.append("host-blas_lib")

    for base_artifact in base_artifacts:
        s3_exec(GENERIC_VARIANT, base_artifact, run_id, build_dir)


def retrieve_enabled_artifacts(args, target, run_id, build_dir):
    base_artifact_path = []
    if args.all or args.blas:
        base_artifact_path.append("blas")
    if args.all or args.fft:
        base_artifact_path.append("fft")
    if args.all or args.miopen:
        base_artifact_path.append("miopen")
    if args.all or args.prim:
        base_artifact_path.append("prim")
    if args.all or args.rand:
        base_artifact_path.append("rand")
    if args.all or args.rccl:
        base_artifact_path.append("rccl")

    enabled_artifacts = []
    for base_path in base_artifact_path:
        enabled_artifacts.append(f"{base_path}_lib")
        if args.test:
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
        "--test",
        default=False,
        help="Include all test artifacts for enabled libraries",
        action=argparse.BooleanOptionalAction,
    )

    artifacts_group.add_argument(
        "--all", help="Include all artifacts", action="store_true"
    )

    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
