#!/usr/bin/env python
# This script provides a somewhat dynamic way to
# retrieve artifacts from s3

import argparse
import subprocess
import sys

GENERIC_VARIANT = "generic"


def log(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


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
    subprocess.run(cmd)


def retrieve_base_artifacts(run_id, build_dir):
    base_artifacts = [
        "core-runtime_run",
        "core-runtime_lib",
        "sysdeps_lib",
        "base_lib",
        "amd-llvm_lib",
        "amd-llvm_run",
        "core-hip_lib",
        "host-blas_lib",
        "rocprofiler-sdk_lib",
        "host-suite-sparse_lib"
    ]
    for base_artifact in base_artifacts:
        s3_exec(GENERIC_VARIANT, base_artifact, run_id, build_dir)


def retrieve_enabled_artifacts(args, test_enabled):
    base_artifact_path = []
    if args.blas or args.all:
        base_artifact_path.append("blas")
    if args.fft or args.all:
        base_artifact_path.append("fft")
    if args.miopen or args.all:
        base_artifact_path.append("miopen")
    if args.prim or args.all:
        base_artifact_path.append("prim")
    if args.rand or args.all:
        base_artifact_path.append("rand")
    if args.rccl or args.all:
        base_artifact_path.append("rccl")

    enabled_artifacts = []
    for base_path in base_artifact_path:
        enabled_artifacts.append(f"{base_path}_lib")
        if test_enabled:
            enabled_artifacts.append(f"{base_path}_test")

    return enabled_artifacts


def run(args):
    run_id = args.run_id
    target = args.target
    build_dir = args.build_dir
    test_enabled = args.test
    retrieve_base_artifacts(run_id, build_dir)
    enabled_artifacts = retrieve_enabled_artifacts(args, test_enabled)

    for enabled_artifact in enabled_artifacts:
        s3_exec(f"{target}", enabled_artifact, run_id, build_dir)


def main(argv):
    parser = argparse.ArgumentParser(prog="fetch_artifacts")
    parser.add_argument(
        "--run-id", type=str, help="GitHub run ID to retrieve artifacts from"
    )

    parser.add_argument(
        "--target", type=str, help="Target variant for specific GPU target"
    )

    parser.add_argument(
        "--build-dir",
        type=str,
        default="build/artifacts",
        help="Path to the artifact build directory",
    )

    parser.add_argument(
        "--test",
        help="If flagged, test artifacts will be retrieved",
        action="store_true",
    )

    parser.add_argument(
        "--blas",
        help="If flagged, blas artifacts will be retrieved",
        action="store_true",
    )

    parser.add_argument(
        "--fft", help="If flagged, fft artifacts will be retrieved", action="store_true"
    )

    parser.add_argument(
        "--miopen",
        help="If flagged, miopen artifacts will be retrieved",
        action="store_true",
    )

    parser.add_argument(
        "--prim",
        help="If flagged, prim artifacts will be retrieved",
        action="store_true",
    )

    parser.add_argument(
        "--rand",
        help="If flagged, rand artifacts will be retrieved",
        action="store_true",
    )

    parser.add_argument(
        "--rccl",
        help="If flagged, rccl artifacts will be retrieved",
        action="store_true",
    )

    parser.add_argument(
        "--all", help="If flagged, all artifacts will be retrieved", action="store_true"
    )

    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
