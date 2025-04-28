#!/usr/bin/env python
# This script provides a somewhat dynamic way to
# retrieve artifacts from s3

# NOTE: This script currently only retrieves the requested artifacts,
# but those artifacts may not have all required dependencies.

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
    if args.blas:
        base_artifacts.append("host-blas_lib")

    for base_artifact in base_artifacts:
        s3_exec(GENERIC_VARIANT, base_artifact, run_id, build_dir)


def retrieve_enabled_artifacts(args, target, run_id, build_dir):
    base_artifact_path = []
    if args.blas:
        base_artifact_path.append("blas")
    if args.fft:
        base_artifact_path.append("fft")
    if args.miopen:
        base_artifact_path.append("miopen")
    if args.prim:
        base_artifact_path.append("prim")
    if args.rand:
        base_artifact_path.append("rand")
    if args.rccl:
        base_artifact_path.append("rccl")

    enabled_artifacts = []
    for base_path in base_artifact_path:
        enabled_artifacts.append(f"{base_path}_lib")
        if args.test:
            enabled_artifacts.append(f"{base_path}_test")

    for enabled_artifact in enabled_artifacts:
        s3_exec(f"{target}", enabled_artifact, run_id, build_dir)
