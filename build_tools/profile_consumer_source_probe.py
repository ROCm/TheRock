#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Actual consumer source-only CI probe; synthetic URLs never fetch SDKs."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

from normal_ci_input_identity import capture_source
from normal_ci_validation import GITLINKS

BRANCH = "users/sa-faizal/build-profile-41147801272-normal-consumer-probe"
WHEEL_IMAGES = {
    "wheel-ubuntu2404": "ghcr.io/rocm/no_rocm_image_ubuntu24_04@sha256:a13e8370a2d1d33a910e47669cdf075dbe9554685e78592f22935ec7c7c35e39",
    "wheel-ubi10": "ghcr.io/rocm/no_rocm_image_ubi10@sha256:d2b4f153ec2ea8176e86db25d8b0714137fd5429922278eec422a41d6084f36b",
}
INSTALL_IMAGES = {
    "rpm-rhel8": "registry.access.redhat.com/ubi8/ubi@sha256:c322aa68f2d630b36b9faab64094357f62720be2f56f7099376ca2d6c4fe389b",
    "rpm-rhel10": "registry.access.redhat.com/ubi10/ubi@sha256:15f0a6e2b448ac7aa24425c9ab541e6402dc32b782844a285ccaeffb1c938fe1",
    "rpm-sles16": "registry.suse.com/bci/bci-base@sha256:38bb07164a0b64c429a285e877277d618fa7531eccd0d31d34375b9ea247c843",
}


def validate(env, event, expected_sha):
    expected = dict(
        GITHUB_ACTIONS="true",
        GITHUB_REPOSITORY="ROCm/TheRock",
        GITHUB_ACTOR="sa-faizal",
        GITHUB_EVENT_NAME="workflow_dispatch",
        GITHUB_REF="refs/heads/" + BRANCH,
        GITHUB_WORKFLOW_REF="ROCm/TheRock/.github/workflows/multi_arch_ci.yml@refs/heads/"
        + BRANCH,
    )
    if any(env.get(k) != v for k, v in expected.items()):
        raise ValueError("Probe repository/actor/event/ref/workflow context mismatch")
    if (
        not re.fullmatch(r"[0-9a-f]{40}", expected_sha)
        or env.get("GITHUB_SHA") != expected_sha
        or env.get("GITHUB_WORKFLOW_SHA") != expected_sha
    ):
        raise ValueError("Explicit frozen probe SHA must match triggering CI SHA")
    inputs = event.get("inputs", {})
    if (
        str(inputs.get("profile_consumer_source_probe", "false")).lower() != "true"
        or inputs.get("profile_consumer_expected_sha") != expected_sha
    ):
        raise ValueError("Explicit probe mode and frozen SHA are required")
    for key in ("profile_normal_validation", "profile_native_fixture_only"):
        if str(inputs.get(key, "false")).lower() != "false":
            raise ValueError("Diagnostic and normal modes cannot be combined")
    for key, default in {
        "profile_normal_case": "baseline",
        "profile_normal_product": "none",
        "profile_normal_cache_regime": "ci-readonly",
    }.items():
        if inputs.get(key, default) != default:
            raise ValueError("Normal comparison options are not probe options")
    for key in (
        "linux_amdgpu_families",
        "windows_amdgpu_families",
        "build_stages",
        "prebuilt_stages",
        "baseline_run_id",
        "changed_projects",
        "pr_labels",
    ):
        if inputs.get(key, ""):
            raise ValueError(
                "Build/reuse/test selection is forbidden for source-only probe"
            )
    if (
        not env.get("GITHUB_RUN_ID", "").isdecimal()
        or not env.get("GITHUB_RUN_ATTEMPT", "").isdecimal()
    ):
        raise ValueError("Actual CI run/attempt identity required")


def digest(path):
    hashed = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hashed.update(chunk)
    return hashed.hexdigest()


def run_probe(workspace, context, env):
    images = {**WHEEL_IMAGES, **INSTALL_IMAGES}
    if context not in images or env.get("NORMAL_IMAGE_PIN") != images[context]:
        raise ValueError("Actual context must use its unchanged pinned image")
    job = (
        "test_wheels"
        if context in WHEEL_IMAGES
        else "test_native_linux_packages_install"
    )
    if env.get("GITHUB_JOB") != job:
        raise ValueError("Actual consumer job ID required")
    if context in WHEEL_IMAGES and sys.version_info[:2] != (3, 12):
        raise ValueError("Wheel source probe requires actual Python3.12")
    start_wall, start_cpu = time.monotonic(), time.process_time()
    source = capture_source(workspace, GITLINKS, digest, env)
    cost = dict(
        wall_s=time.monotonic() - start_wall,
        own_process_cpu_s=time.process_time() - start_cpu,
        scope="Source capture only; own CPU excludes child Git processes; no performance claim",
    )
    git = shutil.which("git")
    mode_exercised = (
        (bool(git) and source["mode"] == "measured-git")
        if context in WHEEL_IMAGES
        else (
            git is None
            and source["mode"] == "git-unavailable; exact-commit archive comparison"
        )
    )
    version = (
        subprocess.run(
            [git, "--version"], capture_output=True, text=True, check=True
        ).stdout.strip()
        if git
        else None
    )
    return dict(
        validated=mode_exercised,
        expected_source_mode_exercised=mode_exercised,
        expected_git_available=context in WHEEL_IMAGES,
        capture_cost=cost,
        source_only=True,
        producer_refs_synthetic=True,
        real_producer_sdk_output_identity_proven=False,
        scope="Source-only fixture; synthetic current-run producer URLs are neither published nor fetched; no real producer/SDK/output or test/performance proof",
        source=source,
        context=context,
        uid=os.geteuid(),
        gid=os.getegid(),
        python=dict(
            executable=sys.executable,
            version=sys.version,
            prefix=sys.prefix,
            base_prefix=sys.base_prefix,
        ),
        git=dict(
            available=bool(git),
            path=git,
            version=version,
            sha256=digest(Path(git)) if git else None,
        ),
        ci={
            k: env.get(k)
            for k in [
                "GITHUB_REPOSITORY",
                "GITHUB_SHA",
                "GITHUB_WORKFLOW_SHA",
                "GITHUB_WORKFLOW_REF",
                "GITHUB_ACTOR",
                "GITHUB_RUN_ID",
                "GITHUB_RUN_ATTEMPT",
                "GITHUB_JOB",
                "RUNNER_NAME",
                "NORMAL_IMAGE_PIN",
                "OS_PROFILE",
                "TEST_TYPE",
                "REPO_URL",
                "PACKAGE_FIND_LINKS_URL",
            ]
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["guard", "run"])
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--context", choices=[*WHEEL_IMAGES, *INSTALL_IMAGES])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = dict(
        validated=False,
        source_only=True,
        producer_refs_synthetic=True,
        uid=os.geteuid() if hasattr(os, "geteuid") else None,
        git_available=bool(shutil.which("git")),
        git_path=shutil.which("git"),
        python_version=sys.version,
    )
    try:
        event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
        validate(os.environ, event, args.expected_sha)
        if args.command == "run":
            workspace = Path(os.environ["GITHUB_WORKSPACE"]).resolve(strict=True)
            if Path(__file__).resolve().parents[1] != workspace:
                raise ValueError("Probe must execute from verified triggering checkout")
            result = run_probe(workspace, args.context, os.environ)
            if not result["expected_source_mode_exercised"]:
                raise ValueError(
                    "Requested actual Git-present wheel or Gitless RPM source mode was not exercised"
                )
        else:
            result.update(
                validated=True,
                expected_sha=args.expected_sha,
                scope="Guard only; no SDK/build/test",
            )
    except Exception as error:
        result.update(error_type=type(error).__name__, error_message=str(error)[:4096])
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
