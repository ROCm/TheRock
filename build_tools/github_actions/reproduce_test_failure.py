#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Reproduces a test failure from CI.

Usage:
    # Full reproduction (runs test)
    python reproduce_test_failure.py --run-id 12345678 --repository ROCm/TheRock \
        --amdgpu-family gfx942 --test-script "python test.py"

    # Setup only (drops into shell)
    python reproduce_test_failure.py --run-id 12345678 --repository ROCm/TheRock \
        --amdgpu-family gfx942 --test-script "python test.py" --setup-only
"""

import argparse
import shutil
import subprocess
import sys

DEFAULT_CONTAINER_IMAGE = "ghcr.io/rocm/no_rocm_image_ubuntu24_04:latest"


def check_docker() -> bool:
    if not shutil.which("docker"):
        return False
    result = subprocess.run(["docker", "info"], capture_output=True)
    return result.returncode == 0


def build_reproduction_command(args: argparse.Namespace) -> str:
    """Build the command string for reproduction."""
    cmd = (
        f"python build_tools/github_actions/reproduce_test_failure.py "
        f"--run-id {args.run_id} "
        f"--repository {args.repository} "
        f"--amdgpu-family {args.amdgpu_family} "
        f'--test-script "{args.test_script}"'
    )
    if args.shard_index != "1":
        cmd += f" --shard-index {args.shard_index}"
    if args.total_shards != "1":
        cmd += f" --total-shards {args.total_shards}"
    if args.test_type != "full":
        cmd += f" --test-type {args.test_type}"
    if args.fetch_artifact_args:
        cmd += f' --fetch-artifact-args="{args.fetch_artifact_args}"'
    return cmd


def run_reproduction(args: argparse.Namespace) -> int:
    if not check_docker():
        print("ERROR: Docker is not available. Install Docker and try again.")
        return 1

    fetch_cmd = (
        f"GITHUB_REPOSITORY={args.repository} "
        f"python build_tools/install_rocm_from_artifacts.py "
        f"--run-id {args.run_id} "
        f"--amdgpu-family {args.amdgpu_family}"
    )
    if args.fetch_artifact_args:
        fetch_cmd += f" {args.fetch_artifact_args}"

    total_steps = 7 if args.setup_only else 8
    step = 0

    def add_step(desc: str, cmd: str) -> list[str]:
        nonlocal step
        step += 1
        return [f"echo '[{step}/{total_steps}] {desc}'", cmd]

    lines = ["set -e"]
    lines += add_step("Installing uv", "curl -LsSf https://astral.sh/uv/install.sh | bash && source $HOME/.local/bin/env")
    lines += add_step("Cloning TheRock", "git clone https://github.com/ROCm/TheRock.git && cd TheRock")
    lines += add_step("Creating virtual environment", "uv venv .venv && source .venv/bin/activate")
    lines += add_step("Installing dependencies", "uv pip install -r requirements-test.txt")
    lines += add_step("Downloading artifacts", fetch_cmd)
    lines += add_step("Setting environment variables", "\n".join([
        "export THEROCK_BIN_DIR=./therock-build/bin",
        "export OUTPUT_ARTIFACTS_DIR=./therock-build",
        f"export SHARD_INDEX={args.shard_index}",
        f"export TOTAL_SHARDS={args.total_shards}",
        f"export TEST_TYPE={args.test_type}",
    ]))

    if args.setup_only:
        lines += add_step("Setup complete", f"echo 'Run: {args.test_script}' && exec /bin/bash")
    else:
        lines += add_step("Running test", args.test_script)

    cmd = [
        "docker", "run", "--rm", "-it",
        "--ipc", "host",
        "--group-add", "video",
        "--device", "/dev/kfd",
        "--device", "/dev/dri",
        args.container_image,
        "/bin/bash", "-c", "\n".join(lines),
    ]

    try:
        return subprocess.run(cmd).returncode
    except KeyboardInterrupt:
        return 130


def main() -> int:
    parser = argparse.ArgumentParser(description="Reproduce a test failure from CI")
    parser.add_argument("--run-id", required=True, help="GitHub Actions run ID")
    parser.add_argument("--repository", required=True, help="GitHub repository")
    parser.add_argument("--amdgpu-family", required=True, help="AMDGPU family")
    parser.add_argument("--test-script", required=True, help="Test script to run")
    parser.add_argument("--shard-index", default="1", help="Shard index")
    parser.add_argument("--total-shards", default="1", help="Total shards")
    parser.add_argument("--test-type", default="full", help="Test type")
    parser.add_argument("--container-image", default=DEFAULT_CONTAINER_IMAGE)
    parser.add_argument("--fetch-artifact-args", default="", help="Extra artifact args")
    parser.add_argument("--setup-only", action="store_true", help="Setup only, don't run test")
    parser.add_argument("--print-cmd", action="store_true", help="Print reproduction command")

    args = parser.parse_args()

    if args.print_cmd:
        print("To reproduce this failure, run:")
        print(f"  {build_reproduction_command(args)}")
        return 0

    return run_reproduction(args)


if __name__ == "__main__":
    sys.exit(main())
