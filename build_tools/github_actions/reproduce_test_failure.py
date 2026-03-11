#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Reproduces a test failure from CI by setting up the environment and running the test.

This script automates the full reproduction workflow:
1. Starts a Docker container with GPU access
2. Clones TheRock and sets up the Python environment
3. Downloads artifacts from the specified CI run
4. Runs the failing test with the same configuration

Usage:
    # Full automated reproduction (requires Docker)
    python ./build_tools/github_actions/reproduce_test_failure.py \
        --run-id 12345678 \
        --repository ROCm/TheRock \
        --amdgpu-family gfx942 \
        --test-script "python build_tools/github_actions/test_executable_scripts/test_rocblas.py"

    # Setup only - drops into shell before running test (requires Docker)
    python ./build_tools/github_actions/reproduce_test_failure.py \
        --run-id 12345678 \
        --repository ROCm/TheRock \
        --amdgpu-family gfx942 \
        --test-script "python build_tools/github_actions/test_executable_scripts/test_rocblas.py" \
        --setup-only

    # Print manual steps (no Docker required, for CI failure output)
    python ./build_tools/github_actions/reproduce_test_failure.py \
        --run-id 12345678 \
        --repository ROCm/TheRock \
        --amdgpu-family gfx942 \
        --test-script "python build_tools/github_actions/test_executable_scripts/test_rocblas.py" \
        --print-steps
"""

import argparse
import shutil
import subprocess
import sys

DEFAULT_CONTAINER_IMAGE = "ghcr.io/rocm/no_rocm_image_ubuntu24_04:latest"


def check_docker() -> bool:
    """Check if Docker is available and running."""
    if not shutil.which("docker"):
        return False

    result = subprocess.run(
        ["docker", "info"],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def build_docker_command(args: argparse.Namespace, script: str) -> list[str]:
    """Build the docker run command."""
    cmd = [
        "docker",
        "run",
        "--rm",
        "-it",
        "--ipc",
        "host",
        "--group-add",
        "video",
        "--device",
        "/dev/kfd",
        "--device",
        "/dev/dri",
        args.container_image,
        "/bin/bash",
        "-c",
        script,
    ]
    return cmd


def build_setup_script(args: argparse.Namespace, run_test: bool) -> str:
    """Build the shell script that runs inside the container."""
    fetch_cmd = (
        f"GITHUB_REPOSITORY={args.repository} "
        f"python build_tools/install_rocm_from_artifacts.py "
        f"--run-id {args.run_id} "
        f"--amdgpu-family {args.amdgpu_family}"
    )
    if args.fetch_artifact_args:
        fetch_cmd += f" {args.fetch_artifact_args}"

    lines = [
        "set -e",
        "",
        "echo '=== Installing uv ==='",
        "curl -LsSf https://astral.sh/uv/install.sh | bash",
        "source $HOME/.local/bin/env",
        "",
        "echo '=== Cloning TheRock ==='",
        "git clone https://github.com/ROCm/TheRock.git",
        "cd TheRock",
        "",
        "echo '=== Setting up Python environment ==='",
        "uv venv .venv",
        "source .venv/bin/activate",
        "uv pip install -r requirements-test.txt",
        "",
        "echo '=== Installing artifacts from CI run ==='",
        fetch_cmd,
        "",
        "echo '=== Setting environment variables ==='",
        "export THEROCK_BIN_DIR=./therock-build/bin",
        "export OUTPUT_ARTIFACTS_DIR=./therock-build",
        f"export SHARD_INDEX={args.shard_index}",
        f"export TOTAL_SHARDS={args.total_shards}",
        f"export TEST_TYPE={args.test_type}",
        "",
    ]

    if run_test:
        lines.extend(
            [
                "echo '=== Running test ==='",
                args.test_script,
            ]
        )
    else:
        lines.extend(
            [
                "echo ''",
                "echo '=== Setup complete ==='",
                "echo 'Run the test with:'",
                f"echo '  {args.test_script}'",
                "echo ''",
                "exec /bin/bash",
            ]
        )

    return "\n".join(lines)


def print_reproduction_steps(args: argparse.Namespace) -> None:
    """Print manual reproduction steps and automated command."""
    print("=" * 60)
    print("TEST FAILURE - REPRODUCTION STEPS")
    print("=" * 60)
    print()
    print("To reproduce this test failure locally, follow these steps:")
    print()
    print("1. Start the Docker container:")
    print("   docker run -it \\")
    print("       --ipc host \\")
    print("       --group-add video \\")
    print("       --device /dev/kfd \\")
    print("       --device /dev/dri \\")
    print(f"       {args.container_image} /bin/bash")
    print()
    print("2. Inside the container, set up the environment:")
    print(
        "   curl -LsSf https://astral.sh/uv/install.sh | bash && source $HOME/.local/bin/env"
    )
    print("   git clone https://github.com/ROCm/TheRock.git && cd TheRock")
    print("   uv venv .venv && source .venv/bin/activate")
    print("   uv pip install -r requirements-test.txt")
    print()
    print("3. Install artifacts from this CI run:")
    print(
        f"   GITHUB_REPOSITORY={args.repository} python build_tools/install_rocm_from_artifacts.py \\"
    )
    print(f"       --run-id {args.run_id} \\")
    if args.fetch_artifact_args:
        print(f"       --amdgpu-family {args.amdgpu_family} \\")
        print(f"       {args.fetch_artifact_args}")
    else:
        print(f"       --amdgpu-family {args.amdgpu_family}")
    print()
    print("4. Set environment variables and run the test:")
    print("   export THEROCK_BIN_DIR=./therock-build/bin")
    print("   export OUTPUT_ARTIFACTS_DIR=./therock-build")
    print(f"   export SHARD_INDEX={args.shard_index}")
    print(f"   export TOTAL_SHARDS={args.total_shards}")
    print(f"   export TEST_TYPE={args.test_type}")
    print(f"   {args.test_script}")
    print()
    print("-" * 60)
    print("AUTOMATED REPRODUCTION (requires Docker):")
    print("-" * 60)
    print()
    cmd = (
        f"python build_tools/github_actions/reproduce_test_failure.py \\\n"
        f"    --run-id {args.run_id} \\\n"
        f"    --repository {args.repository} \\\n"
        f"    --amdgpu-family {args.amdgpu_family} \\\n"
        f'    --test-script "{args.test_script}" \\\n'
        f"    --shard-index {args.shard_index} \\\n"
        f"    --total-shards {args.total_shards} \\\n"
        f"    --test-type {args.test_type}"
    )
    if args.fetch_artifact_args:
        cmd += f' \\\n    --fetch-artifact-args "{args.fetch_artifact_args}"'
    print(cmd)
    print()
    print("Add --setup-only to drop into a shell after setup (before running the test).")
    print()
    print("For more details, see:")
    print(
        "https://github.com/ROCm/TheRock/blob/main/docs/development/test_environment_reproduction.md"
    )
    print("=" * 60)


def run_reproduction(args: argparse.Namespace, run_test: bool) -> int:
    """Run the reproduction in Docker."""
    if not check_docker():
        print("ERROR: Docker is not available or not running.")
        print()
        print("To use automated reproduction, install Docker and ensure the daemon is running.")
        print("Alternatively, use --print-steps to see manual reproduction instructions.")
        return 1

    mode = "full reproduction" if run_test else "setup only"
    print("=" * 60)
    print(f"REPRODUCING TEST FAILURE ({mode})")
    print("=" * 60)
    print()
    print(f"Run ID: {args.run_id}")
    print(f"Repository: {args.repository}")
    print(f"AMDGPU Family: {args.amdgpu_family}")
    print(f"Test Script: {args.test_script}")
    print(f"Container: {args.container_image}")
    print()
    print("Starting Docker container...")
    print()

    script = build_setup_script(args, run_test=run_test)
    cmd = build_docker_command(args, script)

    try:
        result = subprocess.run(cmd)
        return result.returncode
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reproduce a test failure from CI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full automated reproduction
  python reproduce_test_failure.py --run-id 12345678 --repository ROCm/TheRock \\
      --amdgpu-family gfx942 --test-script "python test.py"

  # Setup only (drops into shell before running test)
  python reproduce_test_failure.py --run-id 12345678 --repository ROCm/TheRock \\
      --amdgpu-family gfx942 --test-script "python test.py" --setup-only

  # Print manual steps (no Docker required)
  python reproduce_test_failure.py --run-id 12345678 --repository ROCm/TheRock \\
      --amdgpu-family gfx942 --test-script "python test.py" --print-steps
""",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="GitHub Actions run ID",
    )
    parser.add_argument(
        "--repository",
        type=str,
        required=True,
        help="GitHub repository (e.g., ROCm/TheRock)",
    )
    parser.add_argument(
        "--amdgpu-family",
        type=str,
        required=True,
        help="AMDGPU family (e.g., gfx942, gfx1151)",
    )
    parser.add_argument(
        "--test-script",
        type=str,
        required=True,
        help="Test script command to run",
    )
    parser.add_argument(
        "--shard-index",
        type=str,
        default="1",
        help="Shard index for sharded tests",
    )
    parser.add_argument(
        "--total-shards",
        type=str,
        default="1",
        help="Total number of shards",
    )
    parser.add_argument(
        "--test-type",
        type=str,
        default="full",
        help="Test type (e.g., full, smoke)",
    )
    parser.add_argument(
        "--container-image",
        type=str,
        default=DEFAULT_CONTAINER_IMAGE,
        help="Docker container image to use",
    )
    parser.add_argument(
        "--fetch-artifact-args",
        type=str,
        default="",
        help="Additional arguments for install_rocm_from_artifacts.py",
    )
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="Set up environment and drop into shell without running the test",
    )
    parser.add_argument(
        "--print-steps",
        action="store_true",
        help="Print manual reproduction steps and exit (no Docker required)",
    )

    args = parser.parse_args()

    if args.print_steps:
        print_reproduction_steps(args)
        return 0

    return run_reproduction(args, run_test=not args.setup_only)


if __name__ == "__main__":
    sys.exit(main())
