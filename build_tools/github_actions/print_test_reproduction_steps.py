# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Prints test reproduction steps when a test fails in CI.

This script outputs step-by-step instructions for reproducing a test failure
locally using Docker and the exact artifacts from the failed CI run.

Usage from workflow:
    python ./build_tools/github_actions/print_test_reproduction_steps.py \
        --run-id ${{ github.run_id }} \
        --repository ${{ github.repository }} \
        --amdgpu-family gfx942 \
        --test-script "python build_tools/github_actions/test_executable_scripts/test_rocblas.py" \
        --shard-index 1 \
        --total-shards 4 \
        --test-type full
"""

import argparse

# Default container image used for testing
DEFAULT_CONTAINER_IMAGE = "ghcr.io/rocm/no_rocm_image_ubuntu24_04:latest"


def print_reproduction_steps(args: argparse.Namespace) -> None:
    print("=" * 60)
    print("TEST FAILURE - REPRODUCTION STEPS")
    print("=" * 60)
    print()
    print("To reproduce this test failure locally, follow these steps:")
    print()
    print("1. Start the Docker container:")
    print(f"   docker run -it \\")
    print(f"       --ipc host \\")
    print(f"       --group-add video \\")
    print(f"       --device /dev/kfd \\")
    print(f"       --device /dev/dri \\")
    print(f"       {args.container_image} /bin/bash")
    print()
    print("2. Inside the container, set up the environment:")
    print("   curl -LsSf https://astral.sh/uv/install.sh | bash && source $HOME/.local/bin/env")
    print("   git clone https://github.com/ROCm/TheRock.git && cd TheRock")
    print("   uv venv .venv && source .venv/bin/activate")
    print("   uv pip install -r requirements-test.txt")
    print()
    print("3. Install artifacts from this CI run:")
    print(f"   GITHUB_REPOSITORY={args.repository} python build_tools/install_rocm_from_artifacts.py \\")
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
    print("For more details, see:")
    print("https://github.com/ROCm/TheRock/blob/main/docs/development/test_environment_reproduction.md")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Print test reproduction steps for failed CI tests"
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
    args = parser.parse_args()

    print_reproduction_steps(args)
