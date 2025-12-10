import argparse
import os
import sys

from pathlib import Path

import pytest

from pytorch_utils import detect_amdgpu_family, detect_pytorch_version

THIS_SCRIPT_DIR = Path(__file__).resolve().parent


def cmd_arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="""
Runs PyTorch smoke-tests for AMD GPUs.
"""
    )

    amdgpu_family = os.getenv("AMDGPU_FAMILY")
    parser.add_argument(
        "--amdgpu-family",
        type=str,
        default=os.getenv("AMDGPU_FAMILY", ""),
        required=False,
        help="""Amdgpu family (e.g. "gfx942").
Select (potentially) additional tests to be skipped based on the amdgpu family""",
    )

    default_smoke_tests_dir = THIS_SCRIPT_DIR / "smoke-tests"
    parser.add_argument(
        "--smoke-tests-dir",
        type=Path,
        default=default_smoke_tests_dir,
        help="""Path for the smoke-tests directory, where tests will be sourced from
By default the smoke-tests directory is determined based on this script's location
""",
    )

    parser.add_argument(
        "--no-cache",
        default=False,
        required=False,
        action=argparse.BooleanOptionalAction,
        help="""Disable pytest caching. Useful when only having read-only access to pytorch directory""",
    )

    args = parser.parse_args(argv)

    if not args.smoke_tests_dir.exists():
        parser.error(f"Directory at '{args.smoke_tests_dir}' does not exist.")

    return args


def main() -> int:
    """Main entry point for the PyTorch smoke-tests runner.

    Returns:
        Exit code from pytest (0 for success, non-zero for failures).
    """
    args = cmd_arguments(sys.argv[1:])

    smoke_tests_dir = args.smoke_tests_dir

    # CRITICAL: Determine AMDGPU family and set HIP_VISIBLE_DEVICES
    # BEFORE importing torch/running pytest. Once torch.cuda is initialized,
    # changing HIP_VISIBLE_DEVICES has no effect.
    amdgpu_family = detect_amdgpu_family(args.amdgpu_family)
    print(f"Using AMDGPU family: {amdgpu_family}")

    pytorch_args = [
        f"{smoke_tests_dir}",
        "--log-cli-level=INFO",
        "-v",
    ]

    if args.no_cache:
        pytorch_args += [
            "-p",
            "no:cacheprovider",  # Disable caching: useful when running in a container
        ]

    retcode = pytest.main(pytorch_args)
    print(f"Pytest finished with return code: {retcode}")
    return retcode


if __name__ == "__main__":
    # Lets make this script return pytest exit code (success or failure)
    sys.exit(main())
