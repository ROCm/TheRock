import argparse
import os
import sys

from pathlib import Path

import pytest

from pytorch_utils import set_visible_devices_from_amdgpu_family

THIS_SCRIPT_DIR = Path(__file__).resolve().parent


def cmd_arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="""
Runs PyTorch smoke-tests for AMD GPUs.
"""
    )

    parser.add_argument(
        "--amdgpu-family",
        type=str,
        default=os.getenv("AMDGPU_FAMILY", ""),
        required=False,
        help="""Amdgpu family (e.g. "gfx942").
Select (potentially) additional tests to be skipped based on the amdgpu family""",
    )

    args = parser.parse_args(argv)

    return args


def main() -> int:
    """Main entry point for the PyTorch smoke-tests runner.

    Returns:
        Exit code from pytest (0 for success, non-zero for failures).
    """
    args = cmd_arguments(sys.argv[1:])

    # Assumes that the smoke-tests are located in the same directory as this script
    smoke_tests_dir = THIS_SCRIPT_DIR / "smoke-tests"

    if not smoke_tests_dir.exists():
        logging.error(f"Directory at '{smoke_tests_dir}' does not exist.")
        exit(1)

    # CRITICAL: Determine AMDGPU family and set HIP_VISIBLE_DEVICES
    # BEFORE importing torch/running pytest. Once torch.cuda is initialized,
    # changing HIP_VISIBLE_DEVICES has no effect.
    amdgpu_family = set_visible_devices_from_amdgpu_family(args.amdgpu_family)
    print(f"Using AMDGPU family: {amdgpu_family}")

    pytest_args = [
        f"{smoke_tests_dir}",
        "--log-cli-level=INFO",
        "-v",
    ]

    retcode = pytest.main(pytest_args)
    print(f"Pytest finished with return code: {retcode}")
    return retcode


if __name__ == "__main__":
    # Lets make this script return pytest exit code (success or failure)
    sys.exit(main())
