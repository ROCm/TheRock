# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Runs the prebuilt libhipcxx tests from the test artifact.

The libhipcxx_tests subproject installs the c2h tests and the amdclang++ lit
tests into bin/libhipcxx, and libhipcxx_hiprtc_tests installs the hiprtc lit
tests into bin/libhipcxx/lit-hiprtc. Both directories carry a relocatable CTest
file, so nothing is compiled here.
"""

import argparse
import logging
import os
import shlex
import subprocess
from pathlib import Path

from libhipcxx_utils import prepend_env_path

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

TEST_DIRS = {
    "amdclang": Path("libhipcxx"),
    "hiprtc": Path("libhipcxx") / "lit-hiprtc",
}

logging.basicConfig(level=logging.INFO)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=sorted(TEST_DIRS),
        required=True,
        help="amdclang runs the c2h and amdclang++ lit tests, hiprtc the hiprtc lit tests",
    )
    args = parser.parse_args()

    bin_dir = Path(THEROCK_BIN_DIR).resolve()
    rocm_dir = bin_dir.parent

    env = os.environ.copy()
    env["ROCM_PATH"] = str(rocm_dir)
    prepend_env_path(env, "PATH", str(bin_dir))
    # The prebuilt lit tests have no RPATH.
    prepend_env_path(env, "LD_LIBRARY_PATH", str(rocm_dir / "lib"))

    shard_index = int(os.getenv("SHARD_INDEX", "1")) - 1
    total_shards = int(os.getenv("TOTAL_SHARDS", "1"))
    test_type = os.getenv("TEST_TYPE", "full")

    cmd = [
        "ctest",
        "--test-dir",
        str(bin_dir / TEST_DIRS[args.variant]),
        "--label-regex",
        f"^{test_type}$",
        "--output-on-failure",
        "--verbose",
        "--tests-information",
        f"{shard_index},,{total_shards}",
    ]
    logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")
    subprocess.run(cmd, cwd=THEROCK_DIR, check=True, env=env)


if __name__ == "__main__":
    main()
