#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Run rocSPARSE's GPU-independent native unit-test binary."""

import logging
import os
import shlex
import subprocess
import sys
from pathlib import Path


def main() -> int:
    bin_dir = Path(os.environ["THEROCK_BIN_DIR"]).resolve()
    executable = bin_dir / "rocsparse-unit-test"
    if not executable.is_file():
        print(f"ERROR: required host-ASAN test is missing: {executable}", file=sys.stderr)
        return 1

    command = [str(executable)]
    logging.info("++ Exec %s", shlex.join(command))
    subprocess.run(command, cwd=bin_dir, check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
