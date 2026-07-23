# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Smoke test for the rocjitsu GPU emulator via the mirage CLI.

Runs ``rocminfo`` on top of the rocjitsu emulator (through mirage) and checks
that the emulated GPU is visible. The runner has no physical GPU (no /dev/kfd),
so a GPU agent in rocminfo's output can only be the one rocjitsu emulates. This
is the fast check to run whenever the emulator tooling (``emulation/mirage`` or
``emulation/rocjitsu``) changes.

Environment variables:
  THEROCK_BIN_DIR: Directory containing the ``mirage`` and ``rocminfo`` binaries.
  AMDGPU_FAMILIES: GPU family under test; selects the matching rocjitsu profile.
"""

import logging
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

# Allow importing the shared emulation helpers regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from emulation_utils import (
    build_mirage_run_command,
    find_mirage_binary,
    select_mirage_profile,
)

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
AMDGPU_FAMILIES = os.getenv("AMDGPU_FAMILIES")

logging.basicConfig(level=logging.INFO)


def main():
    if not THEROCK_BIN_DIR:
        raise EnvironmentError("THEROCK_BIN_DIR is not set")

    # rocjitsu only emulates specific agents. If this family has no matching
    # profile, skip rather than run against a mismatched agent.
    profile = select_mirage_profile(AMDGPU_FAMILIES)
    if profile is None:
        logging.warning(
            "Skipping mirage smoke test: no rocjitsu profile is available for "
            "AMDGPU family '%s'.",
            AMDGPU_FAMILIES,
        )
        return

    mirage = find_mirage_binary(THEROCK_BIN_DIR)

    # Sanity: the mirage binary is invokable.
    logging.info("++ %s", shlex.join([mirage, "--version"]))
    subprocess.run([mirage, "--version"], check=True)

    # Run rocminfo on top of the rocjitsu emulator and confirm the simulated GPU
    # is visible.
    cwd_dir = Path(THEROCK_BIN_DIR)
    cmd = build_mirage_run_command(["./rocminfo"], profile=profile, mirage_bin=mirage)

    # rocjitsu locates the ROCm runtime via ROCM_HOME (one level above bin/).
    run_env = os.environ.copy()
    run_env.setdefault("ROCM_HOME", str(cwd_dir.resolve().parent))

    logging.info("++ [%s]$ %s", cwd_dir, shlex.join(cmd))
    result = subprocess.run(
        cmd, cwd=cwd_dir, env=run_env, capture_output=True, text=True
    )
    output = (result.stdout or "") + (result.stderr or "")
    logging.info("rocminfo under rocjitsu (profile '%s'):\n%s", profile, output)

    if result.returncode != 0:
        raise RuntimeError(
            f"rocminfo failed under rocjitsu emulation (exit {result.returncode})."
        )
    if not re.search(r"Device Type:\s*GPU", output):
        raise RuntimeError(
            "rocjitsu emulation did not expose a GPU agent to rocminfo; the "
            "simulated GPU is not visible."
        )
    logging.info("mirage + rocjitsu smoke test passed: simulated GPU is visible.")


if __name__ == "__main__":
    main()
