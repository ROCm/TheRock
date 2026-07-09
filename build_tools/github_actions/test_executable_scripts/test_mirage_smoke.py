# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Basic smoke test for the mirage emulator CLI.

This is intentionally the most minimal emulation check: it verifies the
``mirage`` binary is functional on the runner using the ``noop`` backend, which
runs commands directly with no GPU emulation (so it needs neither a GPU nor the
rocjitsu runtime). It is the natural fast check to run whenever the emulator
tooling (``emulation/mirage`` or ``emulation/rocjitsu``) changes.

Environment variables:
  THEROCK_BIN_DIR: Directory containing the ``mirage`` binary.
"""

import logging
import os
import shlex
import subprocess
import sys
from pathlib import Path

# Allow importing the shared emulation helpers regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from emulation_utils import find_mirage_binary

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")

logging.basicConfig(level=logging.INFO)

SMOKE_MARKER = "MIRAGE_SMOKE_OK"


def _run(cmd, **kwargs):
    logging.info("++ %s", shlex.join(cmd))
    return subprocess.run(cmd, check=True, **kwargs)


def main():
    mirage = find_mirage_binary(THEROCK_BIN_DIR)

    # 1. mirage is invokable and reports a version.
    _run([mirage, "--version"])

    # 2. mirage can enumerate its emulator backends.
    _run([mirage, "emulators"])

    # 3. Exercise the full run path with the noop backend: create a transient
    #    session, run a command, and clean up. noop executes the command
    #    directly, so this works on any CPU node with no GPU or rocjitsu runtime.
    result = _run(
        [
            mirage,
            "run",
            "--emulator",
            "noop",
            "--",
            sys.executable,
            "-c",
            f"print('{SMOKE_MARKER}')",
        ],
        capture_output=True,
        text=True,
    )
    combined_output = (result.stdout or "") + (result.stderr or "")
    if SMOKE_MARKER not in combined_output:
        raise RuntimeError(
            f"mirage noop run did not forward the child output; expected "
            f"'{SMOKE_MARKER}' in:\n{combined_output}"
        )
    logging.info("mirage smoke test passed.")


if __name__ == "__main__":
    main()
