#!/usr/bin/env python

"""Test script for fusilli-plugin.

This script is invoked by CI to run fusilli-plugin tests.
It sets up the environment and runs ctest on the fusilli_plugin test directory.
"""

import logging
import os
import shlex
import subprocess
import tempfile
import venv
from pathlib import Path

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

logging.basicConfig(level=logging.INFO)

# Build the ctest command
cmd = [
    "ctest",
    "--test-dir",
    f"{THEROCK_BIN_DIR}/fusilli_plugin",
    "--output-on-failure",
    "--parallel",
    "8",
    "--timeout",
    "600",
]

# Set up environment variables
environ_vars = os.environ.copy()

# Add THEROCK_BIN_DIR to PATH so rocm_agent_enumerator can be found
# In CI with flattened artifacts, rocm_agent_enumerator is in THEROCK_BIN_DIR
environ_vars["PATH"] = f"{THEROCK_BIN_DIR}:{environ_vars.get('PATH', '')}"

# Determine test filter based on TEST_TYPE environment variable
test_type = os.getenv("TEST_TYPE", "full")

if test_type == "smoke":
    # Exclude tests that start with "Full" during smoke tests
    environ_vars["GTEST_FILTER"] = "-Full*"

logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")
if test_type == "smoke":
    logging.info(f"   TEST_TYPE=smoke: Excluding Full* tests via GTEST_FILTER")

# Create a temporary venv with iree-base-compiler installed.
# iree-base-compiler provides the `iree-compile` binary which fusilli calls in a subprocess.
with tempfile.TemporaryDirectory(prefix="fusilli_test_venv_") as venv_dir:
    venv.create(venv_dir, with_pip=True)
    venv_bin = Path(venv_dir) / "bin"
    python_exe = venv_bin / "python"

    logging.info("Installing iree-base-compiler in temporary venv...")
    subprocess.run(
        [str(python_exe), "-m", "pip", "install", "-v", "iree-base-compiler"],
        check=True,
    )

    # Add venv bin to PATH so iree-compile can be found
    environ_vars["PATH"] = f"{venv_bin}:{environ_vars['PATH']}"

    # Run the tests
    subprocess.run(
        cmd,
        cwd=THEROCK_DIR,
        check=True,
        env=environ_vars,
    )
