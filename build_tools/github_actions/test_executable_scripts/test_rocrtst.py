#!/usr/bin/env python3
import logging
import os
import shlex
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")

# repo + directory setup
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = Path(os.getenv("OUTPUT_ARTIFACTS_DIR")).resolve()
env = os.environ.copy()
platform = os.getenv("RUNNER_OS", "linux").lower()

env["LD_LIBRARY_PATH"] = os.pathsep.join(
    filter(
        None,
        [
            env.get("LD_LIBRARY_PATH"),
            str(THEROCK_DIR / "lib" / "rocrtst" / "lib"),
            str(THEROCK_DIR / "lib" / "rocm_sysdeps" / "lib"),
        ],
    )
)

# Detect GPU architecture using rocm_agent_enumerator
gpu_arch = (
    subprocess.run(
        [str(THEROCK_DIR / "bin" / "rocm_agent_enumerator")],
        capture_output=True,
        text=True,
        check=True,
    )
    .stdout.strip()
    .split("\n")[0]
)

logging.info(f"Detected GPU architecture: {gpu_arch}")

cwd_dir = THEROCK_DIR / "bin" / gpu_arch
cmd = "./rocrtst64"

logging.info(f"++ Exec [{cwd_dir}]$ {cmd}")

subprocess.run(
    cmd,
    cwd=cwd_dir,
    check=True,
    env=env,
)
