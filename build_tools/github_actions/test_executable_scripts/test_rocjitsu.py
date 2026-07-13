#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Smoke test for the installed rocjitsu artifact payload.

This test intentionally stays CPU-only. It verifies that `--rocjitsu` artifacts
provide the installed CLI, runtime libraries, and config/schema files needed by
downstream workflows before those workflows attempt heavier emulation runs.
"""

import logging
import os
import shlex
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")

SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent
THEROCK_BIN_DIR = Path(os.getenv("THEROCK_BIN_DIR", "./build/bin")).resolve()
ROCM_PATH = THEROCK_BIN_DIR.parent


def require_file(path: Path, *, executable: bool = False) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required rocjitsu artifact file not found: {path}")
    if executable and not os.access(path, os.X_OK):
        raise PermissionError(f"Required rocjitsu executable is not executable: {path}")


def require_dir(path: Path) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"Required rocjitsu artifact directory not found: {path}")


rocjitsu_bin = THEROCK_BIN_DIR / "rocjitsu"
configs_dir = ROCM_PATH / "share" / "rocjitsu" / "configs"
schemas_dir = ROCM_PATH / "share" / "rocjitsu" / "schemas"

required_files = [
    ROCM_PATH / "lib" / "librocjitsu.so",
    ROCM_PATH / "lib" / "librocjitsu_hooks.so",
    ROCM_PATH / "lib" / "librocjitsu_kmd.so",
]

logging.info("=== Verifying rocjitsu artifact install layout ===")
logging.info(f"ROCM_PATH={ROCM_PATH}")
logging.info(f"THEROCK_BIN_DIR={THEROCK_BIN_DIR}")

require_file(rocjitsu_bin, executable=True)
for path in required_files:
    require_file(path)
require_dir(configs_dir)
require_dir(schemas_dir)

configs = sorted(p.name for p in configs_dir.glob("*.json"))
schemas = sorted(p.name for p in schemas_dir.glob("*.fbs"))
if not configs:
    raise FileNotFoundError(f"No rocjitsu config JSON files found under {configs_dir}")
if not schemas:
    raise FileNotFoundError(f"No rocjitsu schema files found under {schemas_dir}")

logging.info(f"rocjitsu configs: {configs}")
logging.info(f"rocjitsu schemas: {schemas}")

env = os.environ.copy()
env["LD_LIBRARY_PATH"] = os.pathsep.join(
    filter(
        None,
        [
            str(ROCM_PATH / "lib"),
            str(ROCM_PATH / "lib" / "rocm_sysdeps" / "lib"),
            str(ROCM_PATH / "lib" / "llvm" / "lib"),
            env.get("LD_LIBRARY_PATH", ""),
        ],
    )
)
env["PATH"] = os.pathsep.join(
    filter(None, [str(THEROCK_BIN_DIR), env.get("PATH", "")])
)

cmd = [str(rocjitsu_bin), "--version"]
logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")
subprocess.run(cmd, cwd=THEROCK_DIR, env=env, check=True)

logging.info("rocjitsu artifact smoke test passed")
