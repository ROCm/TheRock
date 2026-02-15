import logging
import os
import shlex
import subprocess
from pathlib import Path

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent
AMDGPU_FAMILIES = os.getenv("AMDGPU_FAMILIES")
environ_vars = os.environ.copy()

logging.basicConfig(level=logging.INFO)

cmd = [f"{THEROCK_BIN_DIR}/test_ck_tile_pooling"]
logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")
subprocess.run(cmd, cwd=THEROCK_DIR, check=True, env=environ_vars)