import logging
import os
import shlex
import subprocess
from pathlib import Path

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent
AMDGPU_FAMILIES = os.getenv("AMDGPU_FAMILIES")

logging.basicConfig(level=logging.INFO)

TESTS_TO_IGNORE = {
    # Issue to fix: https://github.com/ROCm/TheRock/issues/2758
    "gfx950-dcgpu": [
        "hipdnn_backend_tests",
    ],
    "gfx1151": ["hipdnn_backend_tests"],
}

cmd = [
    "ctest",
    "--test-dir",
    f"{THEROCK_BIN_DIR}/hipdnn",
    "--output-on-failure",
    "--parallel",
    "8",
    "--timeout",
    "60",
]

if AMDGPU_FAMILIES in TESTS_TO_IGNORE:
    ignore_list = TESTS_TO_IGNORE[AMDGPU_FAMILIES]
    cmd.extend(["--exclude-regex", "|".join(ignore_list)])

logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")

subprocess.run(
    cmd,
    cwd=THEROCK_DIR,
    check=True,
)
