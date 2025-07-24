import logging
import os
import shlex
import subprocess
from pathlib import Path

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

PLATFORM = os.getenv("PLATFORM")
AMDGPU_FAMILIES = os.getenv("AMDGPU_FAMILIES")

logging.basicConfig(level=logging.INFO)

cmd = [f"{THEROCK_BIN_DIR}/hipblaslt-test", "--gtest_filter=*pre_checkin*"]

tests_to_exclude = {
    # Related issue: https://github.com/ROCm/TheRock/issues/1114
    "gfx1151": {
        "windows": ["_/aux_test.conversion/pre_checkin_aux_auxiliary_func_f16_r"]
    }
}

if AMDGPU_FAMILIES in tests_to_exclude and PLATFORM in tests_to_exclude.get(
    AMDGPU_FAMILIES, {}
):
    exclusion_list = ":".join(tests_to_exclude[AMDGPU_FAMILIES][PLATFORM])
    cmd.append(f"--gtest_filter=-{exclusion_list}")

logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")
subprocess.run(
    cmd,
    cwd=THEROCK_DIR,
    check=True,
)
