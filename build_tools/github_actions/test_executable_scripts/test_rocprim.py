import logging
import os
import shlex
import subprocess
from pathlib import Path

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

logging.basicConfig(level=logging.INFO)

TESTS_TO_IGNORE = "'rocprim.lookback_reproducibility|rocprim.linking|rocprim.device_merge_inplace|rocprim.device_merge_sort|rocprim.device_partition|rocprim.device_radix_sort|rocprim.device_select'"

cmd = [
    "ctest",
    "--test-dir",
    f"{THEROCK_BIN_DIR}/rocprim",
    "--output-on-failure",
    "--parallel",
    "8",
    "--exclude-regex",
    TESTS_TO_IGNORE,
    "--timeout",
    "900",
    "--repeat",
    "until-pass:6",
]
logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")

subprocess.run(
    cmd,
    cwd=THEROCK_DIR,
    check=True,
)
