import os
import subprocess

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
TESTS_TO_IGNORE = "'rocprim.lookback_reproducibility|rocprim.linking|rocprim.device_merge_inplace|rocprim.device_merge_sort|rocprim.device_partition|rocprim.device_radix_sort|rocprim.device_select'"

subprocess.run([
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
    "until-pass:3"
    ""
    ], cwd=THEROCK_BIN_DIR, check=True)
