import os
import subprocess

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")

subprocess.run(
    [f"{THEROCK_BIN_DIR}/hipblaslt-test", "--gtest_filter=*pre_checkin*"],
    cwd=THEROCK_BIN_DIR,
    check=True,
)
