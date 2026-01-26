import json
import logging
import os
import shlex
import subprocess
import tempfile
import tomllib
from pathlib import Path

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

logging.basicConfig(level=logging.INFO)

# Load test configuration from TOML
config_path = SCRIPT_DIR / "hipdnn_test_config.toml"
with open(config_path, "rb") as f:
    config = tomllib.load(f)

logging.info(f"Loaded test config from: {config_path}")

# Write JSON config to temp file for C++ harness
fd, json_config_path = tempfile.mkstemp(suffix=".json", prefix="hipdnn_test_config_")
with os.fdopen(fd, "w") as f:
    json.dump(config, f)

os.environ["HIPDNN_TEST_CONFIG_PATH"] = json_config_path
logging.info(f"Wrote JSON config to: {json_config_path}")

cmd = [
    "ctest",
    "--test-dir",
    f"{THEROCK_BIN_DIR}/hipdnn_integration_tests_test_infra",
    "--output-on-failure",
    "--parallel",
    "8",
    "--timeout",
    "120",
]

logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")

subprocess.run(
    cmd,
    cwd=THEROCK_DIR,
    check=True,
)
