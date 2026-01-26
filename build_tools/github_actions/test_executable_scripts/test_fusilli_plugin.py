import json
import logging
import os
import shlex
import subprocess
import tempfile
import tomllib
import venv
from pathlib import Path

THEROCK_BIN_DIR = Path(os.getenv("THEROCK_BIN_DIR")).resolve()
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

logging.basicConfig(level=logging.INFO)

# Set up environment variables
environ_vars = os.environ.copy()

# Determine test filter based on TEST_TYPE environment variable
test_type = os.getenv("TEST_TYPE", "full")
if test_type == "smoke":
    # Exclude tests that start with "Full" during smoke tests
    environ_vars["GTEST_FILTER"] = "-Full*"
    logging.info("TEST_TYPE=smoke: Excluding Full* tests via GTEST_FILTER")

# Create a temporary venv with `iree-base-compiler` package installed. The
# package provides an `iree-compile` binary, which Fusilli and the Fusilli
# Plugin currently rely on for compilation. This requirement will eventually be
# removed - Fusilli and the plugin will use `libIREECompiler.so` built in
# TheRock.
with tempfile.TemporaryDirectory(prefix="fusilli_test_venv_") as venv_dir:
    venv.create(venv_dir, with_pip=True)
    venv_bin = Path(venv_dir) / "bin"
    python_exe = venv_bin / "python"

    # Read IREE version from build artifacts
    iree_tag_file = (
        Path(THEROCK_BIN_DIR) / "fusilli_plugin_test_infra" / "iree_tag_for_pip.txt"
    )
    if iree_tag_file.exists():
        iree_version = iree_tag_file.read_text().strip()
        logging.info(f"Found IREE version: {iree_version}")
    else:
        raise RuntimeError(f"IREE tag file not found at {iree_tag_file}")

    logging.info("Installing iree-base-compiler in temporary venv...")
    subprocess.run(
        [
            str(python_exe),
            "-m",
            "pip",
            "install",
            "-v",
            "--find-links",
            "https://iree.dev/pip-release-links.html",
            f"iree-base-compiler=={iree_version}",
        ],
        check=True,
    )

    # Add venv bin to PATH for iree-compile
    environ_vars["PATH"] = f"{venv_bin}:{environ_vars['PATH']}"

    # Add THEROCK_BIN_DIR to PATH for rocm_agent_enumerator
    environ_vars["PATH"] = f"{THEROCK_BIN_DIR}:{environ_vars['PATH']}"

    # =========================================================================
    # 1. Run fusilli plugin tests
    # =========================================================================
    fusilli_cmd = [
        "ctest",
        "--test-dir",
        f"{THEROCK_BIN_DIR}/fusilli_plugin_test_infra",
        "--output-on-failure",
        "--parallel",
        "8",
        "--timeout",
        "600",
    ]

    logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(fusilli_cmd)}")
    subprocess.run(
        fusilli_cmd,
        cwd=THEROCK_DIR,
        check=True,
        env=environ_vars,
    )

    # =========================================================================
    # 2. Run hipDNN integration tests (with fusilli as the plugin)
    # =========================================================================

    # Load fusilli-specific test configuration from TOML
    config_path = SCRIPT_DIR / "fusilli_test_config.toml"
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    logging.info(f"Loaded fusilli test config from: {config_path}")

    # Write JSON config to temp file for C++ harness
    fd, json_config_path = tempfile.mkstemp(
        suffix=".json", prefix="fusilli_test_config_"
    )
    with os.fdopen(fd, "w") as f:
        json.dump(config, f)
    environ_vars["HIPDNN_TEST_CONFIG_PATH"] = json_config_path
    logging.info(f"Wrote JSON config to: {json_config_path}")

    hipdnn_cmd = [
        "ctest",
        "--test-dir",
        f"{THEROCK_BIN_DIR}/hipdnn_integration_tests_test_infra",
        "--output-on-failure",
        "--parallel",
        "8",
        "--timeout",
        "600",
    ]

    logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(hipdnn_cmd)}")
    subprocess.run(
        hipdnn_cmd,
        cwd=THEROCK_DIR,
        check=True,
        env=environ_vars,
    )
