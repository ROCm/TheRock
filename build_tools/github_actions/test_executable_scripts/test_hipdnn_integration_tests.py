import logging
import os
import shlex
import subprocess
from pathlib import Path

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

logging.basicConfig(level=logging.INFO)

# Derive plugin path from THEROCK_BIN_DIR
# THEROCK_BIN_DIR = .../dist/rocm/bin
# Plugin path = .../dist/rocm/lib/hipdnn_plugins/engines/<plugin>.so
bin_dir = Path(THEROCK_BIN_DIR)
plugin_dir = bin_dir.parent / "lib" / "hipdnn_plugins" / "engines"

# Find the first available plugin (or specify a particular one)
# For now, use fusilli_plugin as default
plugin_path = plugin_dir / "libfusilli_plugin.so"
if not plugin_path.exists():
    # Fall back to finding any plugin
    plugins = list(plugin_dir.glob("*.so"))
    if plugins:
        plugin_path = plugins[0]
    else:
        raise RuntimeError(f"No plugins found in {plugin_dir}")

# Set environment variable for plugin path
os.environ["HIPDNN_TEST_PLUGIN_PATH"] = str(plugin_path)
logging.info(f"Using plugin: {plugin_path}")

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
