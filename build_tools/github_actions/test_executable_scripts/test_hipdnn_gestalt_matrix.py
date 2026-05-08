# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run hipdnn_integration_tests once with all provider plugins loaded and
dump the combined support matrix to stdout.

The matrix is printed between obvious banner lines so it can be Ctrl-F'd
out of the GitHub Actions log.
"""

import logging
import os
import shlex
import subprocess
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO)

THEROCK_BIN_DIR = Path(os.getenv("THEROCK_BIN_DIR")).resolve()
ROCM_PATH = THEROCK_BIN_DIR.parent

amdgpu_families = os.getenv("AMDGPU_FAMILIES", "unknown")

binary = THEROCK_BIN_DIR / "hipdnn_integration_tests"
plugin_dir = ROCM_PATH / "lib" / "hipdnn_plugins" / "engines"

if not binary.exists():
    raise SystemExit(f"hipdnn_integration_tests not found at {binary}")
if not plugin_dir.exists():
    raise SystemExit(f"Plugin directory not found at {plugin_dir}")

loaded_plugins = sorted(p.name for p in plugin_dir.glob("*.so"))
logging.info("Plugins discoverable in %s: %s", plugin_dir, loaded_plugins)
if len(loaded_plugins) < 3:
    logging.warning(
        "Expected >=3 provider plugins; found %d. Matrix will be partial.",
        len(loaded_plugins),
    )

env = os.environ.copy()
# Defensive: pin plugin discovery to absolute path. By default the loader
# resolves the relative path "hipdnn_plugins/engines/" against the backend
# .so's module directory (PluginCore.hpp:177-183) which works in canonical
# layouts; this removes that dependency.
env["HIPDNN_PLUGIN_DIR"] = str(plugin_dir)
env["ROCM_PATH"] = str(ROCM_PATH)

with tempfile.TemporaryDirectory() as tmpdir:
    matrix_file = Path(tmpdir) / "support_matrix.md"
    cmd = [
        str(binary),
        "--generate-support-matrix",
        str(matrix_file),
        "--skip-graph-validation",
    ]
    logging.info("++ Exec $ %s", shlex.join(cmd))

    # Don't fail on test exit code 1 (gtest reports failures for unrelated
    # bugs). We only care that the matrix file gets written. Hard failures
    # (segfault, plugin load crash) get returncode > 1 and are surfaced.
    result = subprocess.run(cmd, env=env)
    if result.returncode not in (0, 1):
        raise SystemExit(
            f"hipdnn_integration_tests exited with code {result.returncode}"
        )

    if not matrix_file.exists():
        raise SystemExit(f"Support matrix file was not produced: {matrix_file}")

    banner_begin = f"===== HIPDNN SUPPORT MATRIX [{amdgpu_families}] BEGIN ====="
    banner_end = f"===== HIPDNN SUPPORT MATRIX [{amdgpu_families}] END ====="
    print()
    print(banner_begin)
    print(matrix_file.read_text(), end="")
    print(banner_end)
    print()
