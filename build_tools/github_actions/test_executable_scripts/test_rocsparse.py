# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import logging
import os
import shlex
import subprocess
from pathlib import Path

THEROCK_BIN_DIR = Path(os.getenv("THEROCK_BIN_DIR")).resolve()
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent.resolve()

# Resolve OUTPUT_ARTIFACTS_DIR. When unset (install-tree consumers like FFM runners
# that were not provisioned by TheRock's own build job), fall back to the install
# tree's share/ directory so that `clients/matrices/` still resolves.
_output_artifacts_env = os.getenv("OUTPUT_ARTIFACTS_DIR")
if _output_artifacts_env:
    OUTPUT_ARTIFACTS_DIR = Path(_output_artifacts_env).resolve()
else:
    OUTPUT_ARTIFACTS_DIR = (THEROCK_BIN_DIR.parent / "share").resolve()


def _resolve_share_path(rel: str) -> Path:
    """Resolve a share/ path, preferring the install tree over the build tree.

    TheRock's own CI lays out `${THEROCK_DIR}/build/share/...`; FFM and other
    install-tree consumers only have `${THEROCK_BIN_DIR}/../share/...`. Check
    the install-tree location first, then fall back to the build-tree layout
    so TheRock's internal CI sees zero behavior change when the install path
    does not exist.
    """
    install_path = THEROCK_BIN_DIR.parent / "share" / rel
    if install_path.exists():
        return install_path
    return THEROCK_DIR / "build" / "share" / rel


logging.basicConfig(level=logging.INFO)

# GTest sharding
SHARD_INDEX = os.getenv("SHARD_INDEX", 1)
TOTAL_SHARDS = os.getenv("TOTAL_SHARDS", 1)
environ_vars = os.environ.copy()
# For display purposes in the GitHub Action UI, the shard array is 1th indexed. However for shard indexes, we convert it to 0th index.
environ_vars["GTEST_SHARD_INDEX"] = str(int(SHARD_INDEX) - 1)
environ_vars["GTEST_TOTAL_SHARDS"] = str(TOTAL_SHARDS)

rocsparse_smoke_yaml = _resolve_share_path("rocsparse/test/rocsparse_smoke.yaml")

# If quick tests are enabled, we run quick tests only.
# Otherwise, we run the normal test suite
test_type = os.getenv("TEST_TYPE", "full")
if test_type == "quick":
    test_filter = [
        "--yaml",
        str(rocsparse_smoke_yaml),
    ]
else:
    # TODO(#2616): Enable full tests once known test issues are resolved
    test_filter = [
        "--yaml",
        str(rocsparse_smoke_yaml),
    ]

cmd = [
    f"{THEROCK_BIN_DIR}/rocsparse-test",
    "--matrices-dir",
    f"{OUTPUT_ARTIFACTS_DIR}/clients/matrices/",
] + test_filter
logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")
subprocess.run(cmd, cwd=THEROCK_DIR, check=True, env=environ_vars)
