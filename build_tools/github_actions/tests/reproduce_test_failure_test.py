# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import argparse
import os
from pathlib import Path
import shlex
import sys
import unittest

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
from reproduce_test_failure import build_reproduction_command


def _args(**overrides):
    defaults = {
        "run_id": "12345",
        "repository": "ROCm/TheRock",
        "amdgpu_family": "gfx950-dcgpu",
        "amdgpu_targets": "gfx950",
        "test_script": "python test.py",
        "output_dir": "build",
        "shard_index": "1",
        "total_shards": "1",
        "test_type": "full",
        "fetch_artifact_args": "",
        "additional_requirements_files": "",
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class BuildReproductionCommandTest(unittest.TestCase):
    def test_quotes_emulated_test_script_shell_literals(self):
        test_script = (
            "$THEROCK_BIN_DIR/mirage run --profile mi350x --emulator rocjitsu "
            "${THEROCK_BIN_DIR:+--env THEROCK_BIN_DIR=$THEROCK_BIN_DIR} "
            "-- python build_tools/github_actions/test_executable_scripts/test_runner.py"
        )
        fetch_artifact_args = "--rocrtst --tests --mirage --rocjitsu"

        cmd = build_reproduction_command(
            _args(
                test_script=test_script,
                fetch_artifact_args=fetch_artifact_args,
                test_type="quick",
            )
        )
        tokens = shlex.split(cmd)

        self.assertEqual(tokens[tokens.index("--test-script") + 1], test_script)
        self.assertEqual(
            tokens[tokens.index("--fetch-artifact-args") + 1],
            fetch_artifact_args,
        )
        self.assertEqual(tokens[tokens.index("--test-type") + 1], "quick")


if __name__ == "__main__":
    unittest.main()
