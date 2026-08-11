#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for BRANCH_CONFIG.json parsing and CMake generation."""

from io import StringIO
import json
import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from _therock_utils.branch_config import (
    get_source_sets_for_artifact_groups,
    load_branch_config,
)
from _therock_utils.build_topology import BuildTopology
from topology_to_cmake import generate_branch_config_flags


class BranchConfigTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.topology_path = self.temp_dir / "BUILD_TOPOLOGY.toml"
        self.branch_config_path = self.temp_dir / "BRANCH_CONFIG.json"
        self.topology_path.write_text(
            textwrap.dedent(
                """
                [source_sets.optional-hrx]
                description = "Optional HRX"
                external_git_sources = [
                  { name = "hrx", origin = "https://github.com/ROCm/hrx.git", commit = "e642a13425f46bcf909078459dd4e07df0723a0d", path = "optional-sources/hrx" },
                ]

                [artifact_groups.hip-runtime]
                description = "HIP runtime"
                type = "generic"
                source_sets = []
                """
            )
        )

    def tearDown(self):
        for path in sorted(self.temp_dir.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        self.temp_dir.rmdir()

    def write_branch_config(self, data):
        self.branch_config_path.write_text(json.dumps(data))

    def test_missing_config_is_empty(self):
        topology = BuildTopology(str(self.topology_path))
        config = load_branch_config(self.branch_config_path, topology)

        self.assertEqual(config.flags, {})
        self.assertEqual(config.source_sets, [])
        self.assertEqual(config.artifact_groups, {})

    def test_loads_flags_and_source_sets(self):
        self.write_branch_config(
            {
                "flags": {"FOO": "ON", "BAR": False},
                "source_sets": ["optional-hrx"],
                "artifact_groups": {"hip-runtime": {"source_sets": ["optional-hrx"]}},
            }
        )
        topology = BuildTopology(str(self.topology_path))
        config = load_branch_config(self.branch_config_path, topology)

        self.assertEqual(config.flags, {"FOO": "ON", "BAR": "OFF"})
        self.assertEqual(config.source_sets, ["optional-hrx"])
        self.assertEqual(
            get_source_sets_for_artifact_groups(config, ["hip-runtime"]),
            ["optional-hrx"],
        )

    def test_rejects_unknown_source_set(self):
        self.write_branch_config({"source_sets": ["does-not-exist"]})
        topology = BuildTopology(str(self.topology_path))

        with self.assertRaisesRegex(ValueError, "source set 'does-not-exist'"):
            load_branch_config(self.branch_config_path, topology)

    def test_rejects_unknown_artifact_group(self):
        self.write_branch_config(
            {"artifact_groups": {"does-not-exist": {"source_sets": []}}}
        )
        topology = BuildTopology(str(self.topology_path))

        with self.assertRaisesRegex(ValueError, "artifact group 'does-not-exist'"):
            load_branch_config(self.branch_config_path, topology)

    def test_generates_branch_config_flag_macro(self):
        self.write_branch_config({"flags": {"FOO": "ON"}})
        topology = BuildTopology(str(self.topology_path))
        config = load_branch_config(self.branch_config_path, topology)
        output = StringIO()

        generate_branch_config_flags(config, output)

        self.assertIn("macro(therock_apply_branch_config_flags)", output.getvalue())
        self.assertIn('therock_override_flag_default(FOO "ON")', output.getvalue())


if __name__ == "__main__":
    unittest.main()
