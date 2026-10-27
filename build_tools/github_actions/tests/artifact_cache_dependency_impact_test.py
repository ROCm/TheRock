# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from build_tools.github_actions.artifact_cache.dependency_impact import configured_impact


class ConfiguredImpactTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.build = Path(self.temp.name)
        self.graph = {
            "amdsmi": {"consumers": ["rocshmem", "rdc"]},
            "rocshmem": {"consumers": ["client"]},
            "rdc": {"consumers": ["client"]},
            "client": {"consumers": []},
            "rocm-core": {"consumers": []},
        }
        self.artifacts = {
            "core-amdsmi": ["amdsmi"],
            "rocshmem": ["rocSHMEM"],
            "rdc": ["rdc"],
            "client": ["client"],
            "base": ["rocm-core"],
        }

    def run_impact(self, roots=("amdsmi",)):
        (self.build / "therock_consumer_graph.json").write_text(json.dumps(self.graph))
        (self.build / "artifact_subprojects.json").write_text(json.dumps(self.artifacts))
        return configured_impact(self.build, roots)

    def test_direct_transitive_and_unrelated(self):
        report = self.run_impact()
        by_name = {a["artifact"]: a for a in report["affected_artifacts"]}
        self.assertEqual(set(by_name), {"core-amdsmi", "rocshmem", "rdc", "client"})
        self.assertEqual(by_name["rocshmem"]["reasons"][0]["dependency_path"],
                         ["amdsmi", "rocshmem"])
        self.assertEqual(len(by_name["client"]["reasons"][0]["dependency_path"]), 3)
        self.assertFalse(report["build_selection_changed"])

    def test_disabled_optional_edge_excludes_rocshmem(self):
        self.graph["amdsmi"]["consumers"] = ["rdc"]
        names = {a["artifact"] for a in self.run_impact()["affected_artifacts"]}
        self.assertNotIn("rocshmem", names)

    def test_cycles_and_multiple_roots(self):
        self.graph["client"]["consumers"] = ["amdsmi"]
        report = self.run_impact(("amdsmi", "rdc"))
        self.assertEqual(len(report["affected_subprojects"]), 4)
        self.assertEqual(len(report["affected_artifacts"][0]["reasons"]), 2)

    def test_unknown_root(self):
        with self.assertRaisesRegex(ValueError, "unknown"):
            self.run_impact(("typo",))

    def test_missing_metadata(self):
        with self.assertRaisesRegex(ValueError, "cannot read"):
            configured_impact(self.build, ["amdsmi"])

    def test_invalid_graph(self):
        self.graph["amdsmi"]["consumers"] = "rocshmem"
        with self.assertRaisesRegex(ValueError, "invalid"):
            self.run_impact()

    def test_unknown_mapping_target(self):
        self.artifacts["bad"] = ["unknown"]
        with self.assertRaisesRegex(ValueError, "unknown targets"):
            self.run_impact()

    def test_duplicate_json_field(self):
        self.run_impact()
        (self.build / "therock_consumer_graph.json").write_text('{"x": {}, "x": {}}')
        with self.assertRaisesRegex(ValueError, "duplicate"):
            configured_impact(self.build, ["x"])

    def test_ambiguous_case(self):
        self.graph["AMDSMI"] = {"consumers": []}
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            self.run_impact()



class PlannerImpactIntegrationTest(unittest.TestCase):
    def test_impact_is_serialized_without_claiming_skip(self):
        from unittest.mock import Mock
        from build_tools.github_actions.artifact_cache.planner import plan_cache
        db = Mock()
        db.lookup_batch.return_value = {}
        impact = {"affected_artifacts": [{"artifact": "rocshmem"}]}
        report = plan_cache(db=db, records=[], dependency_impact=impact).to_dict()
        self.assertEqual(report["dependency_impact"], impact)
        self.assertEqual(report["skippable_stages"], [])
        self.assertFalse(report["payloads_verified"])

    def test_cli_accepts_multiple_changed_subprojects(self):
        from build_tools.github_actions.artifact_cache.cli import build_parser
        args = build_parser().parse_args([
            "plan", "--db", "cache.db", "--build", "build",
            "--changed-subproject", "amdsmi", "--changed-subproject", "hipify",
        ])
        self.assertEqual(args.changed_subproject, ["amdsmi", "hipify"])


if __name__ == "__main__":
    unittest.main()
