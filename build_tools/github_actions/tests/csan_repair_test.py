# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import csan_repair as repair
from workflow_utils import load_workflow, WORKFLOWS_DIR


class RepairTest(unittest.TestCase):
    def test_input_boundary(self):
        stages = ",".join(sorted(repair.STAGES))
        repair.validate_inputs(stages, "43011431315", "", "gfx94x")
        for args in (
            (stages + ",math-libs", "123", "", "gfx94x"),
            ("all", "123", "", "gfx94x"),
            (stages, "0", "", "gfx94x"),
            (stages, "123", "math-libs", "gfx94x"),
            (stages, "123", "", "all"),
            (stages, "123", "", "gfx94x", True),
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                repair.validate_inputs(*args)

    def test_missing_archive_or_checksum_fails_closed(self):
        class Backend:
            def __init__(self, keys):
                self.keys = keys

            def artifact_exists(self, key):
                return key in self.keys

        repair.check_artifacts(Backend({"a", "a.sha256sum"}), ["a"])
        for keys, archives in (({"a"}, ["a"]), ({"a.sha256sum"}, ["a"]), (set(), [])):
            with self.subTest(keys=keys), self.assertRaises(ValueError):
                repair.check_artifacts(Backend(keys), archives)

    def test_workflow_matches_repair_plan_and_orders_compiler_first(self):
        workflow = load_workflow(WORKFLOWS_DIR / "ci_csan.yml")
        jobs = workflow["jobs"]
        compiler = jobs["repair_compiler"]
        self.assertIn("copy_prebuilt_stages", compiler["needs"])
        self.assertEqual(
            set(compiler["with"]["rebuild_artifacts"].split(",")),
            repair.REBUILDS["compiler-runtime"],
        )
        consumers = jobs["repair_components"]
        self.assertIn("repair_compiler", consumers["needs"])
        self.assertFalse(consumers["strategy"]["fail-fast"])
        plan = consumers["strategy"]["matrix"]["include"]
        for entry in plan:
            if entry["stage"] == "math-libs":
                self.assertEqual(entry["artifacts"], "")
                self.assertGreater(entry["timeout"], 480)
            else:
                self.assertEqual(
                    set(entry["artifacts"].split(",")), repair.REBUILDS[entry["stage"]]
                )
        self.assertEqual(
            {p["stage"] for p in plan},
            (repair.REBUILDS.keys() - {"compiler-runtime"}) | {"math-libs"},
        )
        self.assertIn("!inputs.component_repair", jobs["linux_build"]["if"])
        for name in ("repair_compiler", "repair_components"):
            self.assertIn(name, jobs["summary"]["needs"])


if __name__ == "__main__":
    unittest.main()
