# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from pathlib import Path
import os
import sys
import unittest

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import fetch_package_targets

from workflow_utils import WORKFLOWS_DIR, get_choice_options, load_workflow


class FetchPackageTargetsTest(unittest.TestCase):
    def test_amdgpu_families_accepted_by_choice_workflows(self):
        """Checks workflow choice lists against fetch_package_targets outputs.

        fetch_package_targets.py produces amdgpu_family values that get
        dispatched to workflows. Those workflow inputs can either use
        "type: choice" or "type: string". If they use "type: choice", this tests
        that the choices are valid:

        1. Check that every amdgpu_family from the matrix is in every workflow's
          choice list. If a produced value isn't in the target's choice list,
          GitHub rejects the dispatch.
          See https://github.com/ROCm/TheRock/issues/3634.
        2. Check that the workflow choice list does not contain any extra
          families. While not an error, extra families may indicate an
          incomplete migration.
        """
        # Collect all amdgpu_family values the matrix can produce.
        produced_families = set()
        for platform in ("linux", "windows"):
            targets = fetch_package_targets.determine_package_targets(
                {"AMDGPU_FAMILIES": None, "THEROCK_PACKAGE_PLATFORM": platform}
            )
            for t in targets:
                produced_families.add(t["amdgpu_family"])

        self.assertGreater(len(produced_families), 0)

        # Find all workflows with amdgpu_family as type: choice and check.
        choice_workflows = {}
        for workflow_path in sorted(WORKFLOWS_DIR.glob("*.yml")):
            workflow = load_workflow(workflow_path)
            options = get_choice_options(workflow, "amdgpu_family")
            if options is not None:
                choice_workflows[workflow_path.name] = options

        self.assertGreater(
            len(choice_workflows),
            0,
            "No workflows found with amdgpu_family as type: choice - "
            "was the input renamed?",
        )

        errors = []
        for workflow_name, options in choice_workflows.items():
            # Test for missing families
            missing_families = produced_families - set(options)
            if missing_families:
                errors.append(
                    f"{workflow_name} is missing amdgpu_family options "
                    f"that fetch_package_targets can produce: {sorted(missing_families)}"
                )
            # Test for extra families
            extra_families = set(options) - produced_families
            if extra_families:
                errors.append(
                    f"{workflow_name} has extra amdgpu_family options not listed in fetch_package_targets: {sorted(extra_families)} "
                )

        if errors:
            self.fail("\n".join(errors))

    def test_linux_single_family(self):
        args = {
            "AMDGPU_FAMILIES": "gfx94x",
            "THEROCK_PACKAGE_PLATFORM": "linux",
        }
        targets = fetch_package_targets.determine_package_targets(args)

        self.assertEqual(len(targets), 1)

    def test_linux_multiple_families(self):
        # Note the punctuation that gets stripped and x that gets changed to X.
        args = {
            "AMDGPU_FAMILIES": "gfx94x ,; gfx110x",
            "THEROCK_PACKAGE_PLATFORM": "linux",
        }
        targets = fetch_package_targets.determine_package_targets(args)

        self.assertGreater(
            len(targets),
            1,
        )

    def test_linux_no_families(self):
        args = {
            "AMDGPU_FAMILIES": None,
            "THEROCK_PACKAGE_PLATFORM": "linux",
        }
        targets = fetch_package_targets.determine_package_targets(args)

        self.assertTrue(all("amdgpu_family" in t for t in targets))
        # Standard targets have suffixes and may use X for a family.
        self.assertTrue(any("gfx94X-dcgpu" == t["amdgpu_family"] for t in targets))
        self.assertTrue(any("gfx110X-all" == t["amdgpu_family"] for t in targets))

    def test_windows_single_family(self):
        args = {
            "AMDGPU_FAMILIES": "gfx120x",
            "THEROCK_PACKAGE_PLATFORM": "linux",
        }
        targets = fetch_package_targets.determine_package_targets(args)

        self.assertEqual(len(targets), 1)

    def test_windows_no_families(self):
        args = {
            "AMDGPU_FAMILIES": None,
            "THEROCK_PACKAGE_PLATFORM": "windows",
        }
        targets = fetch_package_targets.determine_package_targets(args)

        self.assertTrue(all("amdgpu_family" in t for t in targets))
        # dcgpu targets are Linux only.
        self.assertFalse(any("gfx94X-dcgpu" == t["amdgpu_family"] for t in targets))
        self.assertTrue(any("gfx110X-all" == t["amdgpu_family"] for t in targets))
        self.assertTrue(any("gfx120X-all" == t["amdgpu_family"] for t in targets))


if __name__ == "__main__":
    unittest.main()
