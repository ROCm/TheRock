# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Keep the real Debian fixture outside the timed package build in both arms."""

from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[2]


class NormalHipifyPreflightTests(unittest.TestCase):
    def test_real_installed_tool_fixture_runs_after_dependencies_before_timed_work(
        self,
    ):
        workflow = yaml.safe_load(
            (
                ROOT / ".github/workflows/multi_arch_build_native_linux_packages.yml"
            ).read_text(encoding="utf-8")
        )
        steps = workflow["jobs"]["build_native_packages"]["steps"]
        names = [step.get("name") for step in steps]
        preflight = names.index("Require HIPIFY Debian symlink preservation")
        self.assertLess(names.index("Install System requirements"), preflight)
        self.assertLess(preflight, names.index("Fetch Artifacts for all GPU families"))
        self.assertLess(preflight, names.index("Capture native package tool identity"))
        self.assertLess(preflight, names.index("Build Packages"))
        step = steps[preflight]
        self.assertEqual(
            step["if"],
            "${{ fromJSON(toJSON(github.event.inputs)).profile_normal_validation == 'true' && inputs.native_package_type == 'deb' }}",
        )
        self.assertNotIn("continue-on-error", step)
        self.assertNotIn("profile_normal_case", step["if"])
        self.assertEqual(
            step["run"],
            "python build_tools/packaging/linux/hipify_deb_fixture.py --output output/normal-validation/hipify-dh-perl-fixture.json",
        )
        upload = steps[names.index("Retain common comparison diagnostics")]
        self.assertIn("always()", upload["if"])
        self.assertIn("output/normal-validation/", upload["with"]["path"])


if __name__ == "__main__":
    unittest.main()
