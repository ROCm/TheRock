# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Consistency guards for the hand-maintained test-label metadata (issue #7782).

Behavior-neutral: these fail only if the duplicated/stale label data that #7782
flags drifts — (1) a STAGE_TO_TEST_LABELS label that resolves to no real test,
and (2) the rocgdb/tensilelite expansions duplicated between build_tools'
TEST_LABEL_GROUPS and test_tools' _CI_TEST_SELECTOR_ALIASES.
"""

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import configure_multi_arch_ci as cm
import fetch_test_configurations as ftc

sys.path.insert(0, os.fspath(Path(__file__).resolve().parents[3] / "test_tools"))
import determine_rocm_test_dependencies as drtd

# Real test labels that are intentionally NOT multi_arch_ci test_matrix jobs:
# kfdtest (artifacts.kfdtest / CORE_KFDTESTS) and composable-kernel
# (math-libs/artifact-composable-kernel.toml) are built and tested through other
# paths, so STAGE_TO_TEST_LABELS lists them without a test_matrix entry.
_LABELS_NOT_IN_TEST_MATRIX = {"kfdtest", "composable-kernel"}


class StageTestLabelConsistency(unittest.TestCase):
    def test_every_stage_label_is_real(self):
        known = (
            set(ftc.test_matrix)
            | set(ftc.TEST_LABEL_GROUPS)
            | _LABELS_NOT_IN_TEST_MATRIX
        )
        labels = {
            label for labels in cm.STAGE_TO_TEST_LABELS.values() for label in labels
        }
        unknown = sorted(labels - known)
        self.assertEqual(
            unknown,
            [],
            "STAGE_TO_TEST_LABELS references label(s) with no test_matrix entry, "
            f"TEST_LABEL_GROUPS expansion, or documented exception: {unknown}",
        )

    def test_allowlist_has_no_stale_entries(self):
        # If an allowlisted label gains a real test_matrix entry, drop it here.
        for label in _LABELS_NOT_IN_TEST_MATRIX:
            self.assertNotIn(
                label,
                set(ftc.test_matrix),
                f"{label} now has a test_matrix entry; remove it from the allowlist",
            )


class DuplicatedExpansionConsistency(unittest.TestCase):
    def test_shared_label_groups_are_identical(self):
        # rocgdb/tensilelite are expanded identically in both packages; this keeps
        # them in sync until they share one source (merge deferred: cross-package,
        # only 2 shared entries).
        tlg = ftc.TEST_LABEL_GROUPS
        csa = drtd._CI_TEST_SELECTOR_ALIASES
        for key in sorted(set(tlg) & set(csa)):
            self.assertEqual(
                tlg[key], csa[key], f"expansion of '{key}' diverged across packages"
            )


if __name__ == "__main__":
    unittest.main()
