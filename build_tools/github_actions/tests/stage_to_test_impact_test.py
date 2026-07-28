# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

sys.path.insert(
    0,
    os.fspath(Path(__file__).parent.parent.parent),
)

from _therock_utils.build_topology import (
    BuildTopology,
    get_topology,
)
from github_actions.stage_to_test_impact import (
    TEST_COMPONENT_ARTIFACT_OVERRIDES,
    compute_test_impact,
    render_test_impact_summary,
    requested_test_components,
)


class StageToTestImpactTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

        self.topology_path = Path(self.temp_dir.name) / "BUILD_TOPOLOGY.toml"

        self.topology_path.write_text(
            textwrap.dedent(
                """
                [artifact_groups.runtime-tests]
                description = "Runtime tests"
                type = "generic"

                [artifact_groups.math-libs]
                description = "Math libraries"
                type = "per-arch"

                [artifact_groups.comm-libs]
                description = "Communication libraries"
                type = "generic"

                [artifact_groups.media-libs]
                description = "Media libraries"
                type = "generic"

                [artifact_groups.conditional]
                description = "Conditional artifact"
                type = "generic"

                [build_stages.runtime-tests]
                description = "Runtime tests"
                artifact_groups = ["runtime-tests"]

                [build_stages.math-libs]
                description = "Math libraries"
                artifact_groups = ["math-libs"]
                type = "per-arch"

                [build_stages.comm-libs]
                description = "Communication libraries"
                artifact_groups = ["comm-libs"]

                [build_stages.media-libs]
                description = "Media libraries"
                artifact_groups = ["media-libs"]

                [build_stages.conditional]
                description = "Conditional artifact"
                artifact_groups = ["conditional"]

                [artifacts.core-hiptests]
                artifact_group = "runtime-tests"
                type = "target-neutral"

                [artifacts.blas]
                artifact_group = "math-libs"
                type = "target-specific"
                split_databases = ["rocblas"]

                [artifacts.rccl]
                artifact_group = "comm-libs"
                type = "target-specific"
                disable_platforms = ["windows"]

                [artifacts.rocdecode]
                artifact_group = "media-libs"
                type = "target-neutral"
                disable_platforms = ["windows"]

                [artifacts.conditional]
                artifact_group = "conditional"
                type = "target-neutral"
                disable_platforms_if_flags_not_set = { windows = "SPECIAL_FLAG" }
                """
            ),
            encoding="utf-8",
        )

        self.topology = BuildTopology(str(self.topology_path))

    def test_maps_components_and_keeps_unknown_enabled(
        self,
    ):
        result = compute_test_impact(
            topology=self.topology,
            platform="linux",
            components=[
                "sanity",
                "rocblas",
                "rccl",
                "unknown-test",
            ],
            rebuild_stages=["math-libs"],
            full_rebuild_required=False,
        )

        self.assertEqual(
            result.would_run,
            (
                "sanity",
                "rocblas",
                "unknown-test",
            ),
        )
        self.assertEqual(
            result.would_skip,
            ("rccl",),
        )
        self.assertEqual(
            result.unmapped,
            ("unknown-test",),
        )
        self.assertEqual(
            result.forced,
            ("sanity",),
        )

    def test_full_rebuild_keeps_all_applicable_enabled(
        self,
    ):
        result = compute_test_impact(
            topology=self.topology,
            platform="linux",
            components=[
                "rocblas",
                "rccl",
            ],
            rebuild_stages=[],
            full_rebuild_required=True,
            reasons=["unknown changed path"],
        )

        self.assertEqual(
            result.would_run,
            (
                "rocblas",
                "rccl",
                "sanity",
            ),
        )
        self.assertEqual(
            result.would_skip,
            (),
        )
        self.assertEqual(
            result.reasons,
            ("unknown changed path",),
        )

    def test_platform_disabled_is_not_applicable(
        self,
    ):
        result = compute_test_impact(
            topology=self.topology,
            platform="windows",
            components=[
                "rccl",
                "rocdecode",
            ],
            rebuild_stages=[
                "comm-libs",
                "media-libs",
            ],
            full_rebuild_required=False,
        )

        self.assertEqual(
            result.not_applicable,
            (
                "rccl",
                "rocdecode",
            ),
        )
        self.assertEqual(
            result.would_run,
            ("sanity",),
        )
        self.assertEqual(
            result.would_skip,
            (),
        )

    def test_conditional_disable_remains_applicable(
        self,
    ):
        result = compute_test_impact(
            topology=self.topology,
            platform="windows",
            components=["conditional"],
            rebuild_stages=[],
            full_rebuild_required=False,
        )

        self.assertEqual(
            result.not_applicable,
            (),
        )
        self.assertEqual(
            result.would_skip,
            ("conditional",),
        )

    def test_requested_components_normalizes_labels(
        self,
    ):
        result = requested_test_components(
            [
                "test:rocblas",
                "rocprim",
                "test_filter:full",
                "",
            ]
        )

        self.assertEqual(
            result,
            (
                "rocblas",
                "rocprim",
            ),
        )

    def test_requested_component_is_forced(self):
        result = compute_test_impact(
            topology=self.topology,
            platform="linux",
            components=["rccl"],
            rebuild_stages=[],
            full_rebuild_required=False,
            forced_components=["rccl"],
        )

        self.assertIn(
            "rccl",
            result.would_run,
        )
        self.assertIn(
            "rccl",
            result.forced,
        )
        self.assertNotIn(
            "rccl",
            result.would_skip,
        )

    def test_render_summary_is_report_only(self):
        result = compute_test_impact(
            topology=self.topology,
            platform="linux",
            components=[
                "rocblas",
                "rccl",
            ],
            rebuild_stages=["math-libs"],
            full_rebuild_required=False,
        )

        summary = render_test_impact_summary([result])

        self.assertIn(
            "### Test impact analysis — dry-run",
            summary,
        )
        self.assertIn(
            "test labels and generated test matrices " "are unchanged",
            summary,
        )
        self.assertIn(
            "`rocblas`",
            summary,
        )
        self.assertIn(
            "`rccl`",
            summary,
        )

    def test_repository_override_artifacts_exist(
        self,
    ):
        topology = get_topology()

        missing = sorted(
            {
                artifact_name
                for artifact_names in (TEST_COMPONENT_ARTIFACT_OVERRIDES.values())
                for artifact_name in artifact_names
                if artifact_name not in topology.artifacts
            }
        )

        self.assertEqual(missing, [])

    def test_key_repository_components_are_mapped(self):
        topology = get_topology()

        result = compute_test_impact(
            topology=topology,
            platform="linux",
            components=[
                "hip-tests",
                "rocblas",
                "rocprim",
                "rocfft",
                "miopen",
                "rccl",
                "rocprofiler-sdk",
                "rocgdb-cpu",
                "rocdecode",
                "rocrtst",
            ],
            rebuild_stages=[],
            full_rebuild_required=False,
        )

        self.assertEqual(
            result.unmapped,
            (),
            f"Unmapped production test components: {result.unmapped}",
        )


if __name__ == "__main__":
    unittest.main()
