# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from find_artifacts_for_commit import (
    ArtifactRunInfo,
    check_if_artifacts_exist,
    find_artifacts_for_commit,
    find_artifacts_for_run,
)
from _therock_utils.workflow_outputs import WorkflowOutputRoot
from github_actions.github_actions_api import (
    GitHubAPIError,
    is_authenticated_github_api_available,
)


def _skip_unless_authenticated_github_api_is_available(test_func):
    """Decorator to skip tests unless GitHub API is available."""
    return unittest.skipUnless(
        is_authenticated_github_api_available(),
        "No authenticated GitHub API available (need GITHUB_TOKEN or authenticated gh CLI)",
    )(test_func)


# --- Mocking strategy ---
#
# These tests make real GitHub API calls to query workflow run metadata, but
# mock check_if_artifacts_exist() which does HTTP HEAD requests to S3. This is
# because:
#
# 1. S3 retention: Artifacts will be subject to a retention policy, so older
#    runs' artifacts may be deleted. Mocking the S3 check avoids false failures
#    when artifacts are cleaned up.
#
# 2. Workflow run stability: The GitHub API workflow run history for these
#    pinned commits is unlikely to change (runs probably won't be re-triggered
#    or deleted for old commits). If tests become brittle we can re-evaluate.

# Known commits with CI workflow runs in ROCm/TheRock:
#   https://github.com/ROCm/TheRock/commit/77f0cb2112d1d0aaae0de6088a6e4337f2488233
#   CI run: https://github.com/ROCm/TheRock/actions/runs/20083647898
TEST_THEROCK_MAIN_COMMIT = "77f0cb2112d1d0aaae0de6088a6e4337f2488233"
TEST_THEROCK_MAIN_RUN_ID = "20083647898"

#   https://github.com/ROCm/TheRock/commit/62bc1eaa02e6ad1b49a718eed111cf4c9f03593a
#   CI run: https://github.com/ROCm/TheRock/actions/runs/20384488184
#   (PR from fork: ScottTodd/TheRock)
#   (attribution is fuzzy here, since branches from forks are often deleted,
#    we really just want to test that therock-ci-artifacts-external is used)
TEST_THEROCK_FORK_COMMIT = "62bc1eaa02e6ad1b49a718eed111cf4c9f03593a"
TEST_THEROCK_FORK_RUN_ID = "20384488184"

# Known commit with multi_arch_ci.yml workflow run in ROCm/TheRock:
#   https://github.com/ROCm/TheRock/commit/903ee444eb935adf456bf5df724e1b6f5c2ce962
#   multi_arch_ci run: https://github.com/ROCm/TheRock/actions/runs/25267756727
TEST_THEROCK_MULTI_ARCH_COMMIT = "903ee444eb935adf456bf5df724e1b6f5c2ce962"
TEST_THEROCK_MULTI_ARCH_RUN_ID = "25267756727"

# Known commit with CI workflow run in ROCm/rocm-libraries:
#   https://github.com/ROCm/rocm-libraries/commit/ab692342ac4d00268ac8a5a4efbc144c194cb45a
#   CI run: https://github.com/ROCm/rocm-libraries/actions/runs/21365647639
TEST_ROCM_LIBRARIES_COMMIT = "ab692342ac4d00268ac8a5a4efbc144c194cb45a"
TEST_ROCM_LIBRARIES_RUN_ID = "21365647639"


class FindArtifactsForCommitTest(unittest.TestCase):
    """Tests for find_artifacts_for_commit() with real GitHub API calls."""

    @_skip_unless_authenticated_github_api_is_available
    @mock.patch("find_artifacts_for_commit.check_if_artifacts_exist", return_value=True)
    def test_therock_main_commit(self, mock_check):
        """Known main commit returns ArtifactRunInfo with correct metadata."""
        results = find_artifacts_for_commit(
            commit=TEST_THEROCK_MAIN_COMMIT,
            artifact_groups=["gfx110X-all"],
            github_repository_name="ROCm/TheRock",
            workflow_file_name="ci.yml",
            platform="linux",
        )

        self.assertEqual(len(results), 1)
        info = results[0]
        self.assertIsInstance(info, ArtifactRunInfo)
        self.assertEqual(info.git_commit_sha, TEST_THEROCK_MAIN_COMMIT)
        self.assertEqual(info.github_repository_name, "ROCm/TheRock")
        self.assertEqual(info.workflow_file_name, "ci.yml")
        self.assertEqual(info.workflow_run_id, TEST_THEROCK_MAIN_RUN_ID)
        self.assertEqual(info.s3_bucket, "therock-ci-artifacts")
        self.assertEqual(info.external_repo, "")
        self.assertEqual(info.platform, "linux")
        self.assertEqual(info.artifact_group, "gfx110X-all")

        mock_check.assert_called()

    @_skip_unless_authenticated_github_api_is_available
    @mock.patch("find_artifacts_for_commit.check_if_artifacts_exist", return_value=True)
    def test_therock_fork_commit(self, mock_check):
        """Fork commit returns ArtifactRunInfo with external bucket."""
        results = find_artifacts_for_commit(
            commit=TEST_THEROCK_FORK_COMMIT,
            artifact_groups=["gfx110X-all"],
            github_repository_name="ROCm/TheRock",
            workflow_file_name="ci.yml",
            platform="linux",
        )

        self.assertEqual(len(results), 1)
        info = results[0]
        self.assertEqual(info.workflow_run_id, TEST_THEROCK_FORK_RUN_ID)
        self.assertEqual(info.s3_bucket, "therock-ci-artifacts-external")
        self.assertEqual(info.external_repo, "ROCm-TheRock/")

    @_skip_unless_authenticated_github_api_is_available
    @mock.patch(
        "find_artifacts_for_commit.check_if_artifacts_exist", return_value=False
    )
    def test_commit_with_runs_but_no_artifacts(self, mock_check):
        """Commit with workflow runs but no S3 artifacts returns empty list."""
        results = find_artifacts_for_commit(
            commit=TEST_THEROCK_MAIN_COMMIT,
            artifact_groups=["gfx110X-all"],
            github_repository_name="ROCm/TheRock",
            workflow_file_name="ci.yml",
            platform="linux",
        )

        self.assertEqual(results, [])
        mock_check.assert_called()

    @_skip_unless_authenticated_github_api_is_available
    @mock.patch("find_artifacts_for_commit.check_if_artifacts_exist", return_value=True)
    def test_platform_windows(self, mock_check):
        """Check that we can find artifacts for Windows as well as Linux."""
        results = find_artifacts_for_commit(
            commit=TEST_THEROCK_MAIN_COMMIT,
            artifact_groups=["gfx110X-all"],
            github_repository_name="ROCm/TheRock",
            workflow_file_name="ci.yml",
            platform="windows",
        )

        self.assertEqual(len(results), 1)
        info = results[0]
        self.assertEqual(info.platform, "windows")
        self.assertIn("windows", info.s3_path)

    @_skip_unless_authenticated_github_api_is_available
    @mock.patch("find_artifacts_for_commit.check_if_artifacts_exist", return_value=True)
    def test_rocm_libraries_commit(self, mock_check):
        """rocm-libraries commit uses therock-ci.yml and external bucket."""
        results = find_artifacts_for_commit(
            commit=TEST_ROCM_LIBRARIES_COMMIT,
            artifact_groups=["gfx94X-dcgpu"],
            github_repository_name="ROCm/rocm-libraries",
            workflow_file_name="therock-ci.yml",
            platform="linux",
        )

        self.assertEqual(len(results), 1)
        info = results[0]
        self.assertIsInstance(info, ArtifactRunInfo)
        self.assertEqual(info.git_commit_sha, TEST_ROCM_LIBRARIES_COMMIT)
        self.assertEqual(info.github_repository_name, "ROCm/rocm-libraries")
        self.assertEqual(info.workflow_file_name, "therock-ci.yml")
        self.assertEqual(info.workflow_run_id, TEST_ROCM_LIBRARIES_RUN_ID)
        self.assertEqual(info.s3_bucket, "therock-ci-artifacts-external")
        self.assertEqual(info.external_repo, "ROCm-rocm-libraries/")
        self.assertEqual(info.platform, "linux")
        self.assertEqual(info.artifact_group, "gfx94X-dcgpu")

        mock_check.assert_called()

    @_skip_unless_authenticated_github_api_is_available
    @mock.patch("find_artifacts_for_commit.check_if_artifacts_exist", return_value=True)
    def test_multi_arch_ci_commit(self, mock_check):
        """multi_arch_ci.yml commit returns ArtifactRunInfo with correct metadata."""
        results = find_artifacts_for_commit(
            commit=TEST_THEROCK_MULTI_ARCH_COMMIT,
            artifact_groups=["gfx110X-all"],
            github_repository_name="ROCm/TheRock",
            workflow_file_name="multi_arch_ci.yml",
            platform="linux",
        )

        self.assertEqual(len(results), 1)
        info = results[0]
        self.assertIsInstance(info, ArtifactRunInfo)
        self.assertEqual(info.git_commit_sha, TEST_THEROCK_MULTI_ARCH_COMMIT)
        self.assertEqual(info.github_repository_name, "ROCm/TheRock")
        self.assertEqual(info.workflow_file_name, "multi_arch_ci.yml")
        self.assertEqual(info.workflow_run_id, TEST_THEROCK_MULTI_ARCH_RUN_ID)
        self.assertEqual(info.s3_bucket, "therock-ci-artifacts")
        self.assertEqual(info.external_repo, "")
        self.assertEqual(info.platform, "linux")
        self.assertEqual(info.artifact_group, "gfx110X-all")

        mock_check.assert_called()

    @_skip_unless_authenticated_github_api_is_available
    @mock.patch("find_artifacts_for_commit.check_if_artifacts_exist", return_value=True)
    def test_multi_arch_ci_default_workflow(self, mock_check):
        """multi_arch_ci.yml is the default workflow_file_name."""
        results = find_artifacts_for_commit(
            commit=TEST_THEROCK_MULTI_ARCH_COMMIT,
            artifact_groups=["gfx110X-all"],
            github_repository_name="ROCm/TheRock",
            platform="linux",
        )

        self.assertEqual(len(results), 1)
        info = results[0]
        self.assertEqual(info.workflow_file_name, "multi_arch_ci.yml")
        self.assertEqual(info.workflow_run_id, TEST_THEROCK_MULTI_ARCH_RUN_ID)

    def test_rate_limit_error_raises_exception(self):
        """Rate limit errors raise GitHubAPIError (not silently return None)."""
        rate_limit_error = GitHubAPIError(
            "GitHub API rate limit exceeded. "
            "Authenticate with `gh auth login` or set GITHUB_TOKEN to increase limits."
        )

        with mock.patch(
            "find_artifacts_for_commit.gha_query_workflow_runs_for_commit",
            side_effect=rate_limit_error,
        ):
            with self.assertRaises(GitHubAPIError) as ctx:
                find_artifacts_for_commit(
                    commit="abc123",
                    artifact_groups=["gfx110X-all"],
                    github_repository_name="ROCm/TheRock",
                )

            self.assertIn("rate limit", str(ctx.exception).lower())


class FindArtifactsForCommitMultiGroupTest(unittest.TestCase):
    """Tests for multi-group behavior of find_artifacts_for_commit()."""

    @_skip_unless_authenticated_github_api_is_available
    @mock.patch("find_artifacts_for_commit.check_if_artifacts_exist", return_value=True)
    def test_multiple_groups_all_found(self, mock_check):
        """All requested groups are returned when all have artifacts."""
        results = find_artifacts_for_commit(
            commit=TEST_THEROCK_MAIN_COMMIT,
            artifact_groups=["gfx110X-all", "gfx120X-all"],
            github_repository_name="ROCm/TheRock",
            workflow_file_name="ci.yml",
            platform="linux",
        )

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].artifact_group, "gfx110X-all")
        self.assertEqual(results[1].artifact_group, "gfx120X-all")
        # Both should come from the same workflow run
        self.assertEqual(results[0].workflow_run_id, results[1].workflow_run_id)

    @_skip_unless_authenticated_github_api_is_available
    @mock.patch("find_artifacts_for_commit.check_if_artifacts_exist")
    def test_multiple_groups_partial(self, mock_check):
        """Only groups with artifacts are returned (partial result)."""

        def only_gfx110x(info):
            return info.artifact_group == "gfx110X-all"

        mock_check.side_effect = only_gfx110x

        results = find_artifacts_for_commit(
            commit=TEST_THEROCK_MAIN_COMMIT,
            artifact_groups=["gfx110X-all", "gfx120X-all"],
            github_repository_name="ROCm/TheRock",
            workflow_file_name="ci.yml",
            platform="linux",
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].artifact_group, "gfx110X-all")

    @_skip_unless_authenticated_github_api_is_available
    @mock.patch("find_artifacts_for_commit.check_if_artifacts_exist", return_value=True)
    def test_multiple_groups_preserves_requested_order(self, mock_check):
        """Results are returned in the same order as requested."""
        results = find_artifacts_for_commit(
            commit=TEST_THEROCK_MAIN_COMMIT,
            artifact_groups=["gfx120X-all", "gfx110X-all"],
            github_repository_name="ROCm/TheRock",
            workflow_file_name="ci.yml",
            platform="linux",
        )

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].artifact_group, "gfx120X-all")
        self.assertEqual(results[1].artifact_group, "gfx110X-all")


class FindArtifactsCrossRunTest(unittest.TestCase):
    """Tests for cross-run artifact accumulation (fully mocked, no API).

    When a workflow is retriggered for the same commit (e.g. manual re-dispatch),
    GitHub creates a new workflow run with a distinct run ID. The API returns
    all runs for the commit, most recent first. These tests verify that
    find_artifacts_for_commit() accumulates groups across those distinct runs.

    Note: This is different from "re-run failed jobs" which creates a new
    *attempt* under the same run ID. Since attempts share a run ID (and thus
    the same S3 path), cross-attempt accumulation happens implicitly.
    """

    # Fake workflow runs representing two retriggered CI runs for the same
    # commit, ordered most-recent-first (as the GitHub API returns them).
    FAKE_RUN_NEWER = {
        "id": 99999999902,
        "status": "completed",
        "conclusion": "failure",
        "html_url": "https://github.com/ROCm/TheRock/actions/runs/99999999902",
    }
    FAKE_RUN_OLDER = {
        "id": 99999999901,
        "status": "completed",
        "conclusion": "success",
        "html_url": "https://github.com/ROCm/TheRock/actions/runs/99999999901",
    }

    @mock.patch("find_artifacts_for_commit.check_if_artifacts_exist")
    @mock.patch("find_artifacts_for_commit.WorkflowOutputRoot.from_workflow_run")
    @mock.patch("find_artifacts_for_commit.gha_query_workflow_runs_for_commit")
    def test_accumulates_groups_across_runs(
        self, mock_query_runs, mock_from_wfr, mock_check
    ):
        """Groups found across different retriggered runs are accumulated."""
        mock_query_runs.return_value = [self.FAKE_RUN_NEWER, self.FAKE_RUN_OLDER]
        mock_from_wfr.return_value = WorkflowOutputRoot(
            bucket="therock-ci-artifacts",
            external_repo="",
            run_id="0",
            platform="linux",
        )

        # Newer run only built gfx120X; older run built gfx110X.
        def check_by_run_and_group(info):
            if info.workflow_run_id == str(self.FAKE_RUN_NEWER["id"]):
                return info.artifact_group == "gfx120X-all"
            if info.workflow_run_id == str(self.FAKE_RUN_OLDER["id"]):
                return info.artifact_group == "gfx110X-all"
            return False

        mock_check.side_effect = check_by_run_and_group

        results = find_artifacts_for_commit(
            commit="abc123",
            artifact_groups=["gfx110X-all", "gfx120X-all"],
            github_repository_name="ROCm/TheRock",
            platform="linux",
        )

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].artifact_group, "gfx110X-all")
        self.assertEqual(results[1].artifact_group, "gfx120X-all")
        # Each group comes from a different run
        self.assertEqual(results[0].workflow_run_id, str(self.FAKE_RUN_OLDER["id"]))
        self.assertEqual(results[1].workflow_run_id, str(self.FAKE_RUN_NEWER["id"]))

    @mock.patch("find_artifacts_for_commit.check_if_artifacts_exist")
    @mock.patch("find_artifacts_for_commit.WorkflowOutputRoot.from_workflow_run")
    @mock.patch("find_artifacts_for_commit.gha_query_workflow_runs_for_commit")
    def test_newer_run_takes_priority(self, mock_query_runs, mock_from_wfr, mock_check):
        """When multiple retriggered runs have the same group, the newer wins."""
        mock_query_runs.return_value = [self.FAKE_RUN_NEWER, self.FAKE_RUN_OLDER]
        mock_from_wfr.return_value = WorkflowOutputRoot(
            bucket="therock-ci-artifacts",
            external_repo="",
            run_id="0",
            platform="linux",
        )
        mock_check.return_value = True  # both runs have all groups

        results = find_artifacts_for_commit(
            commit="abc123",
            artifact_groups=["gfx110X-all"],
            github_repository_name="ROCm/TheRock",
            platform="linux",
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].workflow_run_id, str(self.FAKE_RUN_NEWER["id"]))


class ConcreteArtifactInspectionTest(unittest.TestCase):
    def _info(
        self,
        *,
        artifact_group: str = "gfx94X-dcgpu",
        amdgpu_targets: tuple[str, ...] = (),
        required_patterns: tuple[str, ...] = (),
    ) -> ArtifactRunInfo:
        return ArtifactRunInfo(
            git_commit_sha="abc123",
            github_repository_name="ROCm/TheRock",
            external_repo="",
            platform="linux",
            artifact_group=artifact_group,
            workflow_file_name="multi_arch_ci.yml",
            workflow_run_id="123",
            workflow_run_status="completed",
            workflow_run_conclusion="success",
            workflow_run_html_url="https://example.invalid/run/123",
            s3_bucket="therock-ci-artifacts",
            amdgpu_targets=amdgpu_targets,
            required_artifact_patterns=required_patterns,
            _output_root=mock.sentinel.output_root,
        )

    @mock.patch("find_artifacts_for_commit.S3Backend")
    def test_lists_concrete_artifacts(self, mock_backend_class):
        mock_backend_class.return_value.list_artifacts.return_value = [
            "base_lib_generic.tar.zst",
            "rocblas_lib_gfx94X-dcgpu.tar.zst",
        ]
        info = self._info()

        self.assertTrue(check_if_artifacts_exist(info))
        self.assertEqual(
            info.artifact_filenames,
            (
                "base_lib_generic.tar.zst",
                "rocblas_lib_gfx94X-dcgpu.tar.zst",
            ),
        )

    @mock.patch("find_artifacts_for_commit.S3Backend")
    def test_generic_only_does_not_prove_gpu_group(
        self,
        mock_backend_class,
    ):
        mock_backend_class.return_value.list_artifacts.return_value = [
            "base_lib_generic.tar.zst",
        ]
        info = self._info()

        self.assertFalse(check_if_artifacts_exist(info))
        self.assertEqual(info.artifact_filenames, ())

    @mock.patch("find_artifacts_for_commit.S3Backend")
    def test_individual_target_matches_xnack_variant(
        self,
        mock_backend_class,
    ):
        mock_backend_class.return_value.list_artifacts.return_value = [
            "base_lib_generic.tar.zst",
            "rocblas_lib_gfx942:xnack+.tar.zst",
        ]
        info = self._info(
            amdgpu_targets=("gfx942",),
        )

        self.assertTrue(check_if_artifacts_exist(info))
        self.assertIn(
            "rocblas_lib_gfx942:xnack+.tar.zst",
            info.artifact_filenames,
        )

    @mock.patch("find_artifacts_for_commit.S3Backend")
    def test_missing_required_pattern_rejects_run(
        self,
        mock_backend_class,
    ):
        mock_backend_class.return_value.list_artifacts.return_value = [
            "base_lib_generic.tar.zst",
            "rocblas_lib_gfx94X-dcgpu.tar.zst",
        ]
        info = self._info(
            required_patterns=(r"^amd-llvm_.*_generic\.tar\.(zst|xz)$",),
        )

        self.assertFalse(check_if_artifacts_exist(info))
        self.assertEqual(
            info.missing_required_artifact_patterns,
            (r"^amd-llvm_.*_generic\.tar\.(zst|xz)$",),
        )

    @mock.patch(
        "find_artifacts_for_commit.expand_families",
        return_value=["gfx942"],
    )
    @mock.patch(
        "find_artifacts_for_commit.amdgpu_family_map",
        return_value={"gfx94X-dcgpu": ["gfx942"]},
    )
    @mock.patch("find_artifacts_for_commit.S3Backend")
    def test_family_is_expanded_to_individual_target(
        self,
        mock_backend_class,
        mock_family_map,
        mock_expand_families,
    ):
        mock_backend_class.return_value.list_artifacts.return_value = [
            "base_lib_generic.tar.zst",
            "rocblas_lib_gfx942.tar.zst",
        ]

        info = self._info(
            artifact_group="gfx94X-dcgpu",
        )

        self.assertTrue(check_if_artifacts_exist(info))
        self.assertEqual(
            info.artifact_filenames,
            (
                "base_lib_generic.tar.zst",
                "rocblas_lib_gfx942.tar.zst",
            ),
        )

        mock_expand_families.assert_called_once_with(
            ["gfx94X-dcgpu"],
            {"gfx94X-dcgpu": ["gfx942"]},
            strict=False,
        )

    @mock.patch("find_artifacts_for_commit.S3Backend")
    def test_real_family_mapping_matches_concrete_target(
        self,
        mock_backend_class,
    ):
        mock_backend_class.return_value.list_artifacts.return_value = [
            "base_lib_generic.tar.zst",
            "rocblas_lib_gfx942.tar.zst",
        ]

        info = self._info(
            artifact_group="gfx94X-dcgpu",
        )

        self.assertTrue(check_if_artifacts_exist(info))
        self.assertEqual(
            info.artifact_filenames,
            (
                "base_lib_generic.tar.zst",
                "rocblas_lib_gfx942.tar.zst",
            ),
        )

    def test_missing_output_root_fails_fast(self):
        info = self._info()
        info._output_root = None

        with self.assertRaisesRegex(
            ValueError,
            "output_root is required",
        ):
            check_if_artifacts_exist(info)


class ArtifactRequestValidationTest(unittest.TestCase):
    def test_commit_lookup_rejects_targets_for_multiple_groups(self):
        with self.assertRaisesRegex(
            ValueError,
            "only with one artifact group",
        ):
            find_artifacts_for_commit(
                commit="abc123",
                artifact_groups=[
                    "gfx94X-dcgpu",
                    "gfx120X-all",
                ],
                amdgpu_targets=["gfx942"],
            )

    def test_run_lookup_rejects_targets_for_multiple_groups(self):
        with self.assertRaisesRegex(
            ValueError,
            "only with one artifact group",
        ):
            find_artifacts_for_run(
                workflow_run_id="123",
                artifact_groups=[
                    "gfx94X-dcgpu",
                    "gfx120X-all",
                ],
                amdgpu_targets=["gfx942"],
            )


class SingleRunArtifactSelectionTest(unittest.TestCase):
    @mock.patch("find_artifacts_for_commit.WorkflowOutputRoot.from_workflow_run")
    @mock.patch("find_artifacts_for_commit.check_if_artifacts_exist")
    @mock.patch("find_artifacts_for_commit.gha_query_workflow_runs_for_commit")
    def test_single_run_does_not_combine_workflow_runs(
        self,
        mock_query_runs,
        mock_check,
        mock_output_root,
    ):
        mock_query_runs.return_value = [
            {
                "id": 200,
                "status": "completed",
                "conclusion": "success",
                "html_url": "https://example.invalid/200",
                "run_attempt": 2,
            },
            {
                "id": 100,
                "status": "completed",
                "conclusion": "success",
                "html_url": "https://example.invalid/100",
                "run_attempt": 1,
            },
        ]

        mock_output_root.return_value.external_repo = ""
        mock_output_root.return_value.bucket = "therock-ci-artifacts"

        def check(info):
            if info.workflow_run_id == "200":
                return info.artifact_group == "gfx110X-all"
            return True

        mock_check.side_effect = check

        results = find_artifacts_for_commit(
            commit="abc123",
            artifact_groups=[
                "gfx110X-all",
                "gfx120X-all",
            ],
            require_single_run=True,
        )

        self.assertEqual(len(results), 2)
        self.assertEqual(
            {info.workflow_run_id for info in results},
            {"100"},
        )


class SuccessfulRunFilterTest(unittest.TestCase):
    @mock.patch("find_artifacts_for_commit.WorkflowOutputRoot.from_workflow_run")
    @mock.patch(
        "find_artifacts_for_commit.check_if_artifacts_exist",
        return_value=True,
    )
    @mock.patch("find_artifacts_for_commit.gha_query_workflow_runs_for_commit")
    def test_require_successful_run_skips_queued_run(
        self,
        mock_query_runs,
        mock_check_artifacts,
        mock_output_root,
    ):
        mock_query_runs.return_value = [
            {
                "id": 200,
                "status": "queued",
                "conclusion": None,
                "html_url": "https://example.invalid/200",
                "run_attempt": 2,
            },
            {
                "id": 100,
                "status": "completed",
                "conclusion": "success",
                "html_url": "https://example.invalid/100",
                "run_attempt": 1,
            },
        ]

        mock_output_root.return_value.external_repo = ""
        mock_output_root.return_value.bucket = "therock-ci-artifacts"

        results = find_artifacts_for_commit(
            commit="abc123",
            artifact_groups=["gfx94X-dcgpu"],
            require_successful_run=True,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(
            results[0].workflow_run_id,
            "100",
        )

        # The queued run is filtered before artifact storage is inspected.
        self.assertEqual(
            mock_output_root.call_count,
            1,
        )
        self.assertEqual(
            mock_check_artifacts.call_count,
            1,
        )

    @mock.patch("find_artifacts_for_commit.WorkflowOutputRoot.from_workflow_run")
    @mock.patch(
        "find_artifacts_for_commit.check_if_artifacts_exist",
        return_value=True,
    )
    @mock.patch("find_artifacts_for_commit.gha_query_workflow_runs_for_commit")
    def test_default_preserves_in_progress_artifact_inspection(
        self,
        mock_query_runs,
        mock_check_artifacts,
        mock_output_root,
    ):
        mock_query_runs.return_value = [
            {
                "id": 200,
                "status": "queued",
                "conclusion": None,
                "html_url": "https://example.invalid/200",
                "run_attempt": 2,
            },
        ]

        mock_output_root.return_value.external_repo = ""
        mock_output_root.return_value.bucket = "therock-ci-artifacts"

        results = find_artifacts_for_commit(
            commit="abc123",
            artifact_groups=["gfx94X-dcgpu"],
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(
            results[0].workflow_run_id,
            "200",
        )
        mock_check_artifacts.assert_called_once()


class ExplicitRunLookupTest(unittest.TestCase):
    @mock.patch("find_artifacts_for_commit._find_artifacts_in_workflow_runs")
    @mock.patch("find_artifacts_for_commit.gha_query_workflow_run_by_id")
    def test_run_id_inspects_only_selected_run(
        self,
        mock_query_run,
        mock_find_in_runs,
    ):
        workflow_run = {
            "id": 123,
            "head_sha": "abc123",
            "status": "completed",
            "conclusion": "success",
        }
        mock_query_run.return_value = workflow_run
        mock_find_in_runs.return_value = [mock.sentinel.info]

        results = find_artifacts_for_run(
            workflow_run_id="123",
            artifact_groups=["gfx94X-dcgpu"],
            require_successful_run=True,
        )

        self.assertEqual(results, [mock.sentinel.info])
        mock_query_run.assert_called_once_with(
            github_repository="ROCm/TheRock",
            workflow_run_id="123",
        )

        call_args = mock_find_in_runs.call_args.kwargs
        self.assertEqual(call_args["commit"], "abc123")
        self.assertEqual(call_args["workflow_runs"], [workflow_run])
        self.assertEqual(
            call_args["artifact_groups"],
            ["gfx94X-dcgpu"],
        )
        self.assertTrue(call_args["require_single_run"])
        self.assertTrue(call_args["require_successful_run"])


if __name__ == "__main__":
    unittest.main()
