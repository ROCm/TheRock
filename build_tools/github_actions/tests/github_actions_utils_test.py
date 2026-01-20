import os
from pathlib import Path
import sys
import unittest

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
from github_actions_utils import *

# Note: these tests use the network and require GITHUB_TOKEN to avoid rate limits.


class GitHubActionsUtilsTest(unittest.TestCase):
    def setUp(self):
        # Save environment state
        self._saved_env = {}
        for key in ["RELEASE_TYPE", "GITHUB_REPOSITORY", "IS_PR_FROM_FORK"]:
            if key in os.environ:
                self._saved_env[key] = os.environ[key]
        # Clean environment for tests
        for key in ["RELEASE_TYPE", "GITHUB_REPOSITORY", "IS_PR_FROM_FORK"]:
            if key in os.environ:
                del os.environ[key]

    def tearDown(self):
        # Restore environment state
        for key in ["RELEASE_TYPE", "GITHUB_REPOSITORY", "IS_PR_FROM_FORK"]:
            if key in os.environ:
                del os.environ[key]
        for key, value in self._saved_env.items():
            os.environ[key] = value

    @unittest.skipUnless(
        os.getenv("GITHUB_TOKEN"),
        "GITHUB_TOKEN not set, skipping test that requires GitHub API access",
    )
    def test_gha_query_workflow_run_information(self):
        workflow_run = gha_query_workflow_run_information("ROCm/TheRock", "18022609292")
        self.assertEqual(workflow_run["repository"]["full_name"], "ROCm/TheRock")

        # Useful for debugging
        # import json
        # print(json.dumps(workflow_run, indent=2))

    @unittest.skipUnless(
        os.getenv("GITHUB_TOKEN"),
        "GITHUB_TOKEN not set, skipping test that requires GitHub API access",
    )
    def test_retrieve_bucket_info(self):
        # TODO(geomin12): work on pulling these run IDs more dynamically
        # https://github.com/ROCm/TheRock/actions/runs/18022609292?pr=1597
        external_repo, bucket = retrieve_bucket_info("ROCm/TheRock", "18022609292")
        self.assertEqual(external_repo, "")
        self.assertEqual(bucket, "therock-artifacts")

    @unittest.skipUnless(
        os.getenv("GITHUB_TOKEN"),
        "GITHUB_TOKEN not set, skipping test that requires GitHub API access",
    )
    def test_retrieve_newer_bucket_info(self):
        # https://github.com/ROCm/TheRock/actions/runs/19680190301
        external_repo, bucket = retrieve_bucket_info("ROCm/TheRock", "19680190301")
        self.assertEqual(external_repo, "")
        self.assertEqual(bucket, "therock-ci-artifacts")

    @unittest.skipUnless(
        os.getenv("GITHUB_TOKEN"),
        "GITHUB_TOKEN not set, skipping test that requires GitHub API access",
    )
    def test_retrieve_bucket_info_from_fork(self):
        # https://github.com/ROCm/TheRock/actions/runs/18023442478?pr=1596
        external_repo, bucket = retrieve_bucket_info("ROCm/TheRock", "18023442478")
        self.assertEqual(external_repo, "ROCm-TheRock/")
        self.assertEqual(bucket, "therock-artifacts-external")

    @unittest.skipUnless(
        os.getenv("GITHUB_TOKEN"),
        "GITHUB_TOKEN not set, skipping test that requires GitHub API access",
    )
    def test_retrieve_bucket_info_from_rocm_libraries(self):
        # https://github.com/ROCm/rocm-libraries/actions/runs/18020401326?pr=1828
        external_repo, bucket = retrieve_bucket_info(
            "ROCm/rocm-libraries", "18020401326"
        )
        self.assertEqual(external_repo, "ROCm-rocm-libraries/")
        self.assertEqual(bucket, "therock-artifacts-external")

    @unittest.skipUnless(
        os.getenv("GITHUB_TOKEN"),
        "GITHUB_TOKEN not set, skipping test that requires GitHub API access",
    )
    def test_retrieve_newer_bucket_info_from_rocm_libraries(self):
        # https://github.com/ROCm/rocm-libraries/actions/runs/19784318631
        external_repo, bucket = retrieve_bucket_info(
            "ROCm/rocm-libraries", "19784318631"
        )
        self.assertEqual(external_repo, "ROCm-rocm-libraries/")
        self.assertEqual(bucket, "therock-ci-artifacts-external")

    @unittest.skipUnless(
        os.getenv("GITHUB_TOKEN"),
        "GITHUB_TOKEN not set, skipping test that requires GitHub API access",
    )
    def test_retrieve_bucket_info_for_release(self):
        # https://github.com/ROCm/TheRock/actions/runs/19157864140
        os.environ["RELEASE_TYPE"] = "nightly"
        external_repo, bucket = retrieve_bucket_info("ROCm/TheRock", "19157864140")
        self.assertEqual(external_repo, "")
        self.assertEqual(bucket, "therock-nightly-artifacts")

    def test_retrieve_bucket_info_without_workflow_id(self):
        """Test bucket info retrieval without making API calls."""
        # Test default case (no workflow_run_id, no API call)
        os.environ["GITHUB_REPOSITORY"] = "ROCm/TheRock"
        os.environ["IS_PR_FROM_FORK"] = "false"
        external_repo, bucket = retrieve_bucket_info()
        self.assertEqual(external_repo, "")
        self.assertEqual(bucket, "therock-ci-artifacts")

        # Test external repo case
        os.environ["GITHUB_REPOSITORY"] = "SomeOrg/SomeRepo"
        external_repo, bucket = retrieve_bucket_info()
        self.assertEqual(external_repo, "SomeOrg-SomeRepo/")
        self.assertEqual(bucket, "therock-ci-artifacts-external")

        # Test fork case
        os.environ["GITHUB_REPOSITORY"] = "ROCm/TheRock"
        os.environ["IS_PR_FROM_FORK"] = "true"
        external_repo, bucket = retrieve_bucket_info()
        self.assertEqual(external_repo, "ROCm-TheRock/")
        self.assertEqual(bucket, "therock-ci-artifacts-external")

        # Test release case
        os.environ["RELEASE_TYPE"] = "nightly"
        os.environ["IS_PR_FROM_FORK"] = "false"
        external_repo, bucket = retrieve_bucket_info()
        self.assertEqual(external_repo, "")
        self.assertEqual(bucket, "therock-nightly-artifacts")


if __name__ == "__main__":
    unittest.main()
