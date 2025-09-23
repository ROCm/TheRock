import os
from pathlib import Path
import sys
import unittest

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
from github_actions_utils import *

# Note: these tests use the network and may need credentials to avoid rate limits.
# We might want to mock the network or conditionally disable these tests.


class GitHubActionsUtilsTest(unittest.TestCase):
    def test_gha_query_workflow_run_information(self):
        workflow_run = gha_query_workflow_run_information("ROCm/TheRock", "18022609292")
        self.assertEqual(workflow_run["repository"]["full_name"], "ROCm/TheRock")

        # Useful for debugging
        # import json
        # print(json.dumps(workflow_run, indent=2))

    def test_retrieve_bucket_info(self):
        # https://github.com/ROCm/TheRock/actions/runs/18022609292?pr=1597
        external_repo, bucket = retrieve_bucket_info("ROCm/TheRock", "18022609292")
        self.assertEqual(external_repo, "")
        self.assertEqual(bucket, "therock-artifacts")

    def test_retrieve_bucket_info_from_fork(self):
        # https://github.com/ROCm/TheRock/actions/runs/18023442478?pr=1596
        external_repo, bucket = retrieve_bucket_info("ROCm/TheRock", "18023442478")
        self.assertEqual(external_repo, "ROCm-TheRock/")
        self.assertEqual(bucket, "therock-artifacts-external")

    def test_retrieve_bucket_info_from_rocm_libraries(self):
        # https://github.com/ROCm/rocm-libraries/actions/runs/18020401326?pr=1828
        external_repo, bucket = retrieve_bucket_info(
            "ROCm/rocm-libraries", "18020401326"
        )
        self.assertEqual(external_repo, "ROCm-rocm-libraries/")
        self.assertEqual(bucket, "therock-artifacts-external")


if __name__ == "__main__":
    unittest.main()
