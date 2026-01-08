"""
Decide whether to upload/promote build artifacts in GitHub Actions.


Reads env vars BUILD_RESULT, TEST_RESULT, TEST_RUNS_ON, BYPASS_TESTS_FOR_RELEASES and
sets `upload` ("true"/"false") via `gha_set_env` using this policy:
1) build != "success" → false
2) test runner present and tests failed/skipped → false
3) no test runner and not bypassing → false
4) otherwise → true
"""

import argparse
import os
import sys
from github_actions_utils import *


def determine_upload_flag(
    build_result, test_result, test_runs_on, bypass_tests_for_releases, branch
):
    # Default to false
    upload = "false"
    # 0) If on a release branch, always upload, as
    #    - Release branch has already been tested
    #    - Flaky tests can prevent promotion, as such ignore test results
    #    - Will insure that QA/engineers have a single point of truth to get the packages, which isnt staging
    if branch.startswith("release/therock-"):
        print(
            f"::notice::On release branch: {branch}. Forcing upload independent of test results."
        )
        upload = "true"
    # 1) If the build failed → upload=false
    elif build_result != "success":
        print("::warning::Build failed. Skipping upload.")

    # 2) Else if there was a test runner AND tests did not succeed -> upload=false
    elif test_runs_on and (test_result != "success"):
        print(
            f"::warning::Runner present and tests were not successful (test_result: {test_result}). Skipping upload."
        )

    # 3) Else if BYPASS_TESTS_FOR_RELEASES is not set and there was no test runner → upload=false
    elif not bypass_tests_for_releases and not test_runs_on:
        print(
            "::warning::No test runner and BYPASS_TESTS_FOR_RELEASES not set. Skipping upload."
        )

    # 4) Otherwise → upload=true
    else:
        upload = "true"

    return upload


def main(argv: list[str]):
    ## Added argparse for future enhancements to the script
    p = argparse.ArgumentParser(prog="promote_based_on_policy.py")
    # Read environment variables
    build_result = os.getenv("BUILD_RESULT", "").lower()
    test_result = os.getenv("TEST_RESULT", "").lower()
    test_runs_on = os.getenv("TEST_RUNS_ON", "")
    bypass_tests_for_releases = os.getenv("BYPASS_TESTS_FOR_RELEASES", "")
    branch = os.getenv("GITHUB_REF_NAME", "").lower()

    upload = determine_upload_flag(
        build_result, test_result, test_runs_on, bypass_tests_for_releases, branch
    )

    # Export result so GitHub Actions env variable
    gha_set_env({"upload": upload})


if __name__ == "__main__":
    main(sys.argv[1:])
