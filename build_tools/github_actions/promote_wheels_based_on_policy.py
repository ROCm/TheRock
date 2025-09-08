import argparse
import os
from github_actions_utils import *


def determine_upload_flag(
    build_result, test_result, test_runs_on, bypass_tests_for_releases
):
    # Default to false
    upload = "false"
    # 1) If the build failed → upload=false
    if build_result != "success":
        print("::warning::Build failed. Skipping upload.")

    # 2) Else if there was a test runner AND tests failed or were skipped → upload=false
    elif test_runs_on and (test_result in ["failure", "skipped"]):
        print(
            "::warning::Tests failed or were skipped (runner present). Skipping upload."
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

    upload = determine_upload_flag(
        build_result, test_result, test_runs_on, bypass_tests_for_releases
    )

    # Export result so GitHub Actions env variable
    gha_set_env({"upload": upload})


if __name__ == "__main__":
    main(sys.argv[1:])
