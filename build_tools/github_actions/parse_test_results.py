import argparse
import orjson
import os
from pathlib import Path
import platform

THIS_SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = THIS_SCRIPT_DIR.parent.parent
THEROCK_BIN_DIR = Path(os.getenv("THEROCK_BIN_DIR", THEROCK_DIR / "build" / "bin"))
PLATFORM = platform.system().lower()

"""
schema is:
total_tests
error_message
component_name
test_type
shard_index
total_shards
amdgpu_families
run_id
failed_test_name
"""


def parse_gtest_results(result_file: Path):
    with open(result_file, "rb") as f:
        file_bytes = f.read()
        gtest_data = orjson.loads(file_bytes)
    total_tests = gtest_data.get("tests", 0)
    for test_suite in gtest_data.get("testsuites", []):
        # There are failures in the test suite
        if test_suite.get("failures", 0) > 0:
            for test_case in test_suite.get("testsuite", []):
                test_name = test_case.get("name")
                test_failure_messages = set()
                if "failures" in test_case:
                    for failure in test_case.get("failures", []):
                        message = failure.get("failure", "")
                        test_failure_messages.add(message)
                    # data.append({
                    #     "test_name": test_name,
                    #     "failure_messages": "({})".format(", ".join(test_failure_messages)),
                    #     "total_tests": total_tests
                    # })


def main(args):
    for file in os.listdir(THEROCK_BIN_DIR):
        if file == "report.json":
            parse_gtest_results(THEROCK_BIN_DIR / "report.json")
            break
    # collect test results, based on searching for pytest, ctest or gtest
    # parse accordingly
    # upload results

    # gtest looks report.json
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test result parser")
    parser.add_argument(
        "--component-name",
        type=str,
        required=True,
        help="Name of the component being tested",
    )
    parser.add_argument(
        "--test-type",
        type=str,
        required=True,
        help="Type of test being run (smoke/full)",
    )
    parser.add_argument(
        "--shard-index",
        type=str,
        required=True,
        help="Shard index of the test ran",
    )
    parser.add_argument(
        "--total-shards",
        type=str,
        required=True,
        help="Total number of shards",
    )
    parser.add_argument(
        "--amdgpu-families",
        type=str,
        required=True,
        help="AMDGPU families used in the test",
    )
    parser.add_argument(
        "--artifact-run-id",
        type=str,
        required=True,
        help="GitHub Actions run ID where artifacts are stored",
    )
    args = parser.parse_args()
    main(args)
