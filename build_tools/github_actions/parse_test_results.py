import argparse
import orjson
import os
from pathlib import Path
import platform

THIS_SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = THIS_SCRIPT_DIR.parent.parent
THEROCK_BIN_DIR = Path(os.getenv("THEROCK_BIN_DIR", THEROCK_DIR / "build" / "bin"))
PLATFORM = platform.system().lower()

def parse_gtest_results(args, result_file: Path):
    with open(result_file, "rb") as f:
        file_bytes = f.read()
        gtest_data = orjson.loads(file_bytes)
    total_tests = gtest_data.get("tests", 0)
    for test_suite in gtest_data.get("testsuites", []):
        # If there are failures reported in the test suite
        if test_suite.get("failures", 0) > 0:
            for test_case in test_suite.get("testsuite", []):
                test_name = test_case.get("name")
                test_failure_messages = set()
                if "failures" in test_case:
                    for test_failure in test_case.get("failures", []):
                        message = test_failure.get("failure", "")
                        test_failure_messages.add(message)
                    data_to_insert = {
                        "component_name": args.component_name,
                        "test_type": args.test_type,
                        "shard_index": args.shard_index,
                        "total_shards": args.total_shards,
                        "amdgpu_families": args.amdgpu_families,
                        "artifact_run_id": args.artifact_run_id,
                        "failed_test_name": test_name,
                        "error_message": "({})".format(", ".join(test_failure_messages)),
                        "total_tests": total_tests,
                        "platform": PLATFORM,
                    }
                    print(data_to_insert)


def main(args):
    # Searching through the bin directory for test result files
    for file in os.listdir(THEROCK_BIN_DIR):
        # For gtest JSON test report
        if file == "report.json":
            parse_gtest_results(args, THEROCK_BIN_DIR / "report.json")
            break

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
