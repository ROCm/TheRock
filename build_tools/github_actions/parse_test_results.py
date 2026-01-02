import argparse

def main(args):
    # collect test results, based on searching for pytest, ctest or gtest
    # parse accordingly
    # upload results
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
    args = parser.parse_args()
    main(args)