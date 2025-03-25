"""Main ROCm SDK CLI for managing the Python installation."""

import argparse
import sys

from . import _dist_info as di


def _do_test(args: argparse.Namespace):
    import unittest

    # Start with required test modules.
    ALL_TEST_MODULES = [
        "rocm_sdk.tests.base_test",
        "rocm_sdk.tests.core_test",
    ]
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test modules if they are installed.
    target_family = di.determine_target_family()
    if di.ALL_PACKAGES["libraries"].has_py_package(target_family):
        ALL_TEST_MODULES.append("rocm_sdk.tests.libraries_test")
    else:
        print("NOTE: Skipping libraries tests (not installed for this arch)")

    # Load all tests.
    for test_module_name in ALL_TEST_MODULES:
        suite.addTests(loader.loadTestsFromName(test_module_name))
    if loader.errors:
        print(f"WARNING: Test discovery had errors: {loader.errors}")

    runner = unittest.TextTestRunner(stream=sys.stdout, verbosity=3)
    result = runner.run(suite)
    if not result.wasSuccessful():
        sys.exit(1)


def main(argv: list[str] | None = None):
    if argv is None:
        argv = sys.argv[1:]
    p = argparse.ArgumentParser(
        prog="rocm-sdk",
        usage="rocm-sdk {command} ...",
        description="ROCm SDK Python CLI",
    )
    sub_p = p.add_subparsers(required=True)
    test_p = sub_p.add_parser("test", help="Run installation tests to verify integrity")
    test_p.set_defaults(func=_do_test)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
