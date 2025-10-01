#!/usr/bin/env python3

"""Creates a list of tests to be skipped usable for pytest.

Can also create the inverse: a list of tests on which pytest should only be run on

- Can run standalone
- If called within python, use get_tests()
"""

import argparse
import sys

from .skip_test_autograd import skip_tests as skip_test_autograd
from .skip_test_binary_ufuncs import skip_tests as skip_test_binary_ufuncs
from .skip_test_cuda import skip_tests as skip_test_cuda
from .skip_test_nn import skip_tests as skip_test_nn
from .skip_test_torch import skip_tests as skip_test_torch
from .skip_test_unary_ufuncs import skip_tests as skip_test_unary_ufuncs


def create_list(amdgpu_family="", pytorch_version=""):
    selected_tests = []

    for tests in [
        skip_test_autograd,
        skip_test_binary_ufuncs,
        skip_test_cuda,
        skip_test_nn,
        skip_test_torch,
        skip_test_unary_ufuncs,
    ]:
        if len(tests["always"]) > 0:
            selected_tests += tests["always"]
        if not amdgpu_family == "":
            if amdgpu_family in tests["amdgpu_family"].keys():
                print(
                    "add specific tests for amdgpu_family",
                    amdgpu_family,
                    file=sys.stderr,
                )
                selected_tests += tests["amdgpu_family"][amdgpu_family]
        if not pytorch_version == "":
            if pytorch_version in tests["pytorch_version"].keys():
                print(
                    "add specific tests for pytorch version",
                    pytorch_version,
                    file=sys.stderr,
                )
                selected_tests += tests["pytorch_version"][pytorch_version]

    return selected_tests


def cmd_arguments(argv: list[str]):
    p = argparse.ArgumentParser(
        description="""
Prints a list of tests that should be skipped.
Output can be used with 'pytest -k <list>'
"""
    )
    p.add_argument(
        "--amdgpu_family",
        type=str,
        default="",
        required=False,
        help="""Amdgpu family (e.g. "gfx942").
Select (potentially) additional tests to be skipped based on the amdgpu family""",
    )
    p.add_argument(
        "--pytorch_version",
        type=str,
        default="",
        required=False,
        help="""Pytorch version (e.g. "2.7").
Select (potentially) additional tests to be skipped based on the Pytorch version""",
    )
    p.add_argument(
        "--include-tests",
        default=False,
        required=False,
        action=argparse.BooleanOptionalAction,
        help="""Overwrites the default behavior of this program and creates a list of tests that should be run.
Output can be used with 'pytest -k <list>'""",
    )
    args = p.parse_args(argv)
    return args


def get_tests(amdgpu_family="", pytorch_version="", create_skip_list=True):
    if create_skip_list == True:
        print("Creating list of tests to be skipped")
    else:
        print("Creating list of tests to be included")

    tests = create_list(amdgpu_family=amdgpu_family, pytorch_version=pytorch_version)

    if create_skip_list == True:  # skip list
        expr = "not " + " and not ".join(tests)
    else:  # include list
        expr = " or ".join(tests)

    return expr


if __name__ == "__main__":
    args = cmd_arguments(sys.argv[1:])

    tests = get_tests(args.amdgpu_family, args.pytorch_version, args.include_tests)
    print(tests)
