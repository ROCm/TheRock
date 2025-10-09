#!/usr/bin/env python3

"""Creates a list of tests to be skipped usable for pytest.

Can also create the inverse: a list of tests on which pytest should only be run on

- Can run standalone
- If called within python, use get_tests()
"""

import argparse
import sys

import glob
import importlib.util
import os


def import_skip_tests(pytorch_version=""):
    """
    Dynamic loading of all files required for skipping tests. This includes,
    skip_tests/generic.py and all skip_tests/pytorch_<version>.py files
    """

    this_script_dir = os.path.dirname(os.path.abspath(__file__))

    files = [os.path.join(this_script_dir, "generic.py")]
    if pytorch_version == "all":
        files += glob.glob(os.path.join(this_script_dir, "pytorch_*.py"))
    elif not pytorch_version == "":
        files += [os.path.join(this_script_dir, f"pytorch_{pytorch_version}.py")]

    dict_skipt_tests = {}

    for full_path in files:
        # get filename without .py extension
        module_name = os.path.basename(full_path)[:-3]

        try:
            spec = importlib.util.spec_from_file_location(module_name, full_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            dict_skipt_tests[module_name] = getattr(module, "skip_tests")
        except (ImportError, FileNotFoundError, AttributeError) as ex:
            msg_pytorch = ""
            if "pytorch" in module_name:
                msg_pytorch = f" and given pytorch_version {pytorch_version}"
            print(
                f"Create_skip_tests.py: Failed to import module {module_name}{msg_pytorch} with path {full_path} : {ex}",
                file=sys.stderr,
            )
            # sys.exit(1)  # TODO do we want to exit? means each new pytorch version we would have to add a new file

    return dict_skipt_tests


def create_list(amdgpu_family="", pytorch_version=""):
    selected_tests = []

    filters = ["common"]
    filters += [amdgpu_family]

    # load skip_tests only generic and (pytorch_<version> or "all" pytorch versions)
    dict_skipt_tests = import_skip_tests(pytorch_version)

    # loop over skip_tests of generic.py and pytorch_<version>.py
    for skip_test_module_name, skip_tests in dict_skipt_tests.items():
        for filter in filters:
            if filter in skip_tests.keys():
                for pytorch_test_module in skip_tests[filter].keys():
                    selected_tests += skip_tests[filter][pytorch_test_module]

    # remove duplicates
    selected_tests = list(set(selected_tests))
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
        help="""Pytorch version (e.g. "2.7" or "all).
Select (potentially) additional tests to be skipped based on the Pytorch version.
'All' is also possible. Then all skip tests for all pytorch versions are included.""",
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
        print(
            f"Creating list of tests to be skipped for amdgpu_family {amdgpu_family} and PyTorch verison {pytorch_version}... ",
            end="",
        )
    else:
        print(
            "Creating list of tests to be included for amdgpu_family {amdgpu_family} and PyTorch verison {pytorch_version}... ",
            end="",
        )

    tests = create_list(amdgpu_family=amdgpu_family, pytorch_version=pytorch_version)

    if create_skip_list == True:  # skip list
        expr = "not " + " and not ".join(tests)
    else:  # include list
        expr = " or ".join(tests)

    print("done")
    return expr


if __name__ == "__main__":
    args = cmd_arguments(sys.argv[1:])

    tests = get_tests(args.amdgpu_family, args.pytorch_version, args.include_tests)
    print(tests)
