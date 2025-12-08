#!/usr/bin/env python3
"""PyTorch Test Skip List Generator.

Creates a list of tests to be skipped or the only ones to be run on for PyTorch's pytest.

Key Features
------------
- AMDGPU family-specific test filtering
- PyTorch version-specific test filtering
- Can generate both skip lists and include lists
- Supports standalone CLI usage and programmatic API

Module Structure
----------------
The module expects skip test definitions in the following files:
- generic.py: Common tests to skip across all configurations
- pytorch_<version>.py: Version-specific tests (e.g., pytorch_2.7.py)

For additional information have a look at the README.md file in this directory.

Usage Examples
--------------
Programmatic usage:
    # Create a list of tests to be skipped as they are known to be failing on gfx942 and PyTorch 2.7
    from create_skip_tests import get_tests
    skip_expr = get_tests(["gfx942"], "2.7", create_skip_list=True)
    # Use skip_expr with pytest: pytest -k "{skip_expr}"

    # Skip tests for multiple GPU families
    skip_expr = get_tests(["gfx1011", "gfx1012"], "2.7", create_skip_list=True)

Command-line usage:
    Run all test excluding known failures for gfx942 and PyTorch 2.7:
    $ python create_skip_tests.py --amdgpu-family gfx942 --pytorch-version 2.7

    Run tests excluding failures for multiple GPU families:
    $ python create_skip_tests.py --amdgpu-family "gfx1011,gfx1012" --pytorch-version 2.7

    Run all tests that are normally skipped for gfx942 and all pytorch versions:
    $ python create_skip_tests.py --amdgpu-family gfx942 --pytorch-version all --include-tests

"""

import argparse
import importlib.util
import os
from pathlib import Path
import platform
import sys
from typing import Dict, List


def import_skip_tests(pytorch_version: str = "") -> Dict[str, Dict]:
    """Dynamically load test skip definitions from configuration files.

    Loads skip test definitions from:
    - generic.py (always loaded)
    - pytorch_<version>.py (if version specified)
    - pytorch_*.py (all version files if version="all")

    Args:
        pytorch_version: PyTorch version string (e.g., "2.7", "all", or "").
            - "" (empty): Load only generic.py
            - "all": Load generic.py and all pytorch_*.py files
            - Specific version: Load generic.py and pytorch_<version>.py

    Returns:
        Dictionary mapping module names to their skip_tests dictionaries.
        Format: {"generic": {...}, "pytorch_2.7": {...}}

    """
    this_script_dir = Path(os.path.abspath(__file__)).parent

    files = [os.path.join(this_script_dir, "generic.py")]
    if pytorch_version == "all":
        files += list(this_script_dir.glob("pytorch_*.py"))
    elif pytorch_version:
        files += [this_script_dir / f"pytorch_{pytorch_version}.py"]

    dict_skip_tests = {}

    for full_path in files:
        # Get filename without .py extension
        module_name = Path(full_path).stem

        try:
            spec = importlib.util.spec_from_file_location(module_name, full_path)
            if spec is None or spec.loader is None:
                print(
                    f"[WARNING] Could not create module spec for {module_name} at {full_path}",
                    file=sys.stderr,
                )
                continue

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            dict_skip_tests[module_name] = getattr(module, "skip_tests")
        except (ImportError, FileNotFoundError, AttributeError) as ex:
            msg_pytorch = ""
            if "pytorch" in module_name:
                msg_pytorch = f" for PyTorch version {pytorch_version}"
            print(
                f"[WARNING] In create_skip_tests.py: Failed to import module {module_name}{msg_pytorch} "
                f"from {full_path}: {type(ex).__name__}: {ex}",
                file=sys.stderr,
            )
            # Continue processing other files instead of failing completely
            # This allows running tests even if some version-specific files are missing

    return dict_skip_tests


def create_list(
    amdgpu_family: list[str] = [],
    pytorch_version: str = "",
    platform: str = "",
) -> List[str]:
    """Create a list of test names based on filters.

    Aggregates test names from all applicable skip test definitions based on
    the specified AMDGPU family and PyTorch version.

    Args:
        amdgpu_family: Target AMDGPU families (e.g., ["gfx942", "gfx1151"]).
            Tests marked for this family will be included.
        pytorch_version: PyTorch version for filtering (e.g., "2.7", "all", "").
            Determines which pytorch_*.py files are loaded.

    Returns:
        List of unique test names that match the specified filters.
        Duplicates are automatically removed.

    Notes:
        - Always includes tests from the "common" filter
        - Includes tests from the specified amdgpu_family filter (if provided)


    Examples:
        >>> tests = create_list(["gfx942", "gfx1151"], "2.7")
        >>> # Returns: ["test_dropout", "test_conv2d", ...]
    """
    selected_tests = []

    # Define filters: always include "common", plus specific AMDGPU families
    filters = ["common"]
    filters += amdgpu_family
    filters += [platform.lower()] if platform else []

    # Load skip_tests from generic.py and (pytorch_<version> or "all" pytorch versions)
    dict_skip_tests = import_skip_tests(pytorch_version)

    # Loop over all loaded skip_tests dictionaries from the different pytorch versions
    for skip_test_module_name, skip_tests in dict_skip_tests.items():
        # Apply each filter (common, amdgpu_family)
        for filter_name in filters:
            if filter_name in skip_tests:
                # For each pytorch test module (e.g., test_nn, test_torch) add all the tests
                for pytorch_test_module in skip_tests[filter_name].keys():
                    selected_tests += skip_tests[filter_name][pytorch_test_module]

    # Remove duplicates and return
    return list(set(selected_tests))


def parse_arguments(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="""
Generates a list of tests to skip (or include) for pytest.
Output is a pytest -k expression that can be used directly with pytest.
"""
    )
    parser.add_argument(
        "--amdgpu-family",
        type=str,
        default="",
        required=False,
        help="""AMDGPU family (e.g. "gfx942" or "gfx942, gfx1151").
Select (potentially) additional tests to be skipped based on the amdgpu family""",
    )
    parser.add_argument(
        "--platform",
        type=str,
        default=platform.system(),
        required=False,
        help="""Platform (Linux or Windows) for platform-specific filtering""",
    )
    parser.add_argument(
        "--pytorch-version",
        type=str,
        default="",
        required=False,
        help="""PyTorch version (e.g. "2.7" or "all").
Select (potentially) additional tests to be skipped based on the PyTorch version.
'all' includes skip tests for all pytorch versions.""",
    )
    parser.add_argument(
        "--include-tests",
        default=False,
        required=False,
        action=argparse.BooleanOptionalAction,
        help="""Invert behavior: create a list of tests to include (run) instead of skip.
Output can be used with 'pytest -k <list>'""",
    )
    args = parser.parse_args(argv)
    return args


def get_tests(
    amdgpu_family: list[str] = [],
    pytorch_version: str = "",
    platform: str = "",
    create_skip_list: bool = True,
) -> str:
    """Generate a pytest -k expression for test filtering.

    This is the main API function for programmatic use. It creates a pytest -k
    compatible expression that either skips or includes the specified tests.

    Args:
        amdgpu_family: List of target AMDGPU families (e.g., ["gfx942"], ["gfx1011", "gfx1012"]).
            Determines which family-specific tests to filter. Tests matching any of the
            specified families will be included in the filter.
        pytorch_version: PyTorch version (e.g., "2.7", "all", "").
            Determines which version-specific test files to load.
        create_skip_list: If True, create a skip list (default).
            If False, create an include list (only run specified tests).

    Returns:
        A pytest -k compatible expression string.
        - Skip list format: "not test1 and not test2 and not test3"
        - Include list format: "test1 or test2 or test3"

    """
    list_type = "skipped" if create_skip_list else "included"
    print(
        f"Creating list of tests to be {list_type} for AMDGPU family '{amdgpu_family}' "
        f"and PyTorch version '{pytorch_version}'... ",
        end="",
    )

    # Get the list of test names
    tests = create_list(
        amdgpu_family=amdgpu_family, pytorch_version=pytorch_version, platform=platform
    )

    # Format as pytest -k expression
    if create_skip_list:
        # Skip list: "not test1 and not test2 and not test3"
        expr = "not " + " and not ".join(tests)
    else:
        # Include list: "test1 or test2 or test3"
        expr = " or ".join(tests)

    print("done")
    return expr


if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])

    amdgpu_family = [
        family.strip() for family in args.amdgpu_family.split(",") if args.amdgpu_family
    ]

    tests = get_tests(
        amdgpu_family=amdgpu_family,
        pytorch_version=args.pytorch_version,
        platform=args.platform,
        create_skip_list=not args.include_tests,
    )
    print(tests)
