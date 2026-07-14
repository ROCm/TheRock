# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Sanity checks for the PyTorch skip-test definition files.

Each ``generic.py`` / ``pytorch_<version>.py`` file in this directory must
define a ``skip_tests`` dict in the shape that ``create_skip_tests.py``
consumes: ``section -> pytorch_test_module -> iterable[str]``. These tests guard
against typos and structural mistakes when a new version skip list (e.g.
``pytorch_2.13.py``) is added.
"""

import importlib.util
from pathlib import Path

SKIP_DIR = Path(__file__).parent


def _skip_list_files():
    files = sorted(SKIP_DIR.glob("pytorch_*.py"))
    generic = SKIP_DIR / "generic.py"
    if generic.exists():
        files.append(generic)
    return [f for f in files if f.name != "create_skip_tests.py"]


def _load_skip_tests(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "skip_tests", None)


def test_skip_files_define_skip_tests_dict():
    files = _skip_list_files()
    assert files, "expected at least one skip-list file"
    for path in files:
        skip_tests = _load_skip_tests(path)
        assert isinstance(skip_tests, dict), f"{path.name}: skip_tests must be a dict"
        assert skip_tests, f"{path.name}: skip_tests must not be empty"


def test_skip_tests_entries_are_well_formed():
    for path in _skip_list_files():
        skip_tests = _load_skip_tests(path)
        for section, modules in skip_tests.items():
            assert isinstance(
                section, str
            ), f"{path.name}: section {section!r} not a str"
            assert isinstance(
                modules, dict
            ), f"{path.name}:{section} must map to a dict"
            for module_name, tests in modules.items():
                assert isinstance(
                    module_name, str
                ), f"{path.name}:{section} module {module_name!r} not a str"
                assert isinstance(
                    tests, (list, set, tuple)
                ), f"{path.name}:{section}.{module_name} must be a collection"
                for name in tests:
                    assert (
                        isinstance(name, str) and name
                    ), f"{path.name}:{section}.{module_name} bad entry {name!r}"


if __name__ == "__main__":
    test_skip_files_define_skip_tests_dict()
    test_skip_tests_entries_are_well_formed()
    print("All skip-test sanity checks passed.")
