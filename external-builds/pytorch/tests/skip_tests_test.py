# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Sanity checks for the PyTorch skip-test definition files.

Each ``generic.py`` / ``pytorch_<version>.py`` file in ``../skip_tests/`` must
define a ``skip_tests`` dict in the shape that ``create_skip_tests.py``
consumes: ``section -> pytorch_test_module -> iterable[str]``. These tests guard
against typos and structural mistakes when a new version skip list (e.g.
``pytorch_2.13.py``) is added.
"""

import importlib.util
import unittest
from pathlib import Path

SKIP_DIR = Path(__file__).resolve().parent.parent / "skip_tests"


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


class SkipTestsTest(unittest.TestCase):
    def test_skip_files_define_skip_tests_dict(self):
        files = _skip_list_files()
        self.assertTrue(files, "expected at least one skip-list file")
        for path in files:
            with self.subTest(path=path.name):
                skip_tests = _load_skip_tests(path)
                self.assertIsInstance(skip_tests, dict)
                self.assertTrue(skip_tests, "skip_tests must not be empty")

    def test_skip_tests_entries_are_well_formed(self):
        files = _skip_list_files()
        self.assertTrue(files, "expected at least one skip-list file")
        for path in files:
            with self.subTest(path=path.name):
                skip_tests = _load_skip_tests(path)
                self.assertIsInstance(skip_tests, dict)
                for section, modules in skip_tests.items():
                    with self.subTest(section=section):
                        self.assertIsInstance(section, str)
                        self.assertIsInstance(modules, dict)
                        for module_name, tests in modules.items():
                            with self.subTest(module=module_name):
                                self.assertIsInstance(module_name, str)
                                self.assertIsInstance(tests, (list, set, tuple))
                                for name in tests:
                                    self.assertIsInstance(name, str)
                                    self.assertTrue(name, "test name must not be empty")


if __name__ == "__main__":
    unittest.main()
