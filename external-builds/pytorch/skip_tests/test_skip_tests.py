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


def _load_create_skip_tests():
    path = SKIP_DIR / "create_skip_tests.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_gfx1103_skips_apply_only_to_gfx1103():
    """Arch-scoped sections must not leak into sibling archs.

    The ``gfx1103`` section skips grouped/depthwise conv + RNN tests that only
    fail on that arch. ``create_list`` matches a section against a family via
    substring (``section in family``), so a section key that is a prefix of
    another arch could over-match. Guard that gfx1103's skips are selected for
    gfx1103 yet stay out of the other gfx110X archs and unrelated arches.
    """
    generic = _load_skip_tests(SKIP_DIR / "generic.py")
    gfx1103_tests = set(generic["gfx1103"]["nn"])
    assert gfx1103_tests, "expected gfx1103 skip entries in generic.py"

    create_list = _load_create_skip_tests().create_list

    selected = set(create_list(amdgpu_family=["gfx1103"]))
    assert gfx1103_tests <= selected, "gfx1103 skips must apply to gfx1103"

    for other in ("gfx1100", "gfx1101", "gfx1102", "gfx942"):
        selected_other = set(create_list(amdgpu_family=[other]))
        leaked = gfx1103_tests & selected_other
        assert not leaked, f"gfx1103 skips leaked into {other}: {sorted(leaked)}"


if __name__ == "__main__":
    test_skip_files_define_skip_tests_dict()
    test_skip_tests_entries_are_well_formed()
    test_gfx1103_skips_apply_only_to_gfx1103()
    print("All skip-test sanity checks passed.")
