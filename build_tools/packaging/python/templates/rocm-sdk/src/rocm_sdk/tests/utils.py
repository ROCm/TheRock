"""Test utilities."""

from pathlib import Path


def assert_is_physical_package(mod):
    """Asserts that the given module is a non namespace module on disk defined
    by an __init__.py file."""
    assert (
        mod.__file__ is not None
    ), f"The `{mod.__name__}` module does not exist as a physical directory (__file__ is None)"
    assert (
        Path(mod.__file__).name == "__init__.py"
    ), f"Expected `{mod.__name__}` to be a non-namespace package"
