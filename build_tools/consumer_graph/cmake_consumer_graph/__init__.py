"""Static consumer graph prototype for TheRock CMake declarations."""

import sys as _sys
from pathlib import Path as _Path

# Make the vendored `cmake_parser` importable without a pip install; it lives in
# the sibling _vendor/ dir, prepended to sys.path before any submodule imports it.
_VENDOR = _Path(__file__).resolve().parent.parent / "_vendor"
if str(_VENDOR) not in _sys.path:
    _sys.path.insert(0, str(_VENDOR))
