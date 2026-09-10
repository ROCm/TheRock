# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Pytest bootstrap: put the package and vendored ``cmake_parser`` on sys.path.

Runs before collection so ``from cmake_parser import ...`` in the tests resolves
to the vendored copy without an install.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
for _path in (_ROOT, _ROOT / "_vendor"):
    _entry = str(_path)
    if _entry not in sys.path:
        sys.path.insert(0, _entry)
