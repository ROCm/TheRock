# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Pytest bootstrap for the consumer-graph package.

Makes the package (``cmake_consumer_graph``) and the vendored ``cmake_parser``
importable without an install, so the tests exercise the checked-in bytes.
pytest loads this conftest before collecting the test modules, so imports like
``from cmake_parser import ...`` in the test files resolve to the vendored copy.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
for _path in (_ROOT, _ROOT / "_vendor"):
    _entry = str(_path)
    if _entry not in sys.path:
        sys.path.insert(0, _entry)
