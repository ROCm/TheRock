# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Pytest bootstrap: put the package directory on sys.path.

Runs before collection so ``import cmake_consumer_graph`` resolves without an
editable install. ``cmake_parser`` is a normal dependency from requirements-test.txt.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_entry = str(_ROOT)
if _entry not in sys.path:
    sys.path.insert(0, _entry)
