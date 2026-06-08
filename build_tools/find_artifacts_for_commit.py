#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Compatibility CLI trampoline for :mod:`therock_tools.find_artifacts_for_commit`."""

import os
from pathlib import Path
import sys


_THEROCK_TOOLS_SRC = (
    Path(__file__).resolve().parent.parent / "python" / "therock-tools" / "src"
)
sys.path.insert(0, os.fspath(_THEROCK_TOOLS_SRC))

from therock_tools.find_artifacts_for_commit import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
