#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Compatibility CLI trampoline for :mod:`therock_tools.artifact_manager`."""

import os
from pathlib import Path
import sys


_THEROCK_TOOLS_SRC = (
    Path(__file__).resolve().parent.parent / "python" / "therock-tools" / "src"
)
sys.path.insert(0, os.fspath(_THEROCK_TOOLS_SRC))

from therock_tools.artifact_manager import main

# Keep source-tree invocations compatible; installed console scripts require
# callers to pass --topology explicitly.
_DEFAULT_TOPOLOGY_PATH = Path(__file__).resolve().parent.parent / "BUILD_TOPOLOGY.toml"
_COMMANDS_WITH_TOPOLOGY = {"fetch", "push", "copy", "info", "list-stages"}


def _has_topology_arg(argv: list[str]) -> bool:
    return any(arg == "--topology" or arg.startswith("--topology=") for arg in argv)


def _with_default_topology(argv: list[str]) -> list[str]:
    if _has_topology_arg(argv):
        return argv
    if not any(arg in _COMMANDS_WITH_TOPOLOGY for arg in argv):
        return argv
    return [*argv, f"--topology={_DEFAULT_TOPOLOGY_PATH}"]


if __name__ == "__main__":
    main(_with_default_topology(sys.argv[1:]))
