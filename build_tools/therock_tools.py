# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""In-tree import shim for the extracted therock_tools package.

When running scripts from build_tools/, Python puts build_tools/ on sys.path.
This module makes the source package under python/therock-tools/src importable
without installing a wheel first.
"""

from pathlib import Path


_PACKAGE_DIR = (
    Path(__file__).resolve().parent.parent
    / "python"
    / "therock-tools"
    / "src"
    / "therock_tools"
)
if not _PACKAGE_DIR.exists():
    raise ModuleNotFoundError(
        f"therock_tools package directory not found: {_PACKAGE_DIR}"
    )

# Make this shim module behave like the extracted package for submodule imports.
# __file__ is compatibility metadata for callers that locate package resources.
__path__ = [str(_PACKAGE_DIR)]
__file__ = str(_PACKAGE_DIR / "__init__.py")
