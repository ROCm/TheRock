"""Trampoline for console scripts."""

import importlib
import os
import sys
from pathlib import Path

from ._dist_info import ALL_PACKAGES

CORE_PACKAGE = ALL_PACKAGES["core"]
PLATFORM_NAME = CORE_PACKAGE.get_py_package_name()
PLATFORM_MODULE = importlib.import_module(PLATFORM_NAME)
# NOTE: dependent on there being an __init__.py in the platform package.
PLATFORM_PATH = Path(PLATFORM_MODULE.__file__).parent


def _exec(relpath: str):
    full_path = PLATFORM_PATH / relpath
    os.execv(full_path, [str(full_path)] + sys.argv[1:])
