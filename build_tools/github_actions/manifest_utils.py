#!/usr/bin/env python3
"""
Helpers for manifest generation.
"""


def normalize_python_version_for_filename(python_version: str) -> str:
    """Normalize python version strings for filenames.

    Examples:
      "py3.12" -> "3.12"
      "3.12"   -> "3.12"
    """
    py = python_version.strip()
    if py.startswith("py"):
        py = py[2:]
    return py


def normalize_ref_for_filename(ref: str) -> str:
    """Normalize a git ref for filenames by replacing '/' with '-'.

    Examples:
      "nightly"                -> "nightly"
      "release/0.4.28"         -> "release-0.4.28"
      "users/alice/experiment" -> "users-alice-experiment"
    """
    return ref.replace("/", "-")
