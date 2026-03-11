#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared utilities for git mirror management in TheRock.

Provides common functionality used by both fetch_sources.py and
setup_git_mirrors.py for working with git mirror repositories.
"""

from urllib.parse import urlparse

# Environment variable for configuring mirror directory location
MIRROR_DIR_ENV = "THEROCK_GIT_MIRROR_DIR"


def url_to_mirror_relpath(url: str) -> str:
    """Convert a git URL to a relative mirror directory path.

    The mirror directory structure follows the URL's organization and
    repository name pattern, ensuring each unique repository has a
    distinct mirror location.

    Args:
        url: Git remote URL (e.g., "https://github.com/ROCm/llvm-project.git")

    Returns:
        Relative path for the mirror directory (e.g., "ROCm/llvm-project.git")

    Examples:
        >>> url_to_mirror_relpath("https://github.com/ROCm/llvm-project.git")
        'ROCm/llvm-project.git'
        >>> url_to_mirror_relpath("https://github.com/ROCm/rocm-libraries")
        'ROCm/rocm-libraries.git'
        >>> url_to_mirror_relpath("https://github.com/iree-org/iree.git")
        'iree-org/iree.git'
    """
    parsed = urlparse(url)
    repo_path = parsed.path.strip("/")
    if not repo_path.endswith(".git"):
        repo_path += ".git"
    return repo_path
