#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for _therock_utils.git_mirrors module."""

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from _therock_utils.git_mirrors import MIRROR_DIR_ENV, url_to_mirror_relpath


class UrlToMirrorRelpathTest(unittest.TestCase):
    """Tests for url_to_mirror_relpath."""

    def test_url_with_git_suffix(self):
        result = url_to_mirror_relpath("https://github.com/ROCm/llvm-project.git")
        self.assertEqual(result, "ROCm/llvm-project.git")

    def test_url_without_git_suffix(self):
        result = url_to_mirror_relpath("https://github.com/ROCm/rocm-libraries")
        self.assertEqual(result, "ROCm/rocm-libraries.git")

    def test_different_org(self):
        result = url_to_mirror_relpath("https://github.com/iree-org/iree.git")
        self.assertEqual(result, "iree-org/iree.git")

    def test_strips_leading_trailing_slashes(self):
        result = url_to_mirror_relpath("https://github.com/ROCm/half.git")
        self.assertFalse(result.startswith("/"))

    def test_deeply_nested_path(self):
        result = url_to_mirror_relpath("https://example.com/org/sub/repo.git")
        self.assertEqual(result, "org/sub/repo.git")

    def test_ssh_style_url(self):
        result = url_to_mirror_relpath("ssh://git@github.com/ROCm/hip.git")
        self.assertEqual(result, "ROCm/hip.git")


class MirrorDirEnvTest(unittest.TestCase):
    """Tests for the MIRROR_DIR_ENV constant."""

    def test_env_var_name(self):
        self.assertEqual(MIRROR_DIR_ENV, "THEROCK_GIT_MIRROR_DIR")


if __name__ == "__main__":
    unittest.main()
