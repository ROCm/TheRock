# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from build_prod_wheels import (
    _AOTRITON_ARCH_ARG,
    _AOTRITON_BASE_ARCH_ARG,
    _AOTRITON_BASE_ARCH_BLOCK,
    patch_aotriton_target_arch,
)

_UPSTREAM = """\
    message(STATUS "PYTORCH_ROCM_ARCH ${PYTORCH_ROCM_ARCH}")
    ExternalProject_Add(aotriton
      CMAKE_ARGS
      -DAOTRITON_TARGET_ARCH:STRING=${PYTORCH_ROCM_ARCH}
    )
"""


class PatchAotritonTargetArchTest(unittest.TestCase):
    def test_strips_suffixes_for_aotriton_only(self):
        with TemporaryDirectory() as tmp:
            cmake_path = Path(tmp) / "cmake" / "External" / "aotriton.cmake"
            cmake_path.parent.mkdir(parents=True)
            cmake_path.write_text(_UPSTREAM)

            patch_aotriton_target_arch(Path(tmp))

            text = cmake_path.read_text()
            self.assertIn(_AOTRITON_BASE_ARCH_BLOCK, text)
            self.assertIn(_AOTRITON_BASE_ARCH_ARG, text)
            self.assertNotIn(_AOTRITON_ARCH_ARG, text)
            self.assertIn('string(REGEX REPLACE ":.*$" ""', text)

    def test_is_idempotent(self):
        with TemporaryDirectory() as tmp:
            cmake_path = Path(tmp) / "cmake" / "External" / "aotriton.cmake"
            cmake_path.parent.mkdir(parents=True)
            cmake_path.write_text(_UPSTREAM)

            patch_aotriton_target_arch(Path(tmp))
            once = cmake_path.read_text()
            patch_aotriton_target_arch(Path(tmp))
            self.assertEqual(cmake_path.read_text(), once)

    def test_missing_cmake_is_a_noop(self):
        with TemporaryDirectory() as tmp:
            patch_aotriton_target_arch(Path(tmp))

    def test_moved_upstream_text_is_an_error(self):
        with TemporaryDirectory() as tmp:
            cmake_path = Path(tmp) / "cmake" / "External" / "aotriton.cmake"
            cmake_path.parent.mkdir(parents=True)
            cmake_path.write_text("message(STATUS unrelated)\n")

            with self.assertRaises(RuntimeError):
                patch_aotriton_target_arch(Path(tmp))


if __name__ == "__main__":
    unittest.main()
