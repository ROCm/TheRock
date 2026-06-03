#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

sys.path.insert(
    0,
    os.fspath(Path(__file__).parents[1] / "test_executable_scripts"),
)

import libhipcxx_utils


class GetGpuArchitecturePortableTest(unittest.TestCase):
    @mock.patch.dict(os.environ, {"AMDGPU_TARGETS": "gfx1100,gfx1201"}, clear=True)
    @mock.patch("libhipcxx_utils.subprocess.run")
    def test_falls_back_to_amdgpu_targets_when_offload_arch_missing(self, mock_run):
        mock_run.side_effect = FileNotFoundError()

        self.assertEqual(
            libhipcxx_utils.get_gpu_architecture_portable("/tmp/rocm"),
            "gfx1100;gfx1201",
        )

    @mock.patch.dict(os.environ, {"AMDGPU_TARGETS": "gfx1100; gfx1201"}, clear=True)
    @mock.patch("libhipcxx_utils.subprocess.run")
    def test_falls_back_to_amdgpu_targets_when_offload_arch_empty(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=["offload-arch"],
            returncode=0,
            stdout="\n",
            stderr="",
        )

        self.assertEqual(
            libhipcxx_utils.get_gpu_architecture_portable("/tmp/rocm"),
            "gfx1100;gfx1201",
        )

    @mock.patch.dict(os.environ, {}, clear=True)
    @mock.patch("libhipcxx_utils.subprocess.run")
    def test_returns_none_without_offload_arch_or_targets(self, mock_run):
        mock_run.side_effect = FileNotFoundError()

        self.assertIsNone(libhipcxx_utils.get_gpu_architecture_portable("/tmp/rocm"))


if __name__ == "__main__":
    unittest.main()
