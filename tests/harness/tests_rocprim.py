#!/usr/bin/python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT


class TestRocPrim:
    """This is an Pytest Test Suite Class to test RocPrim component of TheRock"""

    def test_rocprim(self, orch, therock_path, result):
        """A Test case to verify rocprim"""
        result.testVerdict = orch.runCtest(cwd=f"{therock_path}/bin/rocprim")
        assert result.testVerdict
