#!/usr/bin/python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT


class TestRocRand:
    """This is an Pytest Test Suite Class to test RocRand component of TheRock"""

    def test_rocrand(self, orch, therock_path, result):
        """A Test case to verify rocrand"""
        result.testVerdict = orch.runCtest(cwd=f"{therock_path}/bin/rocRAND")
        assert result.testVerdict
