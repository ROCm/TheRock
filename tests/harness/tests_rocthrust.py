#!/usr/bin/python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT


class TestRocThrust:
    """This is an Pytest Test Suite Class to test RocThrust component of TheRock"""

    def test_rocthrust(self, orch, therock_path, result):
        """A Test case to verify rocthrust"""
        result.testVerdict = orch.runCtest(cwd=f"{therock_path}/bin/rocthrust")
        assert result.testVerdict
