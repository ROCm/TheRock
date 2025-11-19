#!/usr/bin/python3


class TestRBT:
    """This is an Pytest Test Suite Class to test Rocm Bandwidth Tests component of TheRock"""

    def test_rbt(self, orch, therock_path, result):
        """A Test case to verify Rocm Bandwidth Tests"""
        result.verdict = orch.runBinary(
            "./rocm-bandwidth-test", cwd=f"{therock_path}/bin"
        )
        assert result.verdict
