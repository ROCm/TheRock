#!/usr/bin/python3


class TestMIOpen:
    """This is an Pytest Test Suite Class to test MIOpen component of TheRock"""

    def test_miopen(self, orch, therock_path, result):
        """A Test case to verify MIOpen tests"""
        result.verdict = orch.runGtest(
            "./miopen_gtest",
            gfilter="-DBSync:DeepBench:MIOpenTestConv",
            cwd=f"{therock_path}/bin",
        )
        assert result.verdict
