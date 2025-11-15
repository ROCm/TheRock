#!/usr/bin/python3


class TestRCCL:
    """This is an Pytest Test Suite Class to test RCCL component of TheRock"""

    def test_rccl(self, orch, therock_path, result):
        """A Test case to verify RCCL tests"""
        result.verdict = orch.runGtest(
            "./rccl-UnitTests",
            cwd=f"{therock_path}/bin",
        )
        assert result.verdict
