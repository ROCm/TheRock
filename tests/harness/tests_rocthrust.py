#!/usr/bin/python3


class TestRocThrust:
    """This is an Pytest Test Suite Class to test RocThrust component of TheRock"""

    def test_rocthrust(self, orch, therock_path, result):
        """A Test case to verify rocthrust"""
        result.verdict = orch.runCtest(cwd=f"{therock_path}/bin/rocthrust")
        assert result.verdict
