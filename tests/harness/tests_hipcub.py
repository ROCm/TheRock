#!/usr/bin/python3


class TestHipCub:
    """This is an Pytest Test Suite Class to test hipcub component of TheRock"""

    def test_hipcub(self, orch, therock_path, result):
        """A Test case to verify hipcub"""
        result.verdict = orch.runCtest(cwd=f"{therock_path}/bin/hipcub")
        assert result.verdict
