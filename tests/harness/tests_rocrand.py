#!/usr/bin/python3


class TestRocRand:
    """This is an Pytest Test Suite Class to test RocRand component of TheRock"""

    def test_rocrand(self, orch, therock_path, result):
        """A Test case to verify rocrand"""
        result.verdict = orch.runCtest(cwd=f"{therock_path}/bin/rocRAND")
        assert result.verdict
