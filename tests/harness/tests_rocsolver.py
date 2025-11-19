#!/usr/bin/python3


class TestRocSolver:
    """This is an Pytest Test Suite Class to test RocSolver component of TheRock"""

    def test_rocsolver(self, orch, ompEnv, rocsolverTestDir, result):
        """A Test case to verify RocSolver tests"""
        result.verdict = orch.runGtest(
            "./rocsolver-test", env=ompEnv, cwd=f"{therock_path}/bin"
        )
        assert result.verdict
