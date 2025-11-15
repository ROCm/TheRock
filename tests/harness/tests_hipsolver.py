#!/usr/bin/python3


class TestHipSolver:
    """This is an Pytest Test Suite Class to test HipSolver component of TheRock"""

    def test_hipsolver(self, orch, ompEnv, therock_path, result):
        """A Test case to verify HipSolver tests"""
        result.verdict = orch.runGtest(
            "./hipsolver-test",
            gfilter="-*known_bug*",
            env=ompEnv,
            cwd=f"{therock_path}/bin",
        )
        assert result.verdict
