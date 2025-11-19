#!/usr/bin/python3


class TestHipBlas:
    """This is an Pytest Test Suite Class to test HipBlas component of TheRock"""

    def test_hipblas(self, orch, ompEnv, therock_path, result):
        """A Test case to verify HipBlas tests"""
        result.verdict = orch.runGtest(
            "./hipblas-test",
            env=ompEnv,
            cwd=f"{therock_path}/bin",
        )
        assert result.verdict
