#!/usr/bin/python3


class TestHipBlasLT:
    """This is an Pytest Test Suite Class to test HipBlasLT component of TheRock"""

    def test_hipblaslt(self, orch, ompEnv, therock_path, ldpathEnv, result):
        """A Test case to verify HipBlasLT tests"""
        result.verdict = orch.runGtest(
            "./hipblaslt-test",
            env={
                **ldpathEnv,
                **ompEnv,
            },
            cwd=f"{therock_path}/bin",
        )
        assert result.verdict
