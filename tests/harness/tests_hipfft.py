#!/usr/bin/python3


class TestHipFFT:
    """This is an Pytest Test Suite Class to test HipFFT component of TheRock"""

    def test_hipfft(self, orch, ompEnv, therock_path, ldpathEnv, result):
        """A Test case to verify HipFFT tests"""
        result.verdict = orch.runGtest(
            "./hipfft-test",
            env={
                **ldpathEnv,
                **ompEnv,
            },
            cwd=f"{therock_path}/bin",
        )
        assert result.verdict
