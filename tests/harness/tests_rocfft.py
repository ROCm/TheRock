#!/usr/bin/python3


class TestRocFFT:
    """This is an Pytest Test Suite Class to test RocFFT component of TheRock"""

    def test_rocfft(self, orch, ompEnv, rocfftTestDir, result):
        """A Test case to verify RocFFT tests"""
        result.verdict = orch.runGtest(
            "./rocfft-test",
            gfilter="*single_gpu*",
            # *multi_gpu* => tdb when multigpu setup is available
            env=ompEnv,
            cwd=f"{therock_path}/bin",
        )
        assert result.verdict
