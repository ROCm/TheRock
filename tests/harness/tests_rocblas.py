#!/usr/bin/python3


class TestRocBlas:
    """This is an Pytest Test Suite Class to test RocBlas component of TheRock"""

    def test_rocblas(self, orch, ompEnv, rocblasTestDir, result):
        """A Test case to verify RocBlas tests"""
        result.verdict = orch.runGtest(
            "./rocblas-test",
            gfilter="*single_gpu*:-*known_bugs*",
            # *multi_gpu*:-*known_bugs* => tdb when multigpu node is available
            trackSuites=True,
            env=ompEnv,
            cwd=f"{therock_path}/bin",
        )
        assert result.verdict
