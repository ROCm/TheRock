#!/usr/bin/python3


class TestRocAlution:
    """This is an Pytest Test Suite Class to test RocAlution component of TheRock"""

    def test_rocalution(self, orch, ompEnv, therock_path, result):
        """A Test case to verify RocAlution tests"""
        result.verdict = orch.runGtest(
            "./rocalution-test",
            env=ompEnv,
            cwd=f"{therock_path}/bin",
        )
        assert result.verdict
