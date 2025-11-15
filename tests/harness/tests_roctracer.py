#!/usr/bin/python3


class TestRocTracer:
    """This is an Pytest Test Suite Class to test RocTracer component of TheRock"""

    def test_roctracer(self, orch, therock_path, result):
        """A Test case to verify RocTracer tests"""
        result.verdict = orch.runBinary(
            "./run_tests.sh",
            cwd=f"{therock_path}/share/roctracer",
        )
        assert result.verdict
