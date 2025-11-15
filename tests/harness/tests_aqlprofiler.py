#!/usr/bin/python3


class TestAQLProfiler:
    """This is an Pytest Test Suite Class to test AQLProfiler component of TheRock"""

    def test_aqlprofiler(self, orch, therock_path, result):
        """A Test case to verify AQLProfiler tests"""
        result.verdict = orch.runBinary(
            "./run_tests.sh",
            cwd=f"{therock_path}/share/hsa-amd-aqlprofile",
        )
        assert result.verdict
