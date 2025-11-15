#!/usr/bin/python3


class TestAmdSMI:
    """This is an Pytest Test Suite Class to test AmdSMI component of TheRock"""

    def test_amdsmi(self, orch, therock_path, result):
        """A Test case to verify AmdSMI tests"""
        result.verdict = orch.runGtest(
            "./amdsmitst",
            srcFile="amdsmitst.exclude",
            gfilter="-${BLACKLIST_ALL_ASICS}",
            cwd=f"{therock_path}/share/amd_smi/tests",
        )
        assert result.verdict
