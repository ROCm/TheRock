#!/usr/bin/python3


class TestRsmi:
    """This is an Pytest Test Suite Class to test Rsmi component of TheRock"""

    def test_rsmi(self, orch, therock_path, result):
        """A Test case to verify Rsmi tests"""
        skipTests = (
            "rsmitstReadOnly.TestVoltCurvRead",
            "rsmitstReadWrite.TestPerfLevelReadWrite",
            "rsmitstReadWrite.TestFrequenciesReadWrite",
            "rsmitstReadWrite.TestPciReadWrite",
            "rsmitstReadWrite.TestPowerCapReadWrite",
            "rsmitstReadWrite.TestPerfDeterminism",
        )
        result.verdict = orch.runGtest(
            "./rsmitst",
            gfilter=f"-{':'.join(skipTests)}",
            env={
                "LD_LIBRARY_PATH": ".:$LD_LIBRARY_PATH",
            },
            cwd=f"{therock_path}/share/rocm_smi/rsmitst_tests",
        )
        assert result.verdict
