#!/usr/bin/python3


class TestRocr:
    """This is an Pytest Test Suite Class to test Rocr component of TheRock"""

    def test_rocrtst(self, orch, rocm, rocrTestDir, result):
        """A Test case to verify Rocr tests"""
        result.verdict = orch.runGtest(
            "./rocrtst64",
            gfilter="-*DISABLED_*",
            env={
                "LD_LIBRARY_PATH": f"{rocm}/lib/rocrtst/lib:{rocm}/lib:$LD_LIBRARY_PATH",
            },
            cwd=f"{therock_path}/bin",
        )
        assert result.verdict
