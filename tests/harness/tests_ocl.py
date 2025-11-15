#!/usr/bin/python3


class TestOCL:
    """This is an Pytest Test Suite Class to test OCL component of TheRock"""

    def test_ocl(self, orch, therock_path, result):
        """A Test case to verify ocl tests"""
        result.verdict, result.failed, out = orch.runBinary(
            "./ocltst",
            "-m",
            "liboclruntime.so",
            "-A",
            "oclruntime.exclude",
            cwd=f"{therock_path}/share/opencl/ocltst",
        )
        assert result.verdict
