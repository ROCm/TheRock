#!/usr/bin/python3


class TestMIGraphx:
    """This is an Pytest Test Suite Class to test MIGraphx component of TheRock"""

    def test_migraphx(self, orch, therock_path, result):
        """A Test case to verify MIGraphx tests"""
        result.verdict = orch.runCtest(
            "-DONNX_USE_PROTOBUF_SHARED_LIBS=ON",
            env={
                "LD_LIBRARY_PATH": f"{therock_path}/lib:$LD_LIBRARY_PATH",
            },
            cwd=f"{therock_path}/libexec/installed-tests/migraphx",
        )
        assert result.verdict
