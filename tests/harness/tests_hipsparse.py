#!/usr/bin/python3


class TestHipSparse:
    """This is an Pytest Test Suite Class to test HipSparse component of TheRock"""

    def test_hipsparse(self, orch, ompEnv, therock_path, result):
        """A Test case to verify HipSparse tests"""
        result.verdict = orch.runGtest(
            "./hipsparse-test",
            env={
                "HIPSPARSE_CLIENTS_MATRICES_DIR": f"{therock_path}/clients/matrices",
                **ompEnv,
            },
            cwd=f"{therock_path}/bin",
        )
        assert result.verdict
