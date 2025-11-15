#!/usr/bin/python3


class TestCK:
    """This is an Pytest Test Suite Class to test CK component of TheRock"""

    @pytest.mark.parametrize(
        argnames=("tensor"),
        ids=(
            "gemm_splitk_1152_32",
            "gemm_splitk_5210_4",
            "gemm_splitk_1280_32",
            "gemm_splitk_5120_8",
        ),
        argvalues=(
            "gemm_splitk 1 0 1 1 0 0 16 1152 5120 5120 1152 1152 32",
            "gemm_splitk 1 0 1 1 0 0 16 5120 384 384 5120 5120 4",
            "gemm_splitk 1 0 1 1 0 0 16 1280 5120 5120 1280 1280 32",
            "gemm_splitk 1 0 1 1 0 0 16 5120 1280 1280 5120 5120 8",
        ),
    )
    def test_ck(self, tensor, orch, therock_path, result):
        """A Test case to verify Composable kernel tests"""
        result.verdict = orch.runBinary(
            "./ckProfiler",
            tensor,
            cwd=f"{therock_path}/bin",
        )
        assert result.verdict
