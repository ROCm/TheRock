#!/usr/bin/python3
import pytest


class TestRVS:
    """This is an Pytest Test Suite Class to test RVS component of TheRock"""

    @pytest.mark.parametrize(
        argnames=("suite"),
        argvalues=(
            "iet",
            "gst",
        ),
    )
    def test_rvs(self, suite, orch, therock_path, result):
        """A Test case to verify RVS tests"""
        result.verdict, result.failed, out = orch.runBinary(
            "./rvs",
            "-c",
            f"{therock_path}/share/rocm-validation-suite/conf/{suite}_single.conf",
            "-d",
            "3",
            cwd=f"{therock_path}/bin",
        )
        assert result.verdict
