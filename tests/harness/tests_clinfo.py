#!/usr/bin/python3
import re


class TestCLInfo:
    """This is an Pytest Test Suite Class to test CLInfo of TheRock"""

    def test_clinfo(self, orch, therock_path, result):
        exprList = (
            r"Number of devices:\s+\d",
            r"Device Type:\s+.*?GPU",
            r"Board name:\s+AMD",
            r"Max compute units:\s+\d+",
            r"Name:\s+gfx",
            r"Vendor:\s+Advanced Micro Devices, Inc",
            r"Extensions:\s+cl_",
            r"Version:\s+OpenCL",
        )
        result.verdict, result.failed, out = orch.runBinary(
            "./clinfo",
            cwd=f"{therock_path}/bin",
        )
        result.verdict = all(
            (
                result.verdict,
                *[
                    re.search(expr, out) or log(f"Expr Not Match: {expr}")
                    for expr in exprList
                ],
            ),
        )
        assert result.verdict
