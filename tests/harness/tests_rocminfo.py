#!/usr/bin/python3
import re


class TestRocmInfo:
    """This is an Pytest Test Suite Class to test RocmInfo component of TheRock"""

    def test_rocminfo(self, orch, therock_path, result):
        exprList = (
            r"ROCk module.*? is loaded",
            r"Name:\s+gfx",
            r"Vendor Name:\s+AMD",
            r"Device Type:\s+GPU",
            r"L2:\s+.*? KB",
        )
        result.verdict, result.failed, out = orch.runBinary(
            "./rocminfo",
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
