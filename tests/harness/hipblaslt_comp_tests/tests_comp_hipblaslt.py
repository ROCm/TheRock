#!/usr/bin/python3
import os
import json
import pytest


@pytest.fixture(scope="session")
def hipblasltBenchDir(orch, therock_path):
    """Fixture to compile hipblaslt benchmark tests"""
    buildDir = "hipblaslt/build"
    os.makedirs(buildDir, exist_ok=True)
    ret, _, _ = orch.node.runCmd('cmake', '..',
        env={'ROCKDIR': therock_path},
        cwd=buildDir,
    )
    assert ret == 0
    yield buildDir


class TestHipBLASLT:
    """This is an Pytest Test Suite Class to test hipblaslt component of TheRock"""

    def test_hipblaslt_bench_sharded(self, orch, therock_path, hipblasltBenchDir, result):
        """A Test case to verify hipblaslt benchmark tests"""
        result.testVerdict = orch.runCtestShards(cwd=hipblasltBenchDir,
            env={'ROCKDIR': therock_path},
        )
        assert result.testVerdict


    def test_hipblaslt_bench_scheduler(self, orch, therock_path, hipblasltBenchDir, result):
        """A Test case to verify hipblaslt benchmark tests"""
        result.testVerdict = orch.runCtestScheduler(cwd=hipblasltBenchDir,
            env={'ROCKDIR': therock_path},
        )
        assert result.testVerdict
