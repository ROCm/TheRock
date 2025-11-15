#!/usr/bin/python3

import logging
from libs import utils


log = logging.getLogger(__name__)


class Orchestrator(object):
    """Orchestrator class to run sharded tests as per the GPUs available"""

    def __init__(self, node):
        self.node = node
        self.gpus = node.getGpus()
        log.info(f"Total GPUs: {len(self.gpus)}")

    def runBinary(self, *args, retries=3, reqOut=False, **kwargs):
        for i in range(retries):
            i and log.info(f"[{self.node.host}][{i+1}]: Rerunning Failed Tests")
            failed = set()
            ret, out = self.node.runCmd(*args, **kwargs, reqOut=True)
            if ret == 0:
                break
            if re.search(r"Reached Timeout", out[-100:]):
                failed.add("Test Process Halt/Stuck")
            else:
                failed.add(f"Unknown tests failed with ret: {ret}")
        return (not failed and ret == 0), failed, out

    # ctests - https://cmake.org/cmake/help/latest/manual/ctest.1.html
    def runCtest(self, *args, retries=3, **kwargs):
        """Runs the CTest based tests in sharded parallel threads"""

        def _runCtest(gpu, tests, *args, **kwargs):
            """Runs an single CTest shard on an assigned GPU with auto retry of failed tests"""
            cmd = ("ctest",)
            for i in range(retries):
                ret, out, _ = gpu.runCmd(*cmd, *tests, *args, **kwargs)
                if ret == 0:
                    return ret, out
                tests = (*tests, "--rerun-failed")
                log.info(f"[{gpu.node.host}]: Rerunning Failed Tests")
            return ret, out

        def _runCtestShards(gpu, shards, iShard, *args, **kwargs):
            """Runs all the tests in default CTest sharding mode"""
            tests = ("--tests-information", f"{iShard+1},,{shards}")
            return _runCtest(gpu, tests, *args, **kwargs)

        # shards tests
        shards = len(self.gpus)
        rets = utils.runParallel(
            *[
                (_runCtestShards, (gpu, shards, iShard, *args), kwargs)
                for iShard, gpu in enumerate(self.gpus)
            ]
        )
        # reporting
        result = True
        for ret, out in rets:
            result &= bool(ret == 0)
        assert result
        return result

    # gtests - https://google.github.io/googletest/
    def runGtest(self, binary, *args, gfilter=None, retries=3, **kwargs):
        """Runs the GTest based tests in sharded parallel threads"""

        def _runGtest(gpu, binary, gfilter, *args, **kwargs):
            """Runs an single GTest shard on an assigned GPU with auto retry of failed tests"""
            for i in range(retries):
                i and log.info(f"[{gpu.node.host}][{i+1}]: Rerunning Failed Tests")
                ret, out = gpu.runCmd(
                    binary,
                    f"--gtest_filter={gfilter}" if gfilter else "",
                    *args,
                    **kwargs,
                    reqOut=True,
                )
                if ret == 0:
                    break
            return ret, out

        def _runGtestShards(gpu, binary, gfilter, shards, iShard, *args, **kwargs):
            """Runs all the tests in default GTest sharding mode"""
            env = kwargs.pop("env", {})
            env.update(
                {
                    "GTEST_TOTAL_SHARDS": shards,
                    "GTEST_SHARD_INDEX": iShard,
                }
            )
            ret = _runGtest(gpu, binary, gfilter, *args, env=env, **kwargs)
            return ret

        # shards tests
        shards = len(self.gpus)
        rets = utils.runParallel(
            *[
                (_runGtestShards, (gpu, binary, gfilter, shards, iShard, *args), kwargs)
                for iShard, gpu in enumerate(self.gpus)
            ]
        )
        # reporting
        result = True
        for ret, out in rets:
            result &= bool(ret == 0)
        assert result
        return result
