#!/usr/bin/python3
import json
import queue
import threading

import logging
from libs import utils


log = logging.getLogger(__name__)


class Orchestrator(object):
    """Orchestrator class to run sharded tests as per the GPUs available"""

    def __init__(self, node):
        self.node = node
        self.gpus = node.getGpus()
        self.testQueue = queue.Queue()
        log.info(f"Total GPUs: {len(self.gpus)}")

    def runCtestShards(self, *args, retries=3, **kwargs):
        """Runs the CTest based tests in sharded parallel threads"""

        def _runCtest(gpu, tests, *args, **kwargs):
            """Runs an single CTest shard on an assigned GPU with auto retry of failed tests"""
            cmd = ("ctest", "--output-junit", f"report_GPU{gpu.index}.xml")
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


    def runCtestScheduler(self, *args, retries=3, **kwargs):
        """Runs the CTest based tests in scheduled sharded parallel threads"""

        def _collectCtest(*args, **kwargs):
            ret, out, _ = self.node.runCmd('pudo', 'ctest', '--show-only=json-v1', *args, **kwargs)
            testData = json.loads(out)
            return {t['name'] for t in testData['tests']}

        def _runCtest(gpu, test, *args, **kwargs):
            """Runs an single CTest shard on an assigned GPU with auto retry of failed tests"""
            cmd = ("ctest", )
            rerun = ()
            for i in range(retries):
                ret, out, _ = gpu.runCmd(*cmd, '--tests-regex', f'{test}$', *rerun, *args, **kwargs)
                if ret == 0:
                    return ret, out
                rerun = ("--rerun-failed", )
                log.info(f"[{gpu.node.host}]: Rerunning Failed Tests")
            return ret, out

        def _runner(gpu):
            rets = []
            # iterate over queue items
            while not self.testQueue.empty():
                testName = self.testQueue.get()
                log.info(f"{testName = }")
                # run test
                ret = _runCtest(gpu, testName, *args, **kwargs)
                rets.append(ret)
            return rets

        # collect tests
        for testName in _collectCtest(*args, **kwargs):
            self.testQueue.put(testName)

        # schedule tests
        rets = utils.runParallel(
            *[(_runner, (gpu, ), {}) for gpu in self.gpus]
        )
        # reporting
        result = True
        for ret, out in sum(rets, []):
            result &= bool(ret == 0)
        assert result
        return result
