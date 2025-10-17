#!/usr/bin/python3
from . import utils
from .utils import log


class Orchestrator(object):
    ''' Orchestrator class to run sharded tests as per the GPUs available '''
    def __init__(self, node):
        self.node = node
        self.gpus = node.getGpus()
        log(f'Total GPUs: {len(self.gpus)}')

    def runCtest(self, *args, retries=3, **kwargs):
        ''' Runs the CTest based tests in sharded parallel threads '''
        def _runCtest(gpu, tests, *args, **kwargs):
            ''' Runs an single CTest shard on an assigned GPU with auto retry of failed tests '''
            cmd = ('ctest', )
            for i in range(retries):
                ret, out = gpu.runCmd(*cmd, *tests, *args, out=True, **kwargs)
                if ret == 0:
                    return ret, out
                tests = (*tests, '--rerun-failed')
                log(f'[{gpu.node.host}]: Rerunning Failed Tests')
            return ret, out
        def _runCtestShards(gpu, shards, iShard, *args, **kwargs):
            ''' Runs all the tests in default CTest sharding mode '''
            tests = ('--tests-information', f'{iShard+1},,{shards}')
            return _runCtest(gpu, tests, *args, **kwargs)
        # shards tests
        shards = len(self.gpus)
        rets = utils.runParallel(*[(
            _runCtestShards, (gpu, shards, iShard, *args), kwargs)
            for iShard, gpu in enumerate(self.gpus)
        ])
        # reporting
        result = True
        for ret, out in rets:
            result &= bool(ret == 0)
        assert result
        return True
