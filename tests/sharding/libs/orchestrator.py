#!/usr/bin/python3
import os
import re
import sys
import time
import fcntl

from . import utils
from .utils import log


class Orchestrator(object):
	''' Orchestrator class to run sharded tests as per the GPUs available '''
	def __init__(self, node):
		self.node = node
		self.gpus = node.getGpus()
		log(f'Total GPUs: {len(self.gpus)}')

	def runBinary(self, *args, **kwargs):
		''' Runs the binary tests and collects the results with auto retries '''
		for i in range(3):
			i and log(f'[{i+1}]: Rerunning Failed Tests')
			if ret := (self.node.runCmd(*args, **kwargs) == 0):
				break
		return ret

	@utils._callOnce
	def _getCacheDir(self):
		cacheDir = os.path.join(os.environ['HOME'], 'testCaches')
		os.makedirs(cacheDir, exist_ok=True)
		return cacheDir

	def _getCacheFile(self, testDir):
		cacheFile = os.path.join(self._getCacheDir(),
			os.path.normpath(testDir).replace('/', '__')
		)
		return cacheFile

	def _getTestCache(self, cacheFile):
		''' Gets the test durations from the cache '''
		if not os.path.exists(cacheFile):
			return {}
		fd = open(cacheFile, 'r')
		fcntl.flock(fd.fileno(), fcntl.LOCK_EX)
		content = fd.read()
		fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
		fd.close()
		cache = {test:int(duration)
			for (test, duration) in re.findall(r'(.*?)\|(\d+)', content)
		}
		return cache

	def _updateTestCache(self, cacheFile, cache):
		''' Updates the test durations into the cache '''
		fd = open(cacheFile, 'w')
		fcntl.flock(fd.fileno(), fcntl.LOCK_EX)
		fd.write('\n'.join([f'{test}|{duration}' for (test, duration) in cache.items()]))
		fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
		fd.close()

	def _splitShards(self, testList, cache):
		''' Splits the tests into efficient shards for a given GPU count '''
		testList = sorted(testList, key=lambda test: cache.get(test, 500) or 1, reverse=True)
		# group all tests as per their execution timings in cache
		shards = [[0, []] for i in range(len(self.gpus))]
		for test in testList:
			leastShard = min(shards, key=lambda shard: shard[0])
			leastShard[0] += cache.get(test, 500) or 1
			leastShard[1].append(test)
		for i, (shardTime, testList) in enumerate(shards):
			log(f'Shard[{i}] => For {shardTime/1000} secs {len(testList)} tests selected')
			for test in testList[:10]:
				log(f'\t{test}: {cache.get(test)}')
			log('\t...')
		return shards

	def runCtest(self, *args, **kwargs):
		''' Runs the CTest based tests in sharded parallel threads '''
		def _runCtest(gpu, tests, *args, **kwargs):
			''' Runs an single CTest shard on an assigned GPU with auto retry of failed tests '''
			cmd = ('ctest', )
			for i in range(3):
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
		def _collectCtest(*args, **kwargs):
			''' Collects all the tests of CTest applying given test filters '''
			import json
			ret, out = self.node.runCmd('ctest', '--show-only=json-v1', *args,
				out=True, verbose=False, **kwargs,
			)
			testData = json.loads(out)
			return {t['name'] for t in testData['tests']}
		def _runCtestScheduler(gpu, testList, *args, **kwargs):
			''' Runs an all the tests in efficiently choosed shards as per their duration '''
			testListFile = f'/tmp/testList{gpu.index}'
			gpu.node.writeFile(testListFile, '\n'.join(testList))
			tests = ('--tests-from-file', testListFile)
			return _runCtest(gpu, tests, *args, **kwargs)
		cacheFile = self._getCacheFile(kwargs.get('cwd', 'ctestMisc'))
		cache = self._getTestCache(cacheFile)
		if cache: # schedule tests
			testList = _collectCtest(args, **kwargs)
			shards = self._splitShards(testList, cache)
			rets = utils.runParallel(*[(
				_runCtestScheduler, (gpu, shards[i][1], *args), kwargs)
				for i, gpu in enumerate(self.gpus)
			])
		else: # shards tests
			shards = len(self.gpus)
			rets = utils.runParallel(*[(
				_runCtestShards, (gpu, shards, iShard, *args), kwargs)
				for iShard, gpu in enumerate(self.gpus)
			])
		# updating the cache
		result = True
		expr = re.compile(r'Test\s+#\d+: (.+?) \..*?Passed\s+([\d\.]+) sec')
		for ret, out in rets:
			result &= bool(ret == 0)
			cache.update({test:int(float(duration)*1000) for (test, duration) in expr.findall(out)})
		self._updateTestCache(cacheFile, cache)
		assert result
		return True

	def runGtest(self, binary, *args, srcFile=None, gfilter=None, **kwargs):
		''' Runs the GTest based tests in sharded parallel threads '''
		def _runGtest(gpu, binary, gfilter, *args, **kwargs):
			''' Runs an single GTest shard on an assigned GPU with auto retry of failed tests '''
			for i in range(3):
				ret, out = gpu.runCmd(binary,
					f'--gtest_filter={gfilter}' if gfilter else '',
					*args, out=True, **kwargs,
				)
				if ret == 0:
					return ret, out
				if failed := re.findall(r'FAILED\s+\] (.+?) \(\d+ ms', out):
					log(f'[{gpu.node.host}]: Rerunning Failed Tests')
					gFilter = ':'.join(failed)
			return ret, out
		def _runGtestShards(gpu, binary, gfilter, shards, iShard, *args, **kwargs):
			''' Runs all the tests in default GTest sharding mode '''
			env = kwargs.pop('env', {})
			env.update({
				'GTEST_TOTAL_SHARDS': shards,
				'GTEST_SHARD_INDEX': iShard,
			})
			return _runGtest(gpu, binary, gfilter, *args, env=env, **kwargs)
		def _collectGtests(binary, srcFile, gfilter, *args, **kwargs):
			''' Collects all the tests of GTest applying given gfilters '''
			import json
			srcCmd = ('source', f'{srcFile};') if srcFile else ()
			jsonFile = f'/tmp/{binary.split()[-1]}.json'
			cmd = (*srcCmd, binary,
				'--gtest_list_tests', '--gtest_color=no',
				f'--gtest_filter={gfilter}' if gfilter else '',
				f'--gtest_output=json:{jsonFile}',
			)
			ret, out = self.node.runCmd('bash', '-c', ' '.join(cmd), *args,
				verbose=False, out=True, **kwargs,
			)
			if os.path.exists(jsonFile):
				fd = open(jsonFile)
				testData = json.load(fd)
				fd.close()
				return {ts['name']:[t['name'] for t in ts['testsuite']] for ts in testData['testsuites']}
			# if test gtest version did not support json output
			testDict = {}
			suite = None
			sExpr = re.compile(r'^(\w+\.)$')
			tExpr = re.compile(r'^\s+(\w+)$')
			for line in out.splitlines():
				if mtch := sExpr.search(line):
					suite = mtch.group(1)
					testDict[suite] = []
				if suite and (mtch := tExpr.search(line)):
					testDict[suite].append(mtch.group(1))
			return testDict
		def _runGtestScheduler(gpu, binary, testList, *args, **kwargs):
			''' Runs an all the tests in efficiently choosed shards as per their duration '''
			maxArgLen = 127*1024  # split the filter so that it should not exceed max bash arg size
			ret, out = 0, ''
			for gFilter in re.findall(rf':?(.{{1,{maxArgLen}}}[^:$]+)', ':'.join(testList)):
				ret, _out = _runGtest(gpu, binary, gFilter, *args, **kwargs)
				out += _out
				if ret != 0:
					break
			return ret, out
		cacheFile = self._getCacheFile(os.path.join(kwargs.get('cwd', ''), binary))
		cache = self._getTestCache(cacheFile)
		if cache: # schedule tests
			testDict = _collectGtests(binary, srcFile, gfilter, *args, **kwargs)
			testList = [f'{ts}.{t}' for ts in testDict for t in testDict[ts]]
			shards = self._splitShards(testList, cache)
			rets = utils.runParallel(*[(
				_runGtestScheduler, (gpu, binary, shards[i][1], *args), kwargs)
				for i, gpu in enumerate(self.gpus)
			])
		else: # shards tests
			shards = len(self.gpus)
			rets = utils.runParallel(*[(
				_runGtestShards, (gpu, binary, gfilter, shards, iShard, *args), kwargs)
				for iShard, gpu in enumerate(self.gpus)
			])
		# updating the cache
		result = True
		expr = re.compile(r'OK \] (.+?) \((\d+) ms')
		for ret, out in rets:
			result &= bool(ret == 0)
			cache.update({test:int(duration) for (test, duration) in expr.findall(out)})
		self._updateTestCache(cacheFile, cache)
		assert result
		return True
