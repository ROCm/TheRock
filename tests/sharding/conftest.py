import os
import re
import sys
import time
import pytest

import logging
logging.getLogger('urllib3').setLevel(logging.WARNING)

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from libs import utils
from libs.utils import log
from libs import clusters
from libs import orchestrator


def pytest_addoption(parser):
	parser.addoption('--namespace', action='store', default='rocmci', help='kube cluster namespace')
	parser.addoption('--replicas', action='store', default=4, type=int, help='no.of nodes to be used')
	parser.addoption('--rock', action='store', default=None, help='install rock from runid/release')
	parser.addoption('--env', nargs='+', action='store', default=[], help='provide extra env')


@pytest.fixture(scope="session")
def extraEnv(pytestconfig):
	return dict(e.split('=') for e in pytestconfig.getoption('env'))


@pytest.fixture(scope='session')
def cluster(pytestconfig):
	namespace = pytestconfig.getoption('namespace')
	replicas = pytestconfig.getoption('replicas')
	version = pytestconfig.getoption('rock')
	appName = f'rock-tests-{version.replace(".", "-")}'
	cluster = clusters.KubeCluster(namespace, appName, replicas)
	assert cluster.deploy(), 'Failed to Deploy Cluster'
	#assert cluster.installPreReq(), 'Failed to install Pre-Requisites'
	yield cluster
	#cluster and cluster.delete(), 'Failed to Delete Cluster'


@pytest.fixture(scope='session')
def orch(cluster):
	return orchestrator.Orchestrator(cluster)


@pytest.fixture(scope='session')
def rock(pytestconfig, cluster):
	version = pytestconfig.getoption('rock')
	rets = cluster.installRock(version)
	assert all(rets), 'Failed to install Rock'
	return rets[0]


@pytest.fixture(scope='session')
def report(request):
	from libs import report
	report = report.Report()
	yield report
	verdict = not(request.session.testsfailed)


@pytest.fixture(scope='class')
def count(pytestconfig, report):
	count = None
	if pytestconfig.getoption('count') > 1:
		count = report.addTable(title='Iteration Report:')
		count.fCount = count.pCount = count.total = 0
		count.addHeader('Test', 'Fail', 'Pass', 'Total')
	yield count
	if pytestconfig.getoption('count') > 1:
		log('\n' + count.pprint())


@pytest.fixture(scope='class')
def table(report):
	table = report.addTable(title='Test Report:')
	table.addHeader('Test', 'Verdict', 'ExecTime')
	yield table
	log('\n' + table.pprint())


@pytest.fixture(scope='function')
def result(pytestconfig, request, report, count, table):
	report.testVerdict = False
	startTime = time.time()
	yield report
	testName = request.node.name
	# verdict
	verdictStr = ('FAIL', 'PASS')[report.testVerdict]
	for mark in request.node.own_markers:
		if mark.name == 'xfail':
			reason = mark.kwargs.get('reason', 'UnknownReason')
			verdictStr = (f'XFAIL [{reason}]', f'XPASS [{reason}]')[report.testVerdict]
			break
	# execution time
	execTime = time.strftime('%H:%M:%S', time.gmtime(time.time()-startTime))
	table.addResult(testName, verdictStr, execTime)
	# iteration report
	if count:
		count.total += 1
		if report.testVerdict:
			count.pCount += 1
		else:
			count.fCount += 1
		count.result(request.node.originalname, count.fCount, count.pCount, count.total)
		count.total == 1 and report.setTitle(f' - {request.node.originalname}')


@pytest.fixture(scope='session')
def ompEnv():
	return {'OMP_NUM_THREADS': '{int(self.getCpuCount()/2) or 1}'}
