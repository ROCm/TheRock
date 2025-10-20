import logging
import pytest
import time

logging.getLogger("urllib3").setLevel(logging.WARNING)

from libs import nodes
from libs import orchestrator


def pytest_addoption(parser):
    """Initialization of cmdline args"""
    parser.addoption(
        "--therock-path", action="store", default="/therock", help="Path of TheRock"
    )


@pytest.fixture(scope="session")
def orch():
    """Fixture to access the Test Orchestrator Object"""
    return orchestrator.Orchestrator(node=nodes.Node())


@pytest.fixture(scope="session")
def therock_path(pytestconfig, orch):
    """Fixture to access the path to the TheRock passed by cmdline arg: --therock-path"""
    rockDir = pytestconfig.getoption("therock_path")
    return rockDir


@pytest.fixture(scope="session")
def report(request):
    """Fixture to access the Test Reporting Object"""
    from libs import report

    report = report.Report()
    yield report
    verdict = not (request.session.testsfailed)
    report.pprint()


@pytest.fixture(scope="class")
def table(report):
    """Fixture to access the Test Result table in Report"""
    table = report.addTable(title="Test Report:")
    table.addRow("Test", "Verdict", "ExecTime")
    return table


@pytest.fixture(scope="function")
def result(pytestconfig, request, report, table):
    """Fixture to access the Result Object"""
    report.testVerdict = False
    startTime = time.time()
    yield report
    testName = request.node.name
    verdictStr = ("FAIL", "PASS")[report.testVerdict]
    execTime = time.strftime("%H:%M:%S", time.gmtime(time.time() - startTime))
    table.addRow(testName, verdictStr, execTime)
