#!/usr/bin/python3

import sys
import logging
import tabulate
from pathlib import Path


log = logging.getLogger(__name__)

THEROCK_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(THEROCK_DIR / "build_tools" / "github_actions"))
from github_actions_utils import gha_append_step_summary


class Table:
    """A class to create table entries in the test report"""

    def __init__(self, title):
        self.title = title
        self.data = [[]]

    def addRow(self, *row):
        self.data.append(list(row))

    def pprint(self):
        if not self.data:
            return "No Data Found in Report"
        fmt = f"{self.title}\n"
        fmt += tabulate.tabulate(
            self.data[1:],
            headers=self.data[0],
            tablefmt="simple_outline",
            rowalign="center",
        )
        return fmt


class Report(object):
    """A class to create test reports"""

    def __init__(self, title=""):
        self.title = title
        self.tables = []

    def addTable(self, title):
        table = Table(title)
        self.tables.append(table)
        return table

    def pprint(self):
        log.info(f": {self.title} :".center(100, "-"))
        # tables
        for table in self.tables:
            log.info(table.pprint())
            gha_append_step_summary(table.pprint())
