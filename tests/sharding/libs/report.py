#!/usr/bin/python3
import os
import re
import sys
import time
import json
import smtplib
import tabulate
import traceback
import pymsteams
import email.mime.text
import email.mime.multipart

from . import utils
from libs.utils import log


class Table():
	''' A class to create table entries in the test report '''
	def __init__(self, title):
		self.title = title
		self.data = [[]]

	def addRow(self, *row):
		self.data.append(list(row))

	def pprint(self):
		if not self.data:
			return 'No Data Found in Report'
		fmt = f'{self.title}\n'
		fmt += tabulate.tabulate(self.data[1:], headers=self.data[0],
			tablefmt='simple_outline', rowalign='center',
		)
		return fmt


class Report(object):
	''' A class to create test reports '''
	def __init__(self, title=''):
		self.title = title
		self.tables = []

	def addTable(self, title):
		table = Table(title)
		self.tables.append(table)
		return table

	def pprint(self):
		log('\n')
		log(f': {self.title} :'.center(100, '-'))
		# tables
		for table in self.tables:
			log(table.pprint())
