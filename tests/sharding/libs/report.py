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

	def addRow(self, *row):
		self.data.append(list(row))

	def addCol(self, *col):
		for index, element in enumerate(col):
			if index >= len(self.data):
				self.data.append([])
			self.data[index].append(element)

	def addHeader(self, *headers):
		self.data = [[]]
		self.keyIndex = None
		for index, header in enumerate(headers):
			self.data[0].append(header)
			if self.keyIndex == None and not isinstance(header, list):
				self.keyIndex = index

	def addResult(self, *fields):
		# check entry exists in results
		rowIndex = None
		key = fields[self.keyIndex]
		for index, row in enumerate(self.data[1:]):
			if row[self.keyIndex] == key:
				rowIndex = index + 1
		# for new entry
		if not rowIndex:
			self.data.append(list(fields))
			return
		# if result already exists
		for index, field in enumerate(fields):
			if not isinstance(field, list):
				self.data[rowIndex][index] = field # overwrite the result
				continue
			self.data[rowIndex][index].append(field[0])
			# adjust header as per the length of the field
			if len(self.data[0][index]) < len(self.data[rowIndex][index]):
				self.data[0][index].append(self.data[0][index][0])

	def formTable(self):
		table = [[]]
		fieldIndexList = []
		# headers
		for header in self.data[0]:
			fieldIndexList.append(len(table[0]))
			if isinstance(header, list):
				if len(header) > 1:
					for shIndex, subHeader in enumerate(header):
						table[0].append(f'{subHeader} - [{shIndex+1}]')
				else:
					table[0].append(header[0])
				continue
			table[0].append(header)
		# data
		for rowIndex, row in enumerate(self.data[1:]):
			table.append([])
			for fIndex, field in enumerate(row):
				if isinstance(field, list):
					headerLen = len(self.data[0][fIndex])
					for i in range(headerLen - len(field)):
						field.append('')
					table[rowIndex+1] += field
				else:
					table[rowIndex+1].append(field)
		return table

	def pprint(self):
		if not self.data:
			return 'No Data Found in Report'
		table = self.formTable()
		fmt = f'{self.title}\n'
		fmt += tabulate.tabulate(table[1:], headers=table[0],
			tablefmt='simple_outline', rowalign='center',
		)
		return fmt


class Report(object):
	''' A class to create test reports '''
	def __init__(self, title=''):
		self.title = title
		self.text = ' '
		self.facts = {}
		self.tables = []
		self.errors = {}
		self.errTitle = ''

	def setTitle(self, title, append=True):
		self.title = self.title+title if append else title

	def setText(self, text, append=True):
		self.text = self.text+text if append else text

	def addFacts(self, **kwargs):
		self.facts.update(kwargs)

	def addTable(self, title):
		table = Table(title)
		self.tables.append(table)
		return table

	def addErrors(self, errors, title=''):
		self.errors[title] = self.errors.get(title, [])
		self.errors[title].extend(errors)

	def toHtml(self, title=True, facts=True, tables=True, errors=True):
		htmlVars = {
			'title': '',
			'facts': '',
			'tables': '',
			'errors': '',
		}
		# title
		if title:
			htmlVars['title'] = self.title
		# facts
		if facts and self.facts:
			factList = ''
			for fact, value in self.facts.items():
				factList += FACT_HTML.format(style=FACT_STYLE, fact=fact, value=value)
			htmlVars['facts'] += FACTS_HTML.format(style=FACTS_STYLE, factList=factList)
		# tables
		if tables:
			for table in self.tables:
				data = table.formTable()
				ths = '\n'.join([TABLE_TH_HTML.format(style=TABLE_TH_STYLE, th=th) for th in data[0]])
				trs = '\n'.join([
					TABLE_TR_HTML.format(style=TABLE_TR_STYLE, tds='\n'.join([
						TABLE_TD_HTML.format(style=TABLE_TD_STYLE, td=td) for td in tr
					])) for tr in data[1:]
				])
				htmlVars['tables'] += TABLE_HTML.format(
					tableStyle=TABLE_STYLE, captionStyle=CAPTION_STYLE,
					title=table.title, ths=ths, trs=trs
				)
		# errors
		if errors and self.errors:
			errorList = ''
			for cmd, ret, out in self.errors:
				errorList += ERROR_HTML.format(errorStyle=ERROR_STYLE,
					errorHeadStyle=ERRORHEAD_STYLE, errorOutStyle=ERROROUT_STYLE,
					errorHead=cmd, errorOut=out.strip()
				)
			htmlVars['errors'] += ERRORS_HTML.format(title=f'{self.errTitle} Errors', errorList=errorList)
		html = HTML.format(**htmlVars)
		return html

	def pprint(self):
		log('\n')
		log(f': {self.title} :'.center(100, '-'))
		# facts
		for fact, value in self.facts.items():
			log(f'{fact:>24} : {value}')
		log(''.center(60, '-'))
		# tables
		for table in self.tables:
			log(table.pprint())
		# errors
		for title, errors in self.errors.items():
			for i, error in enumerate(errors):
				log(f': {title} - Errors[{i+1}/{len(errors)}] :'.center(60, '-'))
				cmd, ret, out = error
				log(f'{cmd}\n{out}')
		log(''.center(100, '-'))


HTML = '''\
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
</head>
<body style="font-family: sans-serif; margin: 20px;">
    <h1>{title}</h1>
	{facts}
	{tables}
	{errors}
</body>
</html>
'''
FACTS_STYLE = '''
display: grid;
grid-template-columns: 1fr 1fr;
gap: 10px;
border: 1px solid #ccc;
padding: 10px;
border-radius: 5px;
margin-bottom: 10px;
'''
FACTS_HTML = '''
	<div style="{style}">
		{factList}
	</div>
'''
FACT_STYLE = 'font-weight: bold;'
FACT_HTML = '<div style="{style}">{fact}</div><div>: {value}</div>\n'

TABLE_STYLE = '''
width: 100%;
border-collapse: collapse;
margin-bottom: 20px;
'''
CAPTION_STYLE = '''
font-weight: bold;
font-size: 1.2em;
padding: 5px;
text-align: left;
margin-right: 10px;
white-space: nowrap;
'''
TABLE_HTML = '''
<table style="{tableStyle}">
	<caption style="{captionStyle}">{title}</caption>
	<thead>
		<tr>
			{ths}
		</tr>
	</thead>
	<tbody>
		{trs}
	</tbody>
</table>
'''
TABLE_TH_STYLE = '''
border: 1px solid #ccc;
background-color: #f2f2f2;
font-weight: bold;
'''
TABLE_TH_HTML = '<th style="{style}">{th}</th>\n'
TABLE_TR_STYLE = '''
border: 1px solid #ccc;
padding: 8px;
text-align: left;
'''
TABLE_TR_HTML = '<tr style="{style}">{tds}</tr>\n'
TABLE_TD_STYLE = TABLE_TR_STYLE
TABLE_TD_HTML = '<td style="{style}">{td}</td>\n'

ERRORS_HTML = '''\
	<h3>{title}: </h3>
	{errorList}
'''
ERROR_STYLE = '''
border: 1px solid #ccc;
padding: 10px;
margin-bottom: 20px;
border-radius: 5px;
'''
ERRORHEAD_STYLE = '''
background-color: #f0f0f0;
padding: 10px;
margin-bottom: 5px;
border-radius: 5px;
color: red;
'''
ERROROUT_STYLE = '''
background-color: #e0e0e0;
padding: 10px;
border-radius: 5px;
white-space: pre-wrap;
'''
ERROR_HTML = '''\
	<div style="{errorStyle}">
		<div style="{errorHeadStyle}">{errorHead}</div>
		<div style="{errorOutStyle}">{errorOut}</div>
	</div>
'''
