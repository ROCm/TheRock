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
		fmt = tabulate.tabulate(table[1:], headers=table[0],
			tablefmt='simple_outline', rowalign='center',
		)
		return fmt


class Report(object):
	def __init__(self, title=''):
		self.title = title
		self.text = ' '
		self.facts = {}
		self.buttons = {}
		self.tables = []
		self.errors = []
		self.errTitle = ''
		self.exceptions = []
		self.expTitle = ''

	def setTitle(self, title, append=True):
		self.title = self.title+title if append else title

	def setText(self, text, append=True):
		self.text = self.text+text if append else text

	def addFacts(self, **kwargs):
		self.facts.update(kwargs)

	def addButtons(self, **kwargs):
		self.buttons.update(kwargs)

	def addTable(self, title):
		table = Table(title)
		self.tables.append(table)
		return table

	def addErrors(self, *error, title=''):
		self.errors.extend(error)
		self.errTitle = title

	def addException(self, exception):
		self.exceptions.append(exception)

	def sendMsg(self, verdict):
		url = 'https://amdcloud.webhook.office.com/webhookb2'
		channelId = '89053b43-3579-452c-b12c-126f4a429f7f'	# AGS DevOps Team
		grpId = '3dd8961f-e488-4e60-8e11-a82d994e183d'
		compId = '323214c2-5f5e-4599-9bea-c32b2102d047'
		#webhookId, keyId = '8c00abb05ec2481783f43e6637ae0c92', 'V2vGIgfc3jI03EFaYcHNF7PWIaTP8MmmPpYgvSVH1Wojw1' # MyNotification
		webhookId, keyId = '50c40b95934a48a081db423032bd9384', 'V2muGEtUmxdPPmvboFHxr1Izu2fbT-TLL13zTpWN7wZ8c1' # Rocm-Tests
		webhookUrl = f'{url}/{channelId}@{grpId}/IncomingWebhook/{webhookId}/{compId}/{keyId}'
		msg = pymsteams.connectorcard(webhookUrl)
		msg.title(f'{self.title} - {("FAIL", "PASS")[verdict]}')
		msg.color(('#FF0000', '#00FF00')[verdict])
		msg.title(self.title)
		msg.text(self.text)
		mainSection = pymsteams.cardsection()
		msg.addSection(mainSection)
		# facts
		for fact, value in self.facts.items():
			mainSection.addFact(fact, value)
		# buttons
		for button, url in self.buttons.items():
			msg.addLinkButton(buttontext=button, buttonurl=url)
		# tables
		for table in self.tables:
			tableSection = pymsteams.cardsection()
			tableSection.title(table.title)
			data = table.formTable()
			# table title and header
			style = ' style="border:2px solid grey; padding: 3px;"'
			html = f'<table{style}>'
			html += f'<tr{style}>'
			for element in data[0]:
				html += f'<th{style}>{element}</th>'
			html += '</tr>'
			# table data
			for tr in data[1:]:
				html += f'<tr{style}>'
				for td in tr:
					html += f'<td{style}>{td}</td>'
				html += '</tr>'
			html += '</table><br>'
			tableSection.text(html)
			msg.addSection(tableSection)
		if len(json.dumps(msg.payload)) > 20000: # squeeze pkg size
			for i,section in enumerate(msg.payload['sections']):
				if 'text' in section:
					msg.payload['sections'][i]['text'] = re.sub(' style=".*?"', '', section['text'])
		# errors
		if self.errors:
			errSection = pymsteams.cardsection()
			errSection.addFact(f'{self.errTitle} Errors:', '')
			for error in self.errors:
				cmd, ret, out = error
				out.strip() and errSection.addFact('', f'<pre><strong>$ {cmd}</strong>\n{out}</pre>')
			msg.addSection(errSection)
		# exceptions
		if self.exceptions:
			msg.color('#FF0000')
			msg.title(f'{self.title} - EXCEPTION')
			expSection = pymsteams.cardsection()
			expSection.addFact(f'{self.expTitle} Exceptions:', '')
			for exception in self.exceptions:
				header, *expLines, error = traceback.format_exception(
					type(exception), exception, exception.__traceback__
				)
				expLines = '\n'.join(expLines)
				expSection.addFact('', f'<pre><strong>{error}</strong>{header}{expLines}</pre>')
			msg.addSection(expSection)
		msg.send()
		log('Notification Sent')

	def toHtml(self, title=True, facts=True, buttons=True, tables=True, errors=True, exceptions=True):
		htmlVars = {
			'title': '',
			'facts': '',
			'buttons': '',
			'tables': '',
			'errors': '',
			'exceptions': '',
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
		# buttons
		if buttons and self.buttons:
			buttonList = ''
			for button, url in self.buttons.items():
				buttonList += BUTTON_HTML.format(style=BUTTON_STYLE, button=button, url=url)
			htmlVars['buttons'] += BUTTONS_HTML.format(style=BUTTONS_STYLE, buttonList=buttonList)
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
		# exceptions
		if exceptions and self.exceptions:
			expList = ''
			for exception in self.exceptions:
				header, *expLines, error = traceback.format_exception(
					type(exception), exception, exception.__traceback__
				)
				expLines = '\n'.join(expLines)
				expList += ERROR_HTML.format(errorStyle=ERROR_STYLE,
					errorHeadStyle=ERRORHEAD_STYLE, errorOutStyle=ERROROUT_STYLE,
					errorHead=error.strip(), errorOut=f'{header}{expLines}'
				)
			htmlVars['exceptions'] += ERRORS_HTML.format(title=f'{self.expTitle} Exceptions', errorList=expList.strip())
		html = HTML.format(**htmlVars)
		return html

	def sendEmail(self, verdict, recipients=('kasula.madhusudhan@amd.com',)):
		emailMsg = email.mime.multipart.MIMEMultipart('alternative')
		emailMsg['Subject'] = f'{self.title} - {("FAIL", "PASS")[verdict]}'
		emailMsg['From'] = 'jenkins-compute@amd.com'
		emailMsg['To'] = ', '.join(recipients)
		emailMsg.attach(email.mime.text.MIMEText(self.toHtml(buttons=False), 'html'))
		session = smtplib.SMTP('torsmtp10.amd.com')
		session.send_message(emailMsg)
		session.quit()
		log('Email Sent')

	def setPipelineDesc(self, pObj, **kwargs):
		pObj.setDesc(self.toHtml(title=False, facts=False, buttons=False), **kwargs)


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
	{buttons}
	{tables}
	{errors}
	{exceptions}
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

BUTTONS_STYLE = '''
display: flex;
gap: 10px;
border: 1px solid #ccc;
padding: 10px;
border-radius: 5px;
margin-bottom: 10px;
'''
BUTTONS_HTML = '''
	<div style="{style}">
		{buttonList}
	</div>
'''
BUTTON_STYLE = '''
padding: 10px 20px;
font-size: 16px;
text-align: center;
text-decoration: none;
display: inline-block;
border: none;
border-radius: 5px;
cursor: pointer;
background-color: #cccccc;
color: #333;
'''
BUTTON_HTML = '<a style="{style}" href="{url}" target="_blank">{button}</a>\n'

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
