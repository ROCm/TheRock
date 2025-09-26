#!/usr/bin/python3
import os
import re
import sys
import json
import time
import base64
import logging
import traceback


class Singleton(type):
	_instances = {}
	def __call__(cls, *args, **kwargs):
		if cls not in cls._instances:
			cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
		return cls._instances[cls]


def _callOnce(funcPointer):
	def funcWrapper(*args, **kwargs):
		if 'ret' not in funcPointer.__dict__:
			funcPointer.ret = funcPointer(*args, **kwargs)
		return funcPointer.ret
	return funcWrapper


@_callOnce
def getLogger(level=logging.INFO, logFile=None):
	logger = logging.getLogger()
	logger.setLevel(level)
	formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
	cHandler = logging.StreamHandler(sys.stdout)
	cHandler.setLevel(logging.DEBUG)
	cHandler.setFormatter(formatter)
	logger.addHandler(cHandler)
	if logFile:
		fHandler = logging.FileHandler(logFile)
		fHandler.setLevel(level)
		fHandler.setFormatter(formatter)
		logger.addHandler(fHandler)
	return logger


def log(msg, newline=True):
	if isinstance(msg, bytes):
		msg = msg.decode('utf-8', errors='ignore')
	msg = msg + ('', '\n')[newline]
	sys.stdout.write(msg) and sys.stdout.flush()


def logExp(e):
	(tbHeader, *tbLines, error) = traceback.format_exception(type(e), e, e.__traceback__)
	log(f'{error}{tbHeader}{"".join(tbLines)}')


def runCmd(*cmd, cwd=None, env={}, stdin=None, nowait=False, verbose=True, out=False, **kwargs):
	import subprocess
	if verbose != None:
		cwdStr = f'cd {cwd}; ' if cwd else ''
		envStr = ''
		for key, value in env.items():
			envStr += f"{key}='{value}' "
		log(f'RunCmd: {cwdStr}{envStr}{" ".join(cmd)}')
	# launch process
	cmdEnv = os.environ.copy()
	env and cmdEnv.update({k:str(v) for k,v in env.items()})
	process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
		stderr=subprocess.STDOUT, cwd=cwd, env=cmdEnv, **kwargs
	)
	# handling stdin
	if isinstance(stdin, str):
		process.stdin.write(stdin if isinstance(stdin, bytes) else stdin.encode())
	elif isinstance(stdin, bytes):
		process.stdin.write(stdin if isinstance(stdin, bytes) else stdin.encode())
	elif isinstance(stdin, type(process.stdout)):
		chunk = None
		while chunk != b'':
			chunk = stdin.read(1024)
			process.stdin.write(chunk)
	process.stdin.close()
	if nowait:  # background process
		return process
	# following process stdout / stderr
	verbose and log('out:')
	out = ''
	chunk = None
	while chunk != b'':
		chunk = process.stdout.read(1)
		verbose and sys.stdout.buffer.write(chunk) and sys.stdout.flush()
		out += chunk.decode(errors='replace')
	# handling return value
	ret = process.wait()
	if ret != 0 and verbose != None:
		log(f'cmd failed: {" ".join(cmd)}')
		verbose or log(f'err: {out}')
	verbose and log(f'ret: {ret}')
	return (ret, out) if out else ret 


def request(method, url, allowCodes=(), ignoreExp=False, verbose=False, **kwargs):
	# /usr/lib/python3/dist-packages/requests/api.py
	# method: GET, POST, DELETE, OPTIONS
	# kwargs: headers, json, data, params, cookies
	import urllib3
	urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
	import requests
	from requests.exceptions import ConnectionError, Timeout, RequestException
	verbose and log(f'{method}: {url}')
	kwargs and verbose and log(f'\t{kwargs}')
	allowCodes = (200, *allowCodes)
	exp = resp = None
	for i in range(3): # retry for 3 times
		try:
			resp = requests.request(method, url, **kwargs)
		except (ConnectionError, Timeout, RequestException) as e:
			(verbose != None) and logExp(e)
			exp = e
		if resp and resp.status_code in allowCodes:
			break
		resp and (verbose != None) and log(f'{method}: {url} => [{resp.status_code}]{resp.reason}')
		(verbose != None) and log(f'Re-Trying...')
		time.sleep(2**(i+1))
	if exp and not ignoreExp:
		raise exp
	if resp:
		if verbose or (verbose == False and resp.status_code not in allowCodes):
			log(f'{method}: {url} => [{resp.status_code}]{resp.reason}')
		if re.search(r'^[\[\{].*[\]\}]$', resp.text, flags=re.DOTALL):
			resp.jsonData = resp.json()
			verbose and log(f'Json: {json.dumps(resp.jsonData, sort_keys=True, indent=2)}')
		else:
			resp.jsonData = None
			verbose and log(f'Text: {resp.text}')
	return resp


def runParallel(*funcs):
	import threading
	rets = [None] * len(funcs)
	def proxy(i, funcPtr, *args, **kwargs):
		rets[i] = funcPtr(*args, **kwargs)
	# launching parallel threads
	threads = []
	for (i, (funcPtr, args, kwargs)) in enumerate(funcs):
		thread = threading.Thread(target=proxy, args=(i, funcPtr, *args), kwargs=kwargs)
		threads.append(thread)
		thread.start()
	# wait for threads join
	while threads:
		for thread in threads:
			thread.join()
			if thread.is_alive():
				continue
			threads.remove(thread)
	return rets
