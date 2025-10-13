#!/usr/bin/python3
import os
import re
import sys
import time
import glob
import json
import dill
import lzma
import base64
import select
import socket
import pickle
import textwrap
import subprocess
import kubernetes

from . import utils
from libs.utils import log

TIMEOUT = 1200


class Gpu(object):
	''' A class to run test cmds using GPU pinning '''
	def __init__(self, node, index=0, env={}):
		self.node = node
		self.index = index
		self.env = env

	def runCmd(self, *cmd, env={}, **kwargs):
		''' Runs cmd on assigned GPU only '''
		env.update(self.env)
		return self.node.runCmd(*cmd, env=env, **kwargs)


class Node(object):
	''' A class to handle all communications with current Node/OS '''
	def __init__(self, env={}):
		self.env = env
		self.host = socket.gethostname()

	def runCmd(self, *cmd, cwd=None, env=None, stdin=None, timeout=TIMEOUT, verbose=True,
		out=False, err=False, **kwargs,
	):
		''' Executes Cmd on the current node:
			*cmd[str-varargs]: of cmd and its arguments
			cwd[str]: current working dirpath from where cmd should run
			env[dict]: extra environment variable to be passed to the cmd
			stdin[str]: input to the cmd via its stdin
			timeout[int]: min time to wait before killing the process when no activity observed
			verbose[bool]: verbose level, True=FullLog, False=OnlyInfo-NoLog, None=NoInfo-NoLog
		'''
		if verbose != None:
			cwdStr = f'cd {cwd}; ' if cwd else ''
			envStr = ''
			if env:
				for key, value in env.items():
					envStr += f"{key}='{value}' "
			log(f'RunCmd: {cwdStr}{envStr}{" ".join(cmd)}')
		if env:
			env = {k:str(v) for k,v in env.items()}
			env.update(os.environ)
		# launch process
		process = subprocess.Popen(cmd,
			stdin=subprocess.PIPE if stdin else None,
			stdout=subprocess.PIPE, stderr=subprocess.PIPE,
			cwd=cwd, env=env, close_fds=True, **kwargs
		)
		# handling stdin
		if stdin:
			process.stdin.write(stdin if isinstance(stdin, bytes) else stdin.encode())
			process.stdin.close()
		# following process stdout / stderr
		os.set_blocking(process.stdout.fileno(), False)
		os.set_blocking(process.stderr.fileno(), False)
		verbose and log('out:')
		rets, stdout, stderr = None, b'', b''
		def _readStream(fd):
			chunk = fd.read(8196)
			verbose and log(chunk, newline=False)
			return chunk
		chunk = None
		while chunk != b'':
			rfds = select.select([process.stdout, process.stderr], [], [], timeout)[0]
			if not rfds:
				msg = f'Reached Timeout of {timeout} sec, Exiting...'
				log(msg)
				stdout += msg.encode()
				process.kill()
				break
			if process.stdout in rfds:
				stdout += (chunk := _readStream(process.stdout))
			if process.stderr in rfds:
				stderr += (chunk := _readStream(process.stderr))
		# handling return value
		ret = process.wait()
		if ret != 0 and verbose != None:
			log(f'cmd failed: {" ".join(cmd)}')
		verbose and log(f'ret: {ret}')
		# returns
		if not out:
			return ret
		if not err:
			return ret, (stdout+stderr).decode()
		return ret, stdout.decode(), stderr.decode()

	@utils._callOnce
	def getGpuCount(self):
		''' Gets the GPU count of the node '''
		return len(glob.glob('/dev/dri/render*'))

	@utils._callOnce
	def getGpus(self):
		''' Gets the GPU Objects of the node with their GPU pinning envs '''
		if (ngpus := self.getGpuCount()) > 1:
			return [Gpu(self, i, env={
				#'ROCR_VISIBLE_DEVICES': i,
				'HIP_VISIBLE_DEVICES': i,
				'HSA_VISIBLE_DEVICES': i,
			}) for i in range(ngpus)]
		return [Gpu(self)]
