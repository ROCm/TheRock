#!/usr/bin/python3
import os
import re
import sys
import json
import time
import base64
import logging
import traceback


def _callOnce(funcPointer):
	''' Decorator function enables calling function to get called only once per execution '''
	def funcWrapper(*args, **kwargs):
		if 'ret' not in funcPointer.__dict__:
			funcPointer.ret = funcPointer(*args, **kwargs)
		return funcPointer.ret
	return funcWrapper


def log(msg, newline=True):
	''' Common logger '''
	if isinstance(msg, bytes):
		msg = msg.decode('utf-8', errors='ignore')
	msg = msg + ('', '\n')[newline]
	sys.stdout.write(msg) and sys.stdout.flush()


def runParallel(*funcs):
	''' Runs the given list of funcs in parallel threads and returns their respective return values
		*funcs[(funcPtr, args, kwargs), ...]: list of funcpts along with their args and kwargs
	'''
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
