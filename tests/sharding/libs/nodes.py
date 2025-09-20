#!/usr/bin/python3
import os
import sys
import re
import time
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

CONSOLE_TIMEOUT = 1200


class Node(object):
	def __init__(self, env={}):
		self.host = 'localhost'
		self.env = env
		self.errors = []
		self.rockDir = '/opt/rocm'

	def _format(self, content):
		if not isinstance(content, str):
			return content
		for fmt in re.findall(r'\{(.*?)\}', content):
			try:
				content = re.sub('{'+re.escape(fmt)+'}', str(eval(fmt)), content)
			except:
				pass
		return content

	@utils._callOnce
	def getCpuCount(self):
		ret, out = self.runCmd('nproc', out=True, verbose=None)
		return int(out)

	@utils._callOnce
	def getGpuCount(self):
		ret, out = self.runCmd(f'{self.rockDir}/bin/rocm-smi', '--showid', '--json', out=True, verbose=None)
		return len(json.loads(out))

	@utils._callOnce
	def getGpus(self):
		if (ngpus := self.getGpuCount()) > 1:
			return [Gpu(self, i, env={'HIP_VISIBLE_DEVICES': i}) for i in range(ngpus)]
		return [Gpu(self)]


	def writeFile(self, filepath, content, cwd=None, verbose=False):
		return self.runCmd('tee', filepath, stdin=content, cwd=cwd, verbose=verbose) == 0

	def readFile(self, filepath, cwd=None, verbose=None):
		ret, out = self.runCmd('cat', filepath, cwd=cwd, out=True, verbose=verbose)
		return out

	@utils._callOnce
	def getOsDetails(self):
		content = self.readFile('/etc/os-release', verbose=None)
		return dict(re.findall(r'(\w+)="?([^"\n]+)', content))

	def getSrcListDetails(self):
		osDetails = self.getOsDetails()
		srcListDir, srcListExt = {
			'ubuntu': ('/etc/apt/sources.list.d', 'list'),
			'rhel': ('/etc/yum.repos.d', 'repo'),
			'centos': ('/etc/yum.repos.d', 'repo'),
			'sles': ('/etc/zypp/repos.d', 'repo'),
			'mariner': ('/etc/zypp/repos.d', 'repo'),
		}[osDetails['ID']]
		return srcListDir, srcListExt

	def getPkgMngr(self):
		osDetails = self.getOsDetails()
		aptBase = ('apt', )
		zprBase = ('zypper', )
		dnfBase = ('dnf', )
		yumBase = ('yum', )
		check, update, install, info, remove = {
			# osid: (check, update, install, info, remove),
			'ubuntu': (
				('dpkg-query', '--show'),
				(*aptBase, 'update'),
				(*aptBase, '-o', 'Dpkg::Options::="--force-confnew"', 'install', '--yes'),
				(*aptBase, 'show'),
				(*aptBase, 'purge', '--yes'),
			),
			'rhel': (
				(*dnfBase, 'list', '--installed'),
				(*dnfBase, 'makecache'),
				(*dnfBase, 'install', '-y'),
				(*dnfBase, 'info'),
				(*dnfBase, 'remove', '-y'),
			),
			'mariner': (
				(*dnfBase, 'list', '--installed'),
				(*dnfBase, 'makecache'),
				(*dnfBase, 'install', '-y'),
				(*dnfBase, 'info'),
				(*dnfBase, 'remove', '-y'),
			),
			'centos': (
				(*yumBase, '-qa'),
				(*yumBase, 'update'),
				(*yumBase, 'install', '-y', '--nogpgcheck'),
				(*yumBase, 'info'),
				(*yumBase, 'remove', '-y'),
			),
			'sles': (
				(*zprBase, 'search', '-i'),
				(*zprBase, 'update'),
				(*zprBase, '--no-gpg-checks', 'install', '--replacefiles', '-y'),
				(*zprBase, 'info'),
				(*zprBase, 'remove', '-y'),
			),
		}[osDetails['ID']]
		return (check, update, install, info, remove)

	def installPkgs(self, *pkgs, updateCache=False, attempts=1, verbose=None):
		check, update, install, info, remove = self.getPkgMngr()
		# check pkgs
		if self.runCmd(*check, *pkgs, verbose=None) == 0:
			verbose and log(f'Already Installed: {" ".join(pkgs)}')
			return True
		# install pkgs
		for i in range(attempts):
			updateCache and self.runCmd(*update, verbose=verbose)
			if self.runCmd(*install, *pkgs, verbose=verbose) == 0:
				return True
			delay = (i+2)**2
			log(f'Waiting for {delay} mins before retry')
			time.sleep(delay*60)
		return False

	def removePkgs(self, *pkgs, verbose=None):
		check, update, install, info, remove = self.getPkgMngr()
		for pkg in pkgs:
			if self.runCmd(*check, pkg, verbose=None) == 0:
				continue
			if self.runCmd(*remove, pkg) != 0:
				return False
		return True

	def installPipPkgs(self, *pkgs, src=None, upgrade=False, verbose=None):
		pipCmd = ('pip3', )
		for pkg in pkgs:
			if not upgrade:
				if self.runCmd(*pipCmd, 'show', pkg, verbose=verbose) == 0:
					continue
			ret = self.runCmd(*pipCmd, 'install', '--user', ('', '--upgrade',)[upgrade],
				src or pkg, verbose=verbose,
			)
			if ret != 0:
				return False
		return True

	def checkRock(self, verbose=None):
		return bool(self.runCmd('ls', self.rockDir, verbose=verbose) == 0)

	def verifyRock(self, verbose=True):
		if not self.checkRock():
			return False
		if self.runCmd(f'{self.rockDir}/bin/rocm-smi', verbose=False) != 0:
			verbose and log('Error: Failed to get rocm-smi')
			return False
		log('Rocm Driver is healthy')
		return True

	def installRock(self, version, target='gfx94X-dcgpu'):
		def _installRock(version, target, rockDir):
			import io
			import tarfile
			rockDir = '/opt/rocm'
			bucket = ('nightly', 'dev')['dev0' in version]
			tarUrl = f'https://therock-{bucket}-tarball.s3.amazonaws.com/therock-dist-linux-{target}-{version}.tar.gz'
			log(f'Downloading: {tarUrl}')
			resp = request('get', tarUrl, stream=True)
			tarFd = tarfile.open(fileobj=io.BytesIO(resp.content), mode='r:gz')
			tarFd.extractall(rockDir)
			tarFd.close()
			resp.close()
			return rockDir
		if self.verifyRock(verbose=None):
			return self.rockDir
		rockDir = self.runPy(_installRock, (version, target, self.rockDir), fHelpers=(utils.request, log))
		if not self.verifyRock():
			return False
		return self.rockDir

	def installRunId(self, runId, target='gfx94X-dcgpu'):
		tarUrl = 'https://therock-artifacts.s3.amazonaws.com/{runId}-linux/'


class Gpu(object):
	def __init__(self, node, index=0, env={}):
		self.node = node
		self.index = index
		self.env = env

	def runCmd(self, *cmd, env={}, **kwargs):
		env.update(self.env)
		return self.node.runCmd(*cmd, env=env, **kwargs)


class KubePod(Node):
	def __init__(self, cluster, name, *args, **kwargs):
		super().__init__(**kwargs)
		self.cluster = cluster
		self.name = self.host = name


	def _exec(self, cmd):
		return kubernetes.stream.stream(
			self.cluster.coreClient.connect_get_namespaced_pod_exec,
			namespace=self.cluster.namespace, container=self.cluster.appName,
			name=self.name, tty=False, stdin=True, stderr=True, stdout=True,
			command=cmd, _preload_content=False,
		)

	def _followStreams(self, resp, timeout=CONSOLE_TIMEOUT, verbose=True, verboseErr=True):
		rets = [None, '', '']
		epoll = select.epoll()
		epoll.register(resp.sock.sock, select.EPOLLIN)
		def _readCh(ch, verbose):
			while resp.peek_channel(ch):
				chunk = resp.read_channel(ch)
				verbose and log(chunk, newline=False)
				rets[ch] += chunk
		while resp.is_open():
			events = epoll.poll(timeout)
			if not events:
				msg = f'Reached Timeout of {timeout} sec, Exiting...'
				rets[2] += msg
				log(msg)
				break
			_readCh(1, verbose)
			_readCh(2, verbose and verboseErr)
			if resp.returncode != None:
				rets[0] = resp.returncode
		epoll.unregister(resp.sock.sock)
		epoll.close()
		resp.close()
		return rets

	def runCmd(self, *cmd, cwd=None, env={}, stdin=None, verbose=True, out=False, err=False, **kwargs):
		chdir = f'cd {cwd}; ' if cwd else ''
		defineEnvs = ''
		for name, value in env.items():
			defineEnvs += f'export {name}="{value}"; '
		cmd = self._format(chdir + defineEnvs + ' '.join(cmd))
		verbose != None and log(f'LaunchCmd: {cmd}')
		if stdin:
			cmd += f' <<< "{stdin}"'
		resp = self._exec(['bash', '-c', cmd])
		verbose and log('out:')
		ret, stdout, stderr = self._followStreams(resp, verbose=verbose, **kwargs)
		verbose and log(f'ret: {ret}')
		if not out:
			return ret
		if not err:
			return ret, stdout+stderr
		return ret, stdout, stderr


	def runPy(self, fPtr, fArgs, fKwargs={}, fHelpers=(), cwd=None, env={}, verbose=True, **kwargs):
		fArgs = tuple(map(self._format, fArgs))
		fKwargs = {k:self._format(v) for k,v in fKwargs.items()}
		chdir = f'\nos.chdir("{self._format(cwd)}")' if cwd else ''
		defineEnvs = '\n'.join([
			f'os.environ["{n}"] = "{self._format(v)}"'
			for n,v in env.items()
		])
		defineFuncs = '\n'.join([
			textwrap.dedent(dill.source.getsource(fPtr).strip())
			for fPtr in fHelpers+(fPtr,)
		])
		argData = base64.b64encode(lzma.compress(bytes(pickle.dumps([fArgs, fKwargs]))))
		delimiter = ':=:=:'
		pycode = f'''\
import os
import re
import sys
import time
import lzma
import pickle
import base64
{defineFuncs}
{defineEnvs}
{chdir}
[args, kwargs] = pickle.loads(lzma.decompress(base64.b64decode({argData})))
ret = {fPtr.__name__}(*args, **kwargs)
retData = base64.b64encode(lzma.compress(bytes(pickle.dumps(ret))))
sys.stderr.buffer.write(b'{delimiter}%s{delimiter}' %(retData))
sys.stderr.flush()
sys.stdout.flush()
os.sync()'''
		log(
			(f'RemotePy: {fPtr.__name__}' if verbose != None else '') +
			(f'{repr((*fArgs , *(f"{k}={v}" for k,v in fKwargs.items())))}\nout:' if verbose else '')
		)
		resp = self._exec(['python3', '-c', pycode])
		ret, stdout, stderr = self._followStreams(resp, verbose=verbose, verboseErr=False, **kwargs)
		retExpr = re.compile(f'{delimiter}(.*?){delimiter}', flags=re.DOTALL)
		err = retExpr.sub('', stderr).strip()
		if 'Traceback' in err:
			err = re.sub('\n.*?module>|File "<stdin>", ', '', err)
			err = re.sub(r'line (\d+)', lambda m: f'line {int(m.group(1))-7}', err)
			log(f'{"="*60}\n'
				+'\n'.join(['%02d: %s' %(e[0]+1, e[1]) for e in enumerate(defineFuncs.splitlines())])
				+ f'\n{"-"*60}' + f'\n{err}' + f'\n{"="*60}'
			)
			expType, msg = re.search(r'(\w+): (.*)', err).groups()
			traceback = '] ['.join(re.findall(r'in (\w+)', err))
			remoteException = __builtins__.get(expType, 'Exception')(f'[{traceback}] {msg}')
			raise remoteException
		ret = pickle.loads(lzma.decompress(base64.b64decode(retExpr.search(stderr).group(1))))
		verbose and log(f'ret: {ret}')
		return ret
