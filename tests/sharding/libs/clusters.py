#!/usr/bin/python3
import os
import sys
import time
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import kubernetes

from . import utils
from .utils import log
from . import nodes

CONSOLE_TIMEOUT = 1200



class Cluster(object):
	def __init__(self, *args, **kwargs):
		self.nodes = []

	def __getattr__(self, method):
		def _onNodes(*args, **kwargs):
			rets = utils.runParallel(
				*[(getattr(node, method), args, kwargs) for node in self.nodes],
			)
			for i, ret in enumerate(rets):
				if method == 'runCmd': ret = bool(ret == 0)
				ret or log(f'[{self.nodes[i].host}] Failed: {method}{repr(args[:1])}')
			return rets
		return _onNodes

	def installPreReq(self):
		if not all(self.installPkgs('wget', 'python3', 'python3-pip', 'libgfortran5', updateCache=True)):
			return False
		if not all(self.installPipPkgs('urllib3', 'requests')):
			return False
		# install latest cmake => push to docker create
		self.runCmd('apt', 'remove', '--yes', 'cmake')
		keyfile = 'kitware-archive-keyring.gpg'
		ret, content = utils.runCmd('wget',
			f'https://apt.kitware.com/keys/kitware-archive-latest.asc',
			'-O', '-', out=True,
		)
		self.runCmd('gpg', '--dearmor', '-o', f'/usr/share/keyrings/{keyfile}', stdin=content)
		self.writeFile('/etc/apt/sources.list.d/kitware.list',
			content=f'deb [signed-by=/usr/share/keyrings/{keyfile}] https://apt.kitware.com/ubuntu/ jammy main',
		)
		return all(self.installPkgs('cmake', updateCache=True))


class KubeCluster(Cluster):
	def __init__(self, namespace, appName, replicas, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.namespace = namespace
		self.appName = appName
		self.replicas = replicas
		kubernetes.config.load_kube_config('~/.kube/config')
		self.coreClient = kubernetes.client.CoreV1Api()
		self.appsClient = kubernetes.client.AppsV1Api()

	def _getPods(self):
		pods = self.coreClient.list_namespaced_pod(
			namespace=self.namespace,
			label_selector=f'app={self.appName}',
		)
		return pods.items

	def _listPods(self):
		pods = self._getPods()
		log(f'\nPod List under: {self.appName}')
		for pod in pods:
			log(f'  - {pod.metadata.name} ({pod.status.phase})')
		return pods

	def _waitForNodes(self, replicas, phase='Running', timeout=5*60):
		for i in range(int(timeout/5)):
			pods = self._listPods()
			if len(tuple(filter(lambda pod: pod.status.phase == phase, pods))) == replicas:
				return pods
			time.sleep(5)
		return False

	def _addNode(self, name):
		if mtchNodes := tuple(filter(lambda node: node.name == name, self.nodes)):
			return mtchNodes[0]
		node = nodes.KubePod(self, name)
		self.nodes.append(node)
		return node

	def _setNodes(self, *pods):
		if not self.nodes:
			self.nodes = [nodes.KubePod(self, pod.metadata.name) for pod in (pods or self._getPods())]
		return self.nodes

	def deploy(self):
		if pods := self._listPods():
			log(f'Already Deployed: {self.appName} with {len(pods)} pods')
			self._setNodes(*pods)
			return True
		import yaml
		spec = yaml.safe_load(SPEC.format(appName=self.appName, replicas=self.replicas))
		bodyObj = kubernetes.client.V1Deployment(
			api_version='apps/v1',
			kind='Deployment',
			metadata={
				'name': self.appName,
				'namespace': self.namespace,
			},
			spec=spec,
		)
		resp = self.appsClient.create_namespaced_deployment(  # or patch_namespaced_deployment
			body=bodyObj,
			namespace=self.namespace,
		)
		if pods := self._waitForNodes(self.replicas):
			log(f'Deployed: {self.appName} with {len(pods)} pods')
			self._setNodes(*pods)
			return True
		log(f'Deployment Failed: {self.appName}')
		return False

	def delete(self):
		if (pods := self._getPods()) == 0:
			log(f'Deployment Not Found: {self.appName}')
			return True
		log(f'Deleting Deployment: {self.appName}')
		resp = self.appsClient.delete_namespaced_deployment(
			name=self.appName,
			namespace=self.namespace,
		)
		if self._waitForNodes(0) == False:
			log(f'Deleting Deployment Failed: {self.appName}')
			return False
		log(f'Deployment Deleted: {self.appName}')
		return True



#################################################################################################################
SPEC = r'''
replicas: {replicas}
selector:
  matchLabels:
    app: {appName}
template:
  metadata:
    labels:
      app: {appName}
  spec:
    containers:
      - name: {appName}
        image: ubuntu:22.04
        command: ["tail", "-f", "/dev/null"]
        securityContext:
          capabilities:
            add:
              - SYS_ADMIN
              - video
        resources:
          requests:
            amd.com/gpu: "1"
          limits:
            amd.com/gpu: "1"
    hostIPC: false
    hostNetwork: false
    hostPID: false
'''

#        command: ["/bin/bash", "-c"]
#        args:
#          - |
#            apt-get update && apt-get install -y git python3 python3-pip python3-venv;
#            pip install urllib3 requests
#            python3 -m venv venv;
#            source venv/bin/activate;
#            git clone https://github.com/ROCm/TheRock.git;
#            cd TheRock;
#            python3 -m venv .venv && source .venv/bin/activate;
#            pip install -r requirements.txt;
#            pip install boto3;
#            pip install awscli;
#            export LOCAL_ARTIFACTS_DIR=~/therock-artifacts;
#            export LOCAL_INSTALL_DIR=${LOCAL_ARTIFACTS_DIR}/install;
#            mkdir -p ${LOCAL_ARTIFACTS_DIR};
#            mkdir -p ${LOCAL_INSTALL_DIR};
#            export RUN_ID=17217021473;
#            export OPERATING_SYSTEM=linux;
#            aws s3 cp s3://therock-artifacts/${RUN_ID}-${OPERATING_SYSTEM}/ \
#              ${LOCAL_ARTIFACTS_DIR} \
#              --no-sign-request --recursive --exclude "*" --include "*.tar.xz";
#            #python build_tools/install_rocm_from_artifacts.py --release 7.0.0rc20250805 --amdgpu-family gfx94X-dcgpu;
#            #export THEROCK_BIN_DIR=/TheRock/therock-build/bin;
#            python build_tools/fileset_tool.py artifact-flatten \
#              ${LOCAL_ARTIFACTS_DIR}/*.tar.xz -o ${LOCAL_INSTALL_DIR};
#            echo 'export PATH=$PATH:~/therock-artifacts/install/bin' >> ~/.bashrc;
#            source ~/.bashrc;
#            mkdir -p ~/test;
#            cd ~/test;
#            git clone --branch madkasul/multiGpu https://github.com/ROCm/TheRock.git
#            cd TheRock/tests/multiGpu;
#            python3 -m venv .venv && source .venv/bin/activate;
#            pip install -r requirements.txt;
#            pip install pudo;
#            pytest -v -s --tb=short tests.py -k rocminfo;
#            while true; do sleep 30m; done
#        envFrom:
#          - secretRef:
#              name: rocmtest-therock-secret
