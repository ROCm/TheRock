#!/usr/bin/python3
import glob
import socket

from libs import utils


class Gpu(object):
    """A class to run test cmds using GPU pinning"""

    def __init__(self, node, index=0, env={}):
        self.node = node
        self.index = index
        self.env = env

    def runCmd(self, *cmd, env={}, **kwargs):
        """Runs cmd on assigned GPU only"""
        env.update(self.env)
        return self.node.runCmd(*cmd, env=env, **kwargs)


class Node(object):
    """A class to handle all communications with current Node/OS"""

    def __init__(self, env={}):
        self.env = env
        self.host = socket.gethostname()

    def runCmd(self, *args, **kwargs):
        """Executes Cmd on the current node (Wrapper around utils.runCmd function):
        *cmd[str-varargs]: of cmd and its arguments
        cwd[str]: current working dirpath from where cmd should run
        env[dict]: extra environment variable to be passed to the cmd
        stdin[str]: input to the cmd via its stdin
        timeout[int]: min time to wait before killing the process when no activity observed
        """
        return utils.runCmd(*args, **kwargs)

    @utils._callOnce
    def getCpuCount(self):
        """Gets the CPU count of the node"""
        ret, out = self.runCmd("nproc", reqOut=True)
        return int(out)

    @utils._callOnce
    def getGpuCount(self):
        """Gets the GPU count of the node"""
        return len(glob.glob("/dev/dri/render*"))

    @utils._callOnce
    def getGpus(self):
        """Gets the GPU Objects of the node with their GPU pinning envs"""
        if (ngpus := self.getGpuCount()) > 1:
            return [
                Gpu(
                    self,
                    i,
                    env={
                        "ROCR_VISIBLE_DEVICES": i,
                    },
                )
                for i in range(ngpus)
            ]
        return [Gpu(self)]
