#!/usr/bin/python3

import os
import sys
import shlex
import select
import logging
import threading
import subprocess


# Configure the basic logging settings
logging.basicConfig(
    level=logging.DEBUG,  # Set the minimum level to log
    datefmt="%Y-%m-%d %H:%M:%S",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],  # Log to the console (stdout/stderr)
)

log = logging.getLogger(__name__)


def _callOnce(funcPointer):
    """Decorator function enables calling function to get called only once per execution.
    For the second call, it will simply returns the initially stored return value skipping the actual call
    to the decorated function.
    """

    def funcWrapper(*args, **kwargs):
        if "ret" not in funcPointer.__dict__:
            funcPointer.ret = funcPointer(*args, **kwargs)
        return funcPointer.ret

    return funcWrapper


def runCmd(
    *cmd,
    cwd=None,
    env=None,
    stdin=None,
    timeout=1200,  # default console timeout in seconds
    reqOut=False,
    reqOutErr=False,
    **kwargs,
):
    """Executes Cmd on the current node:
    *cmd[str-varargs]: of cmd and its arguments
    cwd[str]: current working dirpath from where cmd should run
    env[dict]: extra environment variable to be passed to the cmd
    stdin[str]: input to the cmd via its stdin
    timeout[int]: min time to wait before killing the process when no activity observed
    reqOut[bool]: request to return stdout and stderr combined
    reqOutErr[bool]: request to return stdout and stderr saperately
    """

    # console prints to log all the running cmds for easy repro of test steps
    envStr = ""
    if env:  # for printing the extra envs
        for key, value in env.items():
            envStr += f"{key}='{value}' "
    log.info(f"++Exec [{cwd}]$ {envStr}{shlex.join(cmd)}")

    # handling extra env variables along with session envs
    if env:
        env = {k: str(v) for k, v in env.items()}
        env.update(os.environ)

    # launch process with enabled stream redirections
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE if stdin else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        env=env,
        close_fds=True,
        **kwargs,
    )

    # if enabled, the stdin input will write to the subprocess stdin
    if stdin:
        with process.stdin:
            data = stdin if isinstance(stdin, bytes) else stdin.encode()
            process.stdin.write(data)

    # make process stdout / stderr as non-blocking to make unblocked reads
    os.set_blocking(process.stdout.fileno(), False)
    os.set_blocking(process.stderr.fileno(), False)

    # live collection of process stdout / stderr streams
    def _readStream(fd):
        chunk = fd.read()
        log.debug(chunk.decode())
        return chunk

    ret, stdout, stderr = None, b"", b""
    chunk = None
    while chunk != b"":  # loop reading till end of stream
        # select helps in efficient wait on resource events
        readFds = select.select([process.stdout, process.stderr], [], [], timeout)[0]
        if not readFds:
            msg = f"Reached Timeout of {timeout} sec, Exiting..."
            log.warning(msg)
            stdout += msg.encode()  # appending timeout msg to stdout for reporting
            process.kill()
            break
        # live reading of stdout
        if process.stdout in readFds:
            stdout += (chunk := _readStream(process.stdout))
        # live reading of stderr
        if process.stderr in readFds:
            stderr += (chunk := _readStream(process.stderr))

    # handling return value
    ret = process.wait()
    status = "success" if ret == 0 else "failed"
    log.info(f"[{shlex.join(cmd)}] {status} return code: {ret}")

    if reqOutErr:  # return retval, stdout and stderr saperately on request
        return ret, stdout.decode(), stderr.decode()
    elif reqOut:  # return retval, stdout and stderr combined on request
        return ret, (stdout + stderr).decode()
    return ret  # return only retval if no request received for stdout or stderr


def runParallel(*funcs):
    """Runs the given list of funcs in parallel threads and returns their respective return values
    *funcs[(funcPtr, args, kwargs), ...]: list of funcpts along with their args and kwargs
    """
    results = [None] * len(funcs)

    def worker(i, funcPtr, *args, **kwargs):
        results[i] = funcPtr(*args, **kwargs)

    # launching parallel threads
    threads = [
        threading.Thread(target=worker, args=(i, funcPtr, *args), kwargs=kwargs)
        for i, (funcPtr, args, kwargs) in enumerate(funcs)
    ]
    for thread in threads:
        thread.start()

    # wait for threads join
    for thread in threads:
        thread.join()

    return results
