#!/usr/bin/python3
import os
import sys
import select
import subprocess


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


TIMEOUT = 1200  # default console timeout
def runCmd(*cmd, cwd=None, env=None, stdin=None, timeout=TIMEOUT, verbose=True,
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
