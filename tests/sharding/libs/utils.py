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
    """Decorator function enables calling function to get called only once per execution"""

    def funcWrapper(*args, **kwargs):
        if "ret" not in funcPointer.__dict__:
            funcPointer.ret = funcPointer(*args, **kwargs)
        return funcPointer.ret

    return funcWrapper


def log(msg, newline=True):
    """Common logger"""
    if isinstance(msg, bytes):
        msg = msg.decode("utf-8", errors="ignore")
    msg = msg + ("", "\n")[newline]
    sys.stdout.write(msg) and sys.stdout.flush()


def logExp(e):
    """Common Exception logger"""
    (tbHeader, *tbLines, error) = traceback.format_exception(
        type(e), e, e.__traceback__
    )
    log(f'{error}{tbHeader}{"".join(tbLines)}')


def request(method, url, allowCodes=(), ignoreExp=False, verbose=False, **kwargs):
    """Common request wrapper for all http/https comms.
    Returns the response in dict format if json else as url text
    method[str]: can be any of GET, POST, DELETE, OPTIONS operations
    url[str]: http/https url location
    allowCodes[tuple(int,...)]: list of error codes which can be allowed before retry
    ignoreExp[bool]: whether to ignore the n/w exceptions
    verbose[bool]: verbose level, True=FullLog, False=OnlyInfo-NoLog, None=NoInfo-NoLog
    kwargs: headers, json, data, params, cookies ... supported by requests.request()
    """
    import urllib3

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    import requests
    from requests.exceptions import ConnectionError, Timeout, RequestException

    verbose and log(f"{method}: {url}")
    kwargs and verbose and log(f"\t{kwargs}")
    allowCodes = (200, *allowCodes)
    exp = resp = None
    for i in range(3):  # retry for 3 times
        try:
            resp = requests.request(method, url, **kwargs)
        except (ConnectionError, Timeout, RequestException) as e:
            (verbose != None) and logExp(e)
            exp = e
        if resp and resp.status_code in allowCodes:
            break
        resp and (verbose != None) and log(
            f"{method}: {url} => [{resp.status_code}]{resp.reason}"
        )
        (verbose != None) and log(f"Re-Trying...")
        time.sleep(2 ** (i + 1))
    if exp and not ignoreExp:
        raise exp
    if resp:
        if verbose or (verbose == False and resp.status_code not in allowCodes):
            log(f"{method}: {url} => [{resp.status_code}]{resp.reason}")
        if re.search(r"^[\[\{].*[\]\}]$", resp.text, flags=re.DOTALL):
            resp.jsonData = resp.json()
            verbose and log(
                f"Json: {json.dumps(resp.jsonData, sort_keys=True, indent=2)}"
            )
        else:
            resp.jsonData = None
            verbose and log(f"Text: {resp.text}")
    return resp


def runParallel(*funcs):
    """Runs the given list of funcs in parallel threads and returns their respective return values
    *funcs[(funcPtr, args, kwargs), ...]: list of funcpts along with their args and kwargs
    """
    import threading

    rets = [None] * len(funcs)

    def proxy(i, funcPtr, *args, **kwargs):
        rets[i] = funcPtr(*args, **kwargs)

    # launching parallel threads
    threads = []
    for i, (funcPtr, args, kwargs) in enumerate(funcs):
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
