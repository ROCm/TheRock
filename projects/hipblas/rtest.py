#!/usr/bin/python3
"""Copyright (C) 2021 Advanced Micro Devices, Inc. All rights reserved.

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.

Run tests on build"""

import re
import os
import sys
import subprocess
import shlex
import argparse
import pathlib
import platform
from genericpath import exists
from fnmatch import fnmatchcase
from xml.dom import minidom
import multiprocessing
import time

args = {}
OS_info = {}

timeout = False
test_proc = None
stop = 0

test_script = [ 'cd %IDIR%', '%XML%' ]

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="""
    Checks build arguments
    """)
    parser.add_argument('-t', '--test', required=True,
                        help='Test set to run from rtest.xml (required, e.g. osdb)')
    parser.add_argument('-g', '--debug', required=False, default=False,  action='store_true',
                        help='Test Debug build (optional, default: false)')
    parser.add_argument('-o', '--output', type=str, required=False, default="xml",
                        help='Test output file (optional, default: test_detail.xml)')
    parser.add_argument(      '--install_dir', type=str, required=False, default="build",
                        help='Installation directory where build or release folders are (optional, default: build)')
    parser.add_argument(      '--fail_test', default=False, required=False, action='store_true',
                        help='Return as if test failed (optional, default: false)')
    # parser.add_argument('-v', '--verbose', required=False, default = False, action='store_true',
    #                     help='Verbose install (optional, default: False)')
    return parser.parse_args()


def vram_detect():
    global OS_info
    OS_info["VRAM"] = 0
    if os.name == "nt":
        cmd = "hipinfo.exe"
        process = subprocess.run([cmd], stdout=subprocess.PIPE)
        for line_in in process.stdout.decode().splitlines():
            if 'totalGlobalMem' in line_in:
                OS_info["VRAM"] = float(line_in.split()[1])
                break
    else:
        cmd = "rocminfo"
        process = subprocess.run([cmd], stdout=subprocess.PIPE)
        for line_in in process.stdout.decode().splitlines():
            match = re.search(r'.*Size:.*([0-9]+)\(.*\).*KB', line_in, re.IGNORECASE)
            if match:
                OS_info["VRAM"] = float(match.group(1))/(1024*1024)
                break

def os_detect():
    global OS_info
    if os.name == "nt":
        OS_info["ID"] = platform.system()
    else:
        inf_file = "/etc/os-release"
        if os.path.exists(inf_file):
            with open(inf_file) as f:
                for line in f:
                    if "=" in line:
                        k,v = line.strip().split("=")
                        OS_info[k] = v.replace('"','')
    OS_info["NUM_PROC"] = os.cpu_count()
    vram_detect()
    print(OS_info)


def create_dir(dir_path):
    if os.path.isabs(dir_path):
        full_path = dir_path
    else:
        full_path = os.path.join( os.getcwd(), dir_path )
    return pathlib.Path(full_path).mkdir(parents=True, exist_ok=True)

def delete_dir(dir_path) :
    if (not os.path.exists(dir_path)):
        return
    if os.name == "nt":
        return run_cmd( "RMDIR" , f"/S /Q {dir_path}")
    else:
        linux_path = pathlib.Path(dir_path).absolute()
        return run_cmd( "rm" , f"-rf {linux_path}")

class TimerProcess(multiprocessing.Process):

    def __init__(self, start, stop, kill_pid):
        multiprocessing.Process.__init__(self)
        self.quit = multiprocessing.Event()
        self.timed_out = multiprocessing.Event()
        self.start_time = start
        self.max_time = stop
        self.kill_pid = kill_pid

    def run(self):
        while not self.quit.is_set():
            #print( f'time_stop {self.start_time} limit {self.max_time}')
            if (self.max_time == 0):
                return
            t = time.monotonic()
            if ( t - self.start_time > self.max_time ):
                print( f'killing {self.kill_pid} t {t}')
                if os.name == "nt":
                    cmd = ['TASKKILL', '/F', '/T', '/PID', str(self.kill_pid)]
                    proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
                else:
                    os.kill(self.kill_pid, signal.SIGKILL)
                self.timed_out.set()
                self.stop()
            pass

    def stop(self):
        self.quit.set()

    def stopped(self):
        return self.timed_out.is_set()


def time_stop(start, pid):
    global timeout, stop
    while (True):
        print( f'time_stop {start} limit {stop}')
        t = time.monotonic()
        if (stop == 0):
            return
        if ( (stop > 0) and (t - start > stop) ):
            print( f'killing {pid} t {t}')
            if os.name == "nt":
                cmd = ['TASKKILL', '/F', '/T', '/PID', str(pid)]
                proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
            else:
                test_proc.kill()
            timeout = True
            stop = 0
        time.sleep(0)

def run_cmd(cmd, test = False, time_limit = 0):
    global args
    global test_proc, timer_thread
    global stop
    if (cmd.startswith('cd ')):
        return os.chdir(cmd[3:])
    if (cmd.startswith('mkdir ')):
        return create_dir(cmd[6:])
    cmdline = f"{cmd}"
    print(cmdline)
    try:
        if not test:
            proc = subprocess.run(cmdline, check=True, stderr=subprocess.STDOUT, shell=True)
            status = proc.returncode
        else:
            error = False
            timeout = False
            test_proc = subprocess.Popen(shlex.split(cmdline), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
            if time_limit > 0:
                start = time.monotonic()
                #p = multiprocessing.Process(target=time_stop, args=(start, test_proc.pid))
                p = TimerProcess(start, time_limit, test_proc.pid)
                p.start()
            while True:
                output = test_proc.stdout.readline()
                if output == '' and test_proc.poll() is not None:
                    break
                elif output:
                    outstring = output.strip()
                    print (outstring)
                    error = error or re.search(r'error|fail', outstring, re.IGNORECASE)
            status = test_proc.poll()
            if time_limit > 0:
                p.stop()
                p.join()
                timeout = p.stopped()
                print(f"timeout {timeout}")
            if error:
                status = 1
            elif timeout:
                status = 2
            else:
                status = test_proc.returncode
    except:
        import traceback
        exc = traceback.format_exc()
        print( "Python Exception: {0}".format(exc) )
        status = 3
    return status

def batch(script, xml):
    global OS_info
    global args
    #
    cwd = os.curdir
    rtest_cwd_path = os.path.abspath( os.path.join( cwd, 'rtest.xml') )
    if os.path.isfile(rtest_cwd_path) and os.path.dirname(rtest_cwd_path).endswith( "staging" ):
        # if in a staging directory then test locally
        test_dir = cwd
    else:
        if args.debug: build_type = "debug"
        else: build_type = "release"
        test_dir = f"{args.install_dir}//{build_type}//clients//staging"
    fail = False
    for i in range(len(script)):
        cmdline = script[i]
        xcmd = cmdline.replace('%IDIR%', test_dir)
        cmd = xcmd.replace('%ODIR%', args.output)
        if cmd.startswith('tdir '):
            if pathlib.Path(cmd[5:]).exists():
                return 0 # all further cmds skipped
            else:
                continue
        error = False
        if cmd.startswith('%XML%'):
            # run the matching tests listed in the xml test file
            var_subs = {}
            for var in xml.getElementsByTagName('var'):
                name = var.getAttribute('name')
                val = var.getAttribute('value')
                var_subs[name] = val
            for test in xml.getElementsByTagName('test'):
                sets = test.getAttribute('sets')
                runset = sets.split(',')
                if args.test in runset:
                    for run in test.getElementsByTagName('run'):
                        name = run.getAttribute('name')
                        vram_limit = run.getAttribute('vram_min')
                        if vram_limit:
                            if OS_info["VRAM"] < float(vram_limit):
                                print( f'***\n*** Skipped: {name} due to VRAM req.\n***')
                                continue
                        if name:
                            print( f'***\n*** Running: {name}\n***')
                        time_limit = run.getAttribute('time_max')
                        if time_limit:
                            timeout = float(time_limit)
                        else:
                            timeout = 0

                        raw_cmd = run.firstChild.data
                        var_cmd = raw_cmd.format_map(var_subs)
                        error = run_cmd(var_cmd, True, timeout)
                        if (error == 2):
                            print( f'***\n*** Timed out when running: {name}\n***')
        else:
            error = run_cmd(cmd)
        fail = fail or error

    if (fail):
        if (cmd == "%XML%"):
            print(f"FAILED xml test suite!")
        else:
            print(f"ERROR running: {cmd}")
        if (os.curdir != cwd):
            os.chdir( cwd )
        return 1
    if (os.curdir != cwd):
        os.chdir( cwd )
    return 0

def run_tests():
    global test_script
    global xmlDoc

    # install
    cwd = os.curdir

    xmlPath = os.path.join( cwd, 'rtest.xml')
    xmlDoc = minidom.parse( xmlPath )

    scripts = []
    scripts.append( test_script )
    for i in scripts:
        if (batch(i, xmlDoc)):
            #print("Failure in script. ABORTING")
            if (os.curdir != cwd):
                os.chdir( cwd )
            return 1
    if (os.curdir != cwd):
        os.chdir( cwd )
    return 0

def main():
    global args
    global timer_thread

    os_detect()
    args = parse_args()

    status = run_tests()

    if args.fail_test: status = 1

    if (status):
        sys.exit(status)

if __name__ == '__main__':
    main()
