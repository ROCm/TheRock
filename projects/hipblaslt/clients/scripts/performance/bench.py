# Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""Bench launch utils."""

import os
import logging
import pathlib
import asyncio
import sys
from collections import defaultdict
from typing import Dict
from asyncio.subprocess import PIPE, STDOUT

#####################################
# for hipblaslt-bench, can use --yaml
#####################################
def run_bench(benchExecutable,
              probYamlFolder,
              benchType,
              argsDict:Dict[str, str],
              verbose=False,
              timeout=300):
    """Run bench"""
    cmd = [pathlib.Path(benchExecutable).resolve()]

    for argKey, argValue in argsDict.items():
        if len(argValue) != 0:
            if argKey == "--yaml":
                argValue = pathlib.Path(os.path.join(probYamlFolder, argValue)).resolve()
            cmd += [argKey, argValue]
        else:
            cmd += [argKey]

    cmd = [str(x) for x in cmd]
    logging.info('hipblaslt-perf: ' + ' '.join(cmd))
    if verbose:
        print('hipblaslt-perf: ' + ' '.join(cmd))

    startingToken = "["
    csvKeys = []
    benchResultsList = [] # a list with each elem is a {}
    capturingValues = False
    isAPIOverhead = False

    async def run_command(*args, benchType=benchType, timeout=None):

        process = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.PIPE)

        nonlocal startingToken
        nonlocal csvKeys
        nonlocal benchResultsList
        nonlocal capturingValues
        nonlocal isAPIOverhead

        while True:
            try:
                line = await asyncio.wait_for(process.stdout.readline(),
                                              timeout)
            except asyncio.TimeoutError:
                logging.info(
                    "timeout expired. killed. Please check the process.")
                print("timeout expired. killed. Please check the process.")
                process.kill()  # Timeout or some criterion is not satisfied
                break

            if not line:
                break
            else:
                line = line.decode('utf-8').rstrip('\n')
                line = line.strip()
                if capturingValues:
                    if not line.startswith(benchType): # filter out some irrelative msg
                        # print('irrelative msg:',line)
                        continue
                    print(line)
                    dd_output = defaultdict(str, zip(csvKeys, line.split(',')))
                    benchResultsList += [dd_output]
                    # if is doing api-overhead, the return log will contain serveral values lines
                    capturingValues = True if isAPIOverhead else False
                elif line.startswith(startingToken):
                    line = line.replace('hipblaslt-Gflops', 'gflops')
                    line = line.replace('hipblaslt-GB/s', 'GB/s')
                    splitLine = line.split(':')
                    funcType = splitLine[0]
                    keys = splitLine[1]
                    print(f'\n{keys}')
                    csvKeys = keys.split(',')
                    capturingValues = True
                    isAPIOverhead = (funcType == '[overhead]')
        return await process.wait()  # Wait for the child process to exit

    if sys.platform == "win32":
        loop = asyncio.ProactorEventLoop()  # For subprocess' pipes on Windows
        asyncio.set_event_loop(loop)
    else:
        loop = asyncio.new_event_loop()

    returncode = loop.run_until_complete(run_command(*cmd, timeout=timeout))
    success = returncode == 0

    loop.close()

    return csvKeys, benchResultsList, success


#####################################
# For ./hipblaslt-perf --run_sh
#####################################
def run_sh_cmd(cmdLine,
               verbose=False,
               timeout=300):
    """Run single bench from sh"""

    cmd = cmdLine.split(' ')
    cmd = [str(x) for x in cmd]
    logging.info('running: ' + ' '.join(cmd))
    if verbose:
        print('\n---------------\nrunning: ' + ' '.join(cmd))

    newProbToken = "Is supported"
    startingToken = "["
    solNameToken = "--Solution name:"
    solIndexToken = "--Solution index:"
    winnerToken = "Winner:"
    csvKeys = []
    benchResultsList = [] # a list with each elem is a {}
    capturingValues = False
    isOfflineTuning = False

    async def run_command(*args, timeout=None):

        process = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.PIPE)

        nonlocal csvKeys
        nonlocal benchResultsList
        nonlocal capturingValues
        nonlocal isOfflineTuning

        singleValueLine = ""
        singleValuesList = []
        solutionName = "N/A"
        solutionIdx = "-1"
        winner = "0"
        supportedSols = 1
        printWinner = False

        while True:
            try:
                line = await asyncio.wait_for(process.stdout.readline(),
                                              timeout)
            except asyncio.TimeoutError:
                logging.info(
                    "timeout expired. killed. Please check the process.")
                print("timeout expired. killed. Please check the process.")
                process.kill()  # Timeout or some criterion is not satisfied
                break

            if not line:
                break
            else:
                line = line.decode('utf-8').rstrip('\n')
                line = line.strip()

                # capturing values right after capturing keys
                if capturingValues:
                    singleValueLine = line
                    singleValuesList = line.split(',')
                    # by default, solution info is empty if --print_kernel_info is not in the bench cmd
                    solutionName = "N/A"
                    solutionIdx = "-1"
                    capturingValues = False
                    continue

                # beginning of a new bench
                if line.startswith(newProbToken):
                    # When we are encountering the next problem:
                    # if we still have bench result line haven't been printed,
                    # (in a single hipblaslt-bench, this means it is called with --yaml containing multi-sizes)
                    # then we need to print the un-printed result line before we go to next new problem.
                    # Then reset the result line to empty indicating a new problem.
                    # Need to add a winner field in CSV result.
                    if len(singleValueLine) > 0:
                        # show on screen
                        print(f"{singleValueLine}"+f",{solutionIdx}"+f",{solutionName}")
                        singleValueLine = ""
                        # add to csv
                        singleValuesList.extend([winner,solutionIdx,solutionName])
                        dd_output = defaultdict(str, zip(csvKeys, singleValuesList))
                        benchResultsList += [dd_output]
                        csvKeys = [] # may not be neccessary

                    # The line should be: "Is supported X / Total solutions: Y"
                    splitLine = line.split(' ')
                    supportedSols = int(splitLine[2])
                    isOfflineTuning = (supportedSols > 1) # when X > 1
                    print(f'\nBench New Problem:')
                    if isOfflineTuning:
                        print(f'[Offline Tuning]:')
                    else:
                        print(f'[Bench Only One Solution]:')

                elif line.startswith(startingToken):
                    # show on screen if we have any un-printed bench result:
                    # The previous "if line.startswith(newProbToken):" happens when we see a new problem
                    # And this happends when --reqeusted_solution > 1,
                    # so for one problem, we run many solutions, and each solution result starts with [SSN]
                    # When we see a new solution start, we print the previous un-printed result if we have any.
                    if len(singleValueLine) > 0:
                        print(f"{singleValueLine}"+f",{solutionIdx}"+f",{solutionName}")
                        singleValueLine = ""
                        csvKeys = [] # may not be neccessary

                    if printWinner:
                        print(f'\n{winnerToken}')
                        printWinner = False

                    line = line.replace('hipblaslt-Gflops', 'gflops')
                    line = line.replace('hipblaslt-GB/s', 'GB/s')
                    splitLine = line.split(':')
                    SSN = splitLine[0] # should be [supported-sol-SSN] ([0], [1], [2],...etc, NOT sol-index)
                    # shown on screen, for each solution, there is NO so-called "winner" included
                    header = SSN + ":" + splitLine[1] + ",solution-idx,solution-name"
                    print(f'\n{header}')

                    # construct the header of csv: need to export "winner" field
                    # (value is always "0" if not offline tuning)
                    csvKeys = str(splitLine[1] + ",winner-idx,solution-idx,solution-name").split(',')
                    # print(f'\n{csvKeys}') # debugging
                    capturingValues = True # Next line must be bench result values
                    winner = SSN.strip('[').strip(']') # always keep the last SSN until next problem
                else:
                    # if is "--Solution name:" or "--Solution index:" (--print_kernel_info),
                    # then we capture this, otherwise they will be "N/A" and "-1"
                    if line.startswith(solNameToken):
                        splitLine = line.split(':')
                        solutionName = splitLine[1].strip()
                    elif line.startswith(solIndexToken):
                        splitLine = line.split(':')
                        solutionIdx = splitLine[1].strip()
                    elif line.startswith(winnerToken):
                        printWinner = True
                    # simply ignore irrelative msg
                    else:
                        continue

        # the last one bench result (since we won't see newProbToken nor startingToken anymore,
        # so we must print the last bench result here)
        if len(singleValueLine) > 0:
            # show on screen
            print(f"{singleValueLine}"+f",{solutionIdx}"+f",{solutionName}")
            singleValueLine = ""
            # add to csv
            singleValuesList.extend([winner,solutionIdx,solutionName])
            dd_output = defaultdict(str, zip(csvKeys, singleValuesList))
            benchResultsList += [dd_output]

        return await process.wait()  # Wait for the child process to exit

    if sys.platform == "win32":
        loop = asyncio.ProactorEventLoop()  # For subprocess' pipes on Windows
        asyncio.set_event_loop(loop)
    else:
        loop = asyncio.new_event_loop()

    returncode = loop.run_until_complete(run_command(*cmd, timeout=timeout))
    success = returncode == 0

    loop.close()

    return csvKeys, benchResultsList, success
