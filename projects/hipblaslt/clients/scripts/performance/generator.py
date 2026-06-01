# Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

import logging

from dataclasses import dataclass, field
from subprocess import run, PIPE
from pathlib import Path as path
from typing import Dict, List, Mapping, Generator

top = path(__file__).resolve().parent

@dataclass
class Problem:
    benchType: str = None
    args: Dict[str, str] = field(default_factory=dict)  # storing all arguments

@dataclass
class ProblemSet:
    benchType: str
    name: str
    problems: List[Problem]

    def __iter__(self):
        idx = 0
        while idx < len(self.problems):
            yield self.problems[idx]
            idx += 1

    def get_problemset_name(self):
        return self.benchType + "_" + self.name

    def generate_problems(self):
        for p in self.problems:
            p.benchType = self.benchType
            yield p

def load_suite(suite):
    """Load performance suite from suites.py."""

    tdef = top / 'suites.py'
    logging.info(f'loading suites from {tdef}')
    code = compile(tdef.read_text(), str(tdef), 'exec')
    ns = {}
    exec(code, ns)
    return ns[suite]

@dataclass
class SuiteProblemGenerator:
    suite_names: List[str]
    suites: Mapping[str, Generator[ProblemSet, None,
                                   None]] = field(default_factory=dict)

    # default arch is gfx942 when no argument is set and query is failed
    target_arch_static: str = "gfx942"

    @staticmethod
    def detectISA():
        # TODO- currently this is for PTS only. If we'd like to make it generic,
        # we need to select correct exec from "rocm_agent_enumerator", "amdgpu-arch", "hipinfo"
        enumerator = "/opt/rocm/bin/rocm_agent_enumerator"
        process = run([enumerator], stdout=PIPE)
        for line in process.stdout.decode().split("\n"):
            archStr = line.strip()
            if "gfx" in archStr:
                print(f"Info: Auto Detected GPU with ISA: {archStr}")
                SuiteProblemGenerator.target_arch_static = archStr
                break
        if process.returncode:
            print(f"{enumerator} exited with code {process.returncode}, using default arch gfx942")

    def __post_init__(self):
        for name in self.suite_names:
            self.suites[name] = load_suite(name)

    def generate_problemSet(self):
        for g in self.suites.values():
            yield from g()

@dataclass
class ShellScriptProblemGenerator:
    sh_filename: str = None
    bench_exec: str = None
    benchCMDs: List[str] = field(default_factory=list)

    def __post_init__(self):
        # print("sh file: " + str(self.sh_filename))
        # print("bench_exec: " + str(self.bench_exec))
        # load file and readline
        try:
            with open(self.sh_filename, 'r') as sh_file:
                # Iterate over each line in the file
                for cmd in sh_file:
                    cmd = cmd.strip()
                    # skip invalid line
                    if cmd.startswith("#") or cmd.count("hipblaslt-bench") == 0:
                        continue
                    # --verfiy will output other values, not supported yet
                    if cmd.count("verfiy") > 0 or cmd.count("-v") > 0:
                        print(f'--verify or -v is not supported, skip bench.')
                        continue

                    cmd = cmd.replace("./hipblaslt-bench", str(self.bench_exec))
                    # print("parsed cmd: " + cmd)
                    self.benchCMDs.append(cmd)

        except FileNotFoundError:
            print(f'Error: The file {self.sh_filename} was not found.')

        except Exception as e:
            print(f'An error occurred: {e}')

    def iterate_cmd(self):
        for cmd in self.benchCMDs:
            yield cmd
