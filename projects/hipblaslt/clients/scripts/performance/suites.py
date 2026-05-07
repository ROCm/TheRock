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

from copy import deepcopy
from generator import Problem, ProblemSet, SuiteProblemGenerator

amax_def_args = {'--type'  : 'H',
                 '--dtype' : 'S',
                 '--init'  : 'hpl',
                 }

lengths = {

    'amax_example': [
        (64, 64),
        (128, 128),
    ],

    'amax_set_1': [
        (16,    1024  ),
        (16,    8192  ),
        (16,    65536 ),
        (2048,  1024  ),
        (2048,  8192  ),
        (2048,  65536 ),
        (8192,  1024  ),
        (8192,  8192  ),
        (8192,  65536 ),
        (16,    16384 ),
        (32,    16384 ),
        (16,    2048  ),
        (32,    2048  ),
        (16,    4096  ),
        (32,    4096  ),
    ],
}

# Suite definitions
def api_overhead():
    """API overhead"""

    # set different value for hot iteration of getHeuristic and getAll
    problemlist = [Problem(args={"--cold_iters" : "10" , "--iters" : "1000" , "--get_all_iters" : "100"})]
    yield ProblemSet(benchType="api_overhead", name="benchset_1", problems=problemlist)

def amax_example():
    """AMAX example."""

    problemlist = []

    for length in lengths['amax_example']:
        args = deepcopy(amax_def_args)
        args.update({'--m': str(length[0]),
                     '--n': str(length[1])})
        problemlist.append(Problem(args=args))

    yield ProblemSet(benchType="amax", name="example", problems=problemlist)

def amax_set_1():
    """AMAX benchset 1."""

    problemlist = []

    for length in lengths['amax_set_1']:
        args = deepcopy(amax_def_args)
        args.update({'--m': str(length[0]),
                     '--n': str(length[1])})
        problemlist.append(Problem(args=args))

    yield ProblemSet(benchType="amax", name="benchset_1", problems=problemlist)

def matmul_examples():
    """gemm examples"""

    problemlist = [Problem(args={"--log_function_name" : "" , "--yaml" : "matmul_example_bench.yaml"})]
    yield ProblemSet(benchType="matmul", name="example", problems=problemlist)

def matmul_set_1():
    """gemm benchset 1"""

    problemlist = [Problem(args={"--log_function_name" : "" , "--yaml" : "matmul_probset1_bench.yaml"})]
    yield ProblemSet(benchType="matmul", name="benchset_1", problems=problemlist)

def matmul_set_2():
    """gemm benchset 2"""

    problemlist = [Problem(args={"--log_function_name" : "" , "--yaml" : "matmul_probset2_bench.yaml"})]
    yield ProblemSet(benchType="matmul", name="benchset_2", problems=problemlist)

def matmul_set_3():
    """gemm benchset 3"""

    problemlist = [Problem(args={"--log_function_name" : "" , "--yaml" : "matmul_probset3_bench.yaml"})]
    yield ProblemSet(benchType="matmul", name="benchset_3", problems=problemlist)

def matmul_set_4():
    """gemm benchset 4"""

    problemlist = [Problem(args={"--log_function_name" : "" , "--yaml" : "matmul_probset4_bench.yaml"})]
    yield ProblemSet(benchType="matmul", name="benchset_4", problems=problemlist)

def matmul_set_5():
    """gemm benchset 5"""

    problemlist = [Problem(args={"--log_function_name" : "" , "--yaml" : "matmul_probset5_bench.yaml"})]
    yield ProblemSet(benchType="matmul", name="benchset_5", problems=problemlist)

def matmul_set_6():
    """gemm benchset 6"""

    problemlist = [Problem(args={"--log_function_name" : "" , "--yaml" : "matmul_probset6_bench.yaml"})]
    yield ProblemSet(benchType="matmul", name="benchset_6", problems=problemlist)

def ci_perf_job():
    """run basic job for PR-CI"""

    # general suites for all arches
    yield from api_overhead()
    yield from matmul_set_1()

def all():
    """all routine benchmarks"""

    target_arch = SuiteProblemGenerator.target_arch_static
    # if --targetArch is not set, we have to query it with rocm_agent_enumerator
    if target_arch is None:
        print(f'Info: targetArch is not set, auto-detecting...')
        SuiteProblemGenerator.detectISA()
        target_arch = SuiteProblemGenerator.target_arch_static

    print(f'Info: bench with --suite all with --targetArch {target_arch}')

    # general suites for all arches
    yield from api_overhead()
    yield from amax_set_1()

    if "gfx942" in target_arch:
        # Put focused tests set for gfx942
        yield from matmul_set_1() # this problemset is an initial test example for gfx942
        yield from matmul_set_2()
        yield from matmul_set_3()
        yield from matmul_set_4()
        yield from matmul_set_5()
        yield from matmul_set_6()
    elif "gfx950" in target_arch:
        # Put focused tests set for gfx950
        yield from matmul_set_2()
    elif "gfx90a" in target_arch:
        # FIXME- please replace the examples with the real interested problems for gfx90a
        yield from matmul_set_3()
        yield from matmul_set_4()
        yield from matmul_set_5()
    elif "gfx110" in target_arch:
        # FIXME- please replace the examples with the real interested problems for gfx110?
        yield from matmul_set_3()
    elif "gfx120" in target_arch:
        # FIXME- please replace the examples with the real interested problems for gfx120?
        yield from matmul_set_3()

