#!/usr/bin/env python3

import argparse
import subprocess
import os
from pathlib import Path
import yaml
import sys

script_dir = os.path.dirname( __file__ )
getspec_module_dir = os.path.join(script_dir, '..', 'blas')
sys.path.append(getspec_module_dir)
import getspecs

def L1_theo_max(GBps):
    L1_theo_max_dict = {}

    L1_theo_max_dict['saxpy'] = GBps * ( 1 /  6 )
    L1_theo_max_dict['daxpy'] = GBps * ( 1 / 12 )
    L1_theo_max_dict['caxpy'] = GBps * ( 1 /  3 )
    L1_theo_max_dict['zaxpy'] = GBps * ( 1 /  6 )

    L1_theo_max_dict['sdot'] = GBps * ( 1 / 4 )
    L1_theo_max_dict['ddot'] = GBps * ( 1 / 8 )
    L1_theo_max_dict['cdot'] = GBps * ( 1 / 2 )
    L1_theo_max_dict['zdot'] = GBps * ( 1 / 4 )

    L1_theo_max_dict['sscal'] = GBps * ( 1 /  8 )
    L1_theo_max_dict['dscal'] = GBps * ( 1 / 16 )
    L1_theo_max_dict['cscal'] = GBps * ( 3 /  8 )
    L1_theo_max_dict['zscal'] = GBps * ( 3 / 16 )

    return L1_theo_max_dict

def L2_theo_max(GBps):
    L2_theo_max_dict = {}

    # matrix vector multiplication
    L2_theo_max_dict['sgemv'] = GBps * ( 1 / 2 )
    L2_theo_max_dict['dgemv'] = GBps * ( 1 / 4 )
    L2_theo_max_dict['cgemv'] = GBps * ( 1     )
    L2_theo_max_dict['zgemv'] = GBps * ( 1 / 2 )

#   L2_theo_max_dict['sgbmv'] = GBps * ( 1 / 2 )
#   L2_theo_max_dict['dgbmv'] = GBps * ( 1 / 4 )
#   L2_theo_max_dict['cgbmv'] = GBps * ( 1     )
#   L2_theo_max_dict['zgbmv'] = GBps * ( 1 / 2 )

    L2_theo_max_dict['strmv'] = GBps * ( 1 / 2 )
    L2_theo_max_dict['dtrmv'] = GBps * ( 1 / 4 )
    L2_theo_max_dict['ctrmv'] = GBps * ( 1     )
    L2_theo_max_dict['ztrmv'] = GBps * ( 1 / 2 )

#   L2_theo_max_dict['stbmv'] = GBps * ( 1 / 2 )
#   L2_theo_max_dict['dtbmv'] = GBps * ( 1 / 4 )
#   L2_theo_max_dict['ctbmv'] = GBps * ( 1     )
#   L2_theo_max_dict['ztbmv'] = GBps * ( 1 / 2 )

    L2_theo_max_dict['stpmv'] = GBps * ( 1 / 2 )
    L2_theo_max_dict['dtpmv'] = GBps * ( 1 / 4 )
    L2_theo_max_dict['ctpmv'] = GBps * ( 1     )
    L2_theo_max_dict['ztpmv'] = GBps * ( 1 / 2 )

    # Symmetric matrix vector multiplication
    L2_theo_max_dict['ssymv'] = GBps * ( 1     )
    L2_theo_max_dict['dsymv'] = GBps * ( 1 / 2 )

#   L2_theo_max_dict['ssbmv'] = GBps * ( 1     )
#   L2_theo_max_dict['dsbmv'] = GBps * ( 1 / 2 )

    L2_theo_max_dict['sspmv'] = GBps * ( 1     )
    L2_theo_max_dict['dspmv'] = GBps * ( 1 / 2 )

    # Hermitian matrix vector multiplication
    L2_theo_max_dict['chemv'] = GBps * ( 2     )
    L2_theo_max_dict['zhemv'] = GBps * ( 1     )

    L2_theo_max_dict['chpmv'] = GBps * ( 2     )
    L2_theo_max_dict['zhpmv'] = GBps * ( 1     )

#   L2_theo_max_dict['chbmv'] = GBps * ( 2     )
#   L2_theo_max_dict['zhbmv'] = GBps * ( 1     )

    # rank 1 updates
    L2_theo_max_dict['sger'] = GBps * ( 1 / 4 )
    L2_theo_max_dict['dger'] = GBps * ( 1 / 8 )

    L2_theo_max_dict['ssyr'] = GBps * ( 1 / 4 )
    L2_theo_max_dict['dsyr'] = GBps * ( 1 / 8 )

    L2_theo_max_dict['sspr'] = GBps * ( 1 / 4 )
    L2_theo_max_dict['dspr'] = GBps * ( 1 / 8 )

    L2_theo_max_dict['ssyr2'] = GBps * ( 1 / 2 )
    L2_theo_max_dict['dsyr2'] = GBps * ( 1 / 4 )

    L2_theo_max_dict['sspr2'] = GBps * ( 1 / 2 )
    L2_theo_max_dict['dspr2'] = GBps * ( 1 / 4 )

    # Hermitian rank 1 updates
    L2_theo_max_dict['cher'] = GBps * ( 1 / 2 )
    L2_theo_max_dict['zher'] = GBps * ( 1 / 4 )

    L2_theo_max_dict['chpr'] = GBps * ( 1 / 2 )
    L2_theo_max_dict['zhpr'] = GBps * ( 1 / 4 )

    L2_theo_max_dict['cher2'] = GBps * ( 1     )
    L2_theo_max_dict['zher2'] = GBps * ( 1 / 2 )

    L2_theo_max_dict['chpr2'] = GBps * ( 1     )
    L2_theo_max_dict['zhpr2'] = GBps * ( 1 / 2 )

    # triangle solves
    L2_theo_max_dict['strsv'] = GBps * ( 1 / 2 )
    L2_theo_max_dict['dtrsv'] = GBps * ( 1 / 4 )
    L2_theo_max_dict['ctrsv'] = GBps * ( 1     )
    L2_theo_max_dict['ztrsv'] = GBps * ( 1 / 2 )

#   L2_theo_max_dict['stbsv'] = GBps * ( 1 / 2 )
#   L2_theo_max_dict['dtbsv'] = GBps * ( 1 / 4 )
#   L2_theo_max_dict['ctbsv'] = GBps * ( 1     )
#   L2_theo_max_dict['ztbsv'] = GBps * ( 1 / 2 )

    L2_theo_max_dict['stpsv'] = GBps * ( 1 / 2 )
    L2_theo_max_dict['dtpsv'] = GBps * ( 1 / 4 )
    L2_theo_max_dict['ctpsv'] = GBps * ( 1     )
    L2_theo_max_dict['ztpsv'] = GBps * ( 1 / 2 )

    return L2_theo_max_dict

def write_machine_spec_yaml(machine_spec_filename):
    machine_spec_dict = {}
    device_number = 1
    cuda = False
    machine_spec_dict['arch'] = getspecs.getgfx(device_number, cuda)

    if machine_spec_dict['arch'] == 'gfx906':
        GBps = 1000
    elif machine_spec_dict['arch'] == 'gfx908':
        GBps = 1100
    elif machine_spec_dict['arch'] == 'gfx90a':
        GBps = 1600
    else:
        print("do not know GBps memory bandwidth for ", machine_spec_dict['arch'])
        print("add GBps to", sys.argv[0])
        print("quitting ", sys.argv[0])
        quit()

    machine_spec_dict['GBps'] = GBps
    machine_spec_dict.update(L1_theo_max(GBps))
    machine_spec_dict.update(L2_theo_max(GBps))

    with open(machine_spec_filename, 'w') as outfile:
        yaml.dump(machine_spec_dict, outfile)


def run_command(bench_command, input_yaml_file, output_csv_file):

    with open(input_yaml_file, 'r') as file:
        problems = file.read()

    output = subprocess.check_output([bench_command,
                                      '--log_function_name',
                                      '--log_datatype',
                                      '--yaml', input_yaml_file])
    output = output.decode('utf-8')

    with open(output_csv_file, 'w') as f:
        f.write(output)

if __name__ =='__main__':

    parser = argparse.ArgumentParser(
            description='run hipblas-bench for multiple yaml files',
            epilog='Example usage: python3 benchmark.py ' +
                    '-l blas1 -t gfx906  -f scal -f axpy ' +
                    '-b ../../../build_tensile/release/clients/staging/hipblas-bench ')

    parser.add_argument('-l', help='level.',          dest='level',          default="blas1")
    parser.add_argument('-t', help='tag.',            dest='tag',            default="gfx906")
    parser.add_argument('-f', help='function names.', dest='function_names', required=True, action='append')
    parser.add_argument('-b', help='bench command.',  dest='bench_command',  default="../../../build/release/clients/staging/hipblas-bench")

    args = parser.parse_args()

    device_number = 1
    cuda = False
    arch = getspecs.getgfx(device_number, cuda)
    if arch not in args.tag:
        print("Warning: tag does not contain architecture, it is recommended that tag contain the architecture")
        print("         tag, architecture = ", args.tag, ", ", arch)

    output_dir = os.path.join(args.level,args.tag)

    if os.path.exists(output_dir):
        print("Warning: directory exists: ", output_dir)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    machine_spec_filename = os.path.join(args.level, args.tag, "machine_spec.yaml")

    write_machine_spec_yaml(machine_spec_filename)

    for function_name in args.function_names:

        input_filename = os.path.join(args.level, function_name+".yaml")
        output_filename = os.path.join(args.level, args.tag, function_name+".csv")

        run_command(args.bench_command, input_filename, output_filename)
