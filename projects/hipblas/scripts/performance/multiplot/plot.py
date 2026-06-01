#!/usr/bin/env python3

import argparse
import os
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import yaml
from pathlib import Path

def plot_data(title, gflops_dicts1, gflops_dicts2, const_args_dicts, funcname_list, machine_spec_dict, savedir, theo_max, perf_vs_perf, size_arg = 'N'):
    """
    plots gflops data from dictionaries, one plot for each common precision present in all dictionaries.

    Parameters:
        title (string): title for plots
        gflops_dicts1 (list[dict{string: list[(int, float)]}]): data for one machine as given by :func:`get_data_from_directories`.
        gflops_dicts2 (list[dict{string: list[(int, float)]}]): data for another machine as given by :func:`get_data_from_directories`.
        const_args_dicts (list[dict{string: string}]): arguments that remain constant
        funcname_list (list[string]): a list of funcname for each data set to be plotted and used as a savefile name.
        machine_spec_dict (dict{string: string}): specification and peak performance for machine
        savedir (string): directory where resulting plots will be saved.
        theo_max (string): true for plotting performance versus theoretical maximum performance
        perf_vs_perf (string): true for plotting relative performance of one machine versus another machine
        size_arg (string): x axis title on plot.
    """
    if len(gflops_dicts1) == 0:
        return

    gflops_dict0 = gflops_dicts1[0]
    for prec, _ in gflops_dict0.items():
        colors=iter(cm.rainbow(np.linspace(0,1,len(gflops_dicts1))))
        figure, axes = plt.subplots(figsize=(7,7))

        for gflops_dict1, gflops_dict2, funcname, const_args_dict in zip(gflops_dicts1, gflops_dicts2, funcname_list, const_args_dicts):
            cur_color = next(colors)
            if prec not in gflops_dict1:
                continue
            gflops1 = gflops_dict1[prec]
            gflops2 = gflops_dict2[prec]
            if (perf_vs_perf != True):
                gflops1.append((0, 0)) # I prefer having a 0 at the bottom so the performance looks more accurate
                gflops2.append((0, 0)) # I prefer having a 0 at the bottom so the performance looks more accurate
            sorted_tuples1 = sorted(gflops1)
            sorted_tuples2 = sorted(gflops2)

            sorted_sizes1 = [x[0] for x in sorted_tuples1]
            sorted_sizes2 = [x[0] for x in sorted_tuples2]

            sorted_gflops1 = [x[1] for x in sorted_tuples1]
            sorted_gflops2 = [x[1] for x in sorted_tuples2]

            if sorted_sizes1 != sorted_sizes2:
                print("sizes are not the same for the two datasets")
                return

            if (perf_vs_perf == True):
                for i in range(len(sorted_gflops1)):
                    if sorted_gflops2[i] != 0:
                        sorted_gflops1[i] /= sorted_gflops2[i]

            if(prec == "f32_r"):
                function_label = "s" + funcname
            elif(prec == "f64_r"):
                function_label = "d" + funcname
            elif(prec == "f32_c"):
                function_label = "c" + funcname
            elif(prec == "f64_c"):
                function_label = "z" + funcname

            if(theo_max == True):
                theo_max_value = machine_spec_dict[function_label]
                sorted_gflops1[:] = [gf / theo_max_value for gf in sorted_gflops1]

            function_label = function_label + " :  " + const_args_dict[prec]

            axes.scatter(sorted_sizes1, sorted_gflops1, color=cur_color, label=function_label)
            axes.plot(sorted_sizes1, sorted_gflops1, '-o', color=cur_color)

        if(theo_max == True):
            axes.set_ylim(0, 1)
            axes.set_ylabel('gflops / theoretical_maximum_gflops')
        elif(perf_vs_perf == True):
            axes.set_ylabel('gflops / gflops')
        else:
            axes.set_ylabel('gflops')

        axes.set_xlabel('='.join(size_arg)) # in case we add multiple params

        # magic numbers from performancereport.py to make plots look nice
        axes.legend(fontsize=10, bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                    mode='expand', borderaxespad=0.)
        figure.tight_layout(rect=(0,0.05,1.0,0.94))

        figure.suptitle(title, y=0.96)

        filename = ''
        for funcname in funcname_list:
            if filename != '':
                filename += '_'
            filename += funcname
        filename += '_' + prec
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        figure.savefig(os.path.join(os.getcwd(), savedir, filename))

def get_function_name(filename):
    function_str = "function"
    if os.path.exists(filename):
        lines = open(filename, 'r').readlines()
    else:
        print(filename + " does not exist")
    for i in range(0, len(lines)):
        if(function_str in lines[i]):
            arg_line = lines[i].split(",")
            data_line = re.split(r',\s*(?![^()]*\))', lines[i+1])
            function_idx = arg_line.index(function_str)
            return data_line[function_idx]

def get_data_from_file(filename, output_param='hipblas-Gflops', xaxis_str1='N', xaxis_str2='M', yaxis_str='hipblas-Gflops'):

    precision_str = "compute_type"
    if os.path.exists(filename):
        lines = open(filename, 'r').readlines()

    function_name = get_function_name(filename)

    cur_dict = {}
    for i in range(0, len(lines)):
        if(output_param in lines[i]):
            arg_line = lines[i].split(",")
            data_line = re.split(r',\s*(?![^()]*\))', lines[i+1])
            if(xaxis_str1 in arg_line):
                xaxis_idx = arg_line.index(xaxis_str1)
            if(xaxis_str2 in arg_line):
                xaxis_idx = arg_line.index(xaxis_str2)
            yaxis_idx = arg_line.index(yaxis_str)
            size_perf_tuple = (int(data_line[xaxis_idx]), float(data_line[yaxis_idx]))

            precision_idx = arg_line.index(precision_str)
            precision = data_line[precision_idx]
            if precision in cur_dict:
                cur_dict[precision].append(size_perf_tuple)
            else:
                cur_dict[precision] = [size_perf_tuple]

    return cur_dict

tracked_param_list = [ 'transA', 'transB', 'uplo', 'diag', 'side', 'M', 'N', 'K', 'KL', 'KU', 'alpha', 'alphai', 'beta', 'betai',
                       'incx', 'incy', 'lda', 'ldb', 'ldd', 'stride_x', 'stride_y', 'stride_a', 'stride_b', 'stride_c', 'stride_d',
                       'batch_count']

# return string of arguments that remain constant. For example, transA, transB, alpha, beta, incx may remain
# constant. By contrast, M, N, K, lda, ldb, ldc may change
#def get_const_args_str(filename, output_param='hipblas-Gflops'):
def get_const_args_dict(filename, output_param='hipblas-Gflops'):

    if os.path.exists(filename):
        lines = open(filename, 'r').readlines()


    precision_str = "compute_type"
    precisions = []
    for i in range(0, len(lines)):
        if(output_param in lines[i]):
            arg_line = lines[i].split(",")
            data_line = re.split(r',\s*(?![^()]*\))', lines[i+1])

            precision_idx = arg_line.index(precision_str)
            precision = data_line[precision_idx]
            if precision not in precisions:
                precisions.append(precision)

    const_args_dict = {}

    for precision in precisions:

        function_param_list = tracked_param_list

        arg_line_index_dict = {}
        arg_line_value_dict = {}
        for i in range(0, len(lines)):
            if((output_param in lines[i]) and (precision in lines[i+1])):
                arg_line = lines[i].split(",")
                data_line = re.split(r',\s*(?![^()]*\))', lines[i+1])

                if not arg_line_index_dict:
                    for arg in arg_line :
                        if(arg in function_param_list):
                            index = arg_line.index(arg)
                            value = data_line[index]
                            arg_line_index_dict[arg]=index
                            arg_line_value_dict[arg]=value
                else:
                    for arg in arg_line :
                        if(arg in function_param_list):
                            index = arg_line.index(arg)
                            value = data_line[index]
                            previous_value = arg_line_value_dict[arg]
                            if(value != previous_value):
                                function_param_list.remove(arg)
                                del arg_line_value_dict[arg]

        const_args_str = ""
        for key, value in arg_line_value_dict.items():
            if(const_args_str == ""):
                const_args_str += key + "=" + value
            else:
                const_args_str += ", " + key + "=" + value

        const_args_dict[precision] = const_args_str
    return const_args_dict

if __name__ =='__main__':

    parser = argparse.ArgumentParser(
            description='plot hipblas-bench results for multiple csv files',
            epilog='Example usage: python3 plot_benchmarks.py ' +
                    '-l blas1 -t gfx906  -f scal -f axpy  --label1 "N" --label2 "M"')

    parser.add_argument('-l', '--level',          help='BLAS level',          dest='level',          default='blas1')
    parser.add_argument('-t',  '--tag1',          help='tag1',                dest='tag1',           default='gfx906')
    parser.add_argument(       '--tag2',          help='tag2',                dest='tag2',           default='ref')
    parser.add_argument(     '--label1',          help='label1',              dest='label1',         default='N')
    parser.add_argument(     '--label2',          help='label2',              dest='label2',         default='M')
    parser.add_argument('-f'           ,          help='function name',       dest='function_names', required=True, action='append')
    parser.add_argument(     '--theo_max',        help="perf vs theo_max",    dest='theo_max', default="false", action='store_true')
    parser.add_argument(     '--no_theo_max',     help="no perf vs theo_max", dest='theo_max', action='store_false')
    parser.add_argument(     '--perf_vs_perf',    help="perf vs perf",        dest='perf_vs_perf', default="false", action='store_true')
    parser.add_argument(     '--no_perf_vs_perf', help="no perf vs perf",     dest='perf_vs_perf', action='store_false')

    args = parser.parse_args()

    funcname_list = []

    gflops_dicts1 = []
    gflops_dicts2 = []
    const_args_dicts = []

    const_args_list = []

    if (args.theo_max == True):
        savedir = os.path.join(args.level, args.tag1, "plots_vs_theo_max")
        title = args.tag1 +  "(  performance / theoretical_maximum_performance )"
    elif (args.perf_vs_perf == True):
        savedir = os.path.join(args.level, args.tag1+"_"+args.tag2, "plots_perf_vs_perf")
        title = "Relative Performance ( " + args.tag1 + " /  " + args.tag2 + " )"
    else:
        savedir = os.path.join(args.level, args.tag1, "plots_gflops")
        title = "Performance " + args.tag1

    machine_spec_yaml_file = os.path.join(args.level, args.tag1, "machine_spec.yaml")

    machine_spec_dict = yaml.safe_load(Path(machine_spec_yaml_file).read_text())

    for function_name in args.function_names:

        output_filename1 = os.path.join(args.level, args.tag1, function_name+".csv")
        output_filename2   = os.path.join(args.level, args.tag2, function_name+".csv")

        gflops_dict1 = get_data_from_file(output_filename1, "hipblas-Gflops", args.label1, args.label2, "hipblas-Gflops")
        gflops_dict2 = get_data_from_file(output_filename2, "hipblas-Gflops", args.label1, args.label2, "hipblas-Gflops")

        gflops_dicts1.append(gflops_dict1)
        gflops_dicts2.append(gflops_dict2)

        const_args_dict = get_const_args_dict(output_filename1, "hipblas-Gflops")

        const_args_dicts.append(const_args_dict)

        function_name = get_function_name(output_filename1)
        funcname_list.append(function_name)

    print("plotting for: ", funcname_list)
    plot_data(title, gflops_dicts1, gflops_dicts2, const_args_dicts, funcname_list, machine_spec_dict, savedir, args.theo_max, args.perf_vs_perf, args.label1)
