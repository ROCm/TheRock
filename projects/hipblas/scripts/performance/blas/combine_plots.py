#!/usr/bin/env python3
import argparse
import os
import re
import sys
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

parser = argparse.ArgumentParser(description='Used with completed datasets from performancereport.py to compare different functions\' performance.\n' +
                                             'Works with hipblas-bench and hipBLAS\' performancereport.py. Can also be used with rocblas-bench and rocBLAS\' performancereport.py with ' +
                                             'command line arguments.',
                                 epilog='Example usage: ./combine_plots.py -s N -d ./output_gemm/run00 -d ./output_symm/run00')
parser.add_argument('-s', '--size_arg', help='Value which to base the x axis "size" on.', dest='size_arg', default='N')
parser.add_argument('-d', '--dir', action='append', help='Each directory containing data which you would like to compare.', dest='dirs', required=True)
parser.add_argument('-t', '--titles', action='append', help='The subtitle name for the resulting plot for each directory, in the same order as the arguments passed in --dir.', dest='titles')
parser.add_argument('-f', '--savedir', type=str, help='Directory where resulting plots will be saved.', dest='savedir', default='combine_plots')
parser.add_argument('-x', '--executable', type=str, help='Name of the executable used to run perf tests.', dest='executable_name', default='hipblas-bench')
parser.add_argument('--search_str', type=str, help='A search string to find data line in data files.', dest='search_string', default='hipblas-Gflops')

args = parser.parse_args()

def get_all_files(directory_str):
    """
    returns a list of *.out files in the given directory.

    Parameters:
        directory_str (string): the directory which we are gathering files from
    Returns:
        ret_list (list[string]): a list of file paths in the directory which end in .out
    """
    ret_list = []

    # only reading .out files as they contain the input parameters we need (func name, precision),
    # along with the output parameters we need (gflops)
    for f in os.listdir(os.fsencode(directory_str)):
        filename = os.fsdecode(f)
        if filename.endswith(".out"):
            ret_list.append(os.path.join(directory_str, filename))

    return ret_list

def get_output_val_from_file(filename, output_param='hipblas-Gflops', gflops_str='hipblas-Gflops'):
    """
    parses through file and returns the val as given in the file.

    Parameters:
        filename (string): path of the file to parse.
        output_param (string): the output parameter which we are parsing the file for.
        gflops_str (string): string to parse file for in csv portion.
    Returns:
        value (string): the gflops as listed in the file, refer to example file (TODO).
    """
    if os.path.exists(filename):
        lines = open(filename, 'r').readlines()

        for i in range(0, len(lines)):
            if(output_param in lines[i]):
                arg_line = lines[i].split(",")
                data_line = re.split(r',\s*(?![^()]*\))', lines[i+1])
                idx = arg_line.index(gflops_str)
                return data_line[idx]

    return '-1'

def get_input_param_from_file(filename, input_param, executable_name = 'hipblas-bench'):
    """
    parses through file and returns the function name as given in the file (by the -f argument passed to xxx-bench).

    Parameters:
        filename (string): path of the file to parse.
        input_param (string): the input parameter which we are parsing the file for. For example, '-f' to parse function name.
        executable_name (string): executable name that we can use to parse the function name in the data file
    Returns:
        funcname (string): function name as given by the -f argument passed to xxx-bench.
    """
    if os.path.exists(filename):
        lines = open(filename, 'r').readlines()

    for line in lines:
        if executable_name in line:
            linesplit = line.split()
            return linesplit[linesplit.index(input_param)+1]

    raise RuntimeError('Cannot find input param ' + input_param + ' in file: ' + filename)

def get_data_from_directories(directories, size_param = 'N', executable_name = 'hipblas-bench', search_string = 'hipblas-Gflops'):
    """
    For each directory in directories, gathers function name, precisions, sizes, and gflops from within files in that directory and returns it.

    Parameters:
        directories (list[string]): list of directory names to gather information from
        size_param (string): parameter in file which defines the "size"
        executable_name (string): executable name that we can use to parse the function name in the data file
        search_string (string): the string which we used to find the data line in the .out file
    Returns:
        res_dicts (list[dict{string: list[(int, float)]}]): for each directory; for each precision in any file within that directory, this dictionary contains a list
                                                            of tuples for that precision containing sizes (as defined by size_param) and the corresponding
                                                            gflops value from a file.
        res_funcs (list[string]): a list of function names, one function name is gathered from each directory. Each function name corresponds to dictionary
                                  at the same index in res_dicts.
    """
    res_dicts = []
    res_funcs = []
    for directory in directories:
        cur_funcname = None
        cur_dict = {}

        for f in get_all_files(directory):
            # append funcname to list of funcnames, only for first file in each directory as we assume
            # each directory has data for only one function (but multiple precisions)
            if cur_funcname is None:
                cur_funcname = get_input_param_from_file(f, '-f')

            prec = get_input_param_from_file(f, '-r', executable_name)

            # a tuple of (size, gflops) as gathered from the current file
            size_perf_tuple = (int(get_output_val_from_file(f, search_string, size_param)), float(get_output_val_from_file(f, search_string)))
            if prec in cur_dict:
                cur_dict[prec].append(size_perf_tuple)
            else:
                cur_dict[prec] = [size_perf_tuple]

        res_dicts.append(cur_dict)
        res_funcs.append(cur_funcname)

    return res_dicts, res_funcs

def plot_data(gflops_dicts, titles, savedir, size_arg = 'N'):
    """
    plots gflops data from dictionaries, one plot for each common precision present in all dictionaries.

    Parameters:
        gflops_dicts (list[dict{string: list[(int, float)]}]): data as given by :func:`get_data_from_directories`.
        titles (list[string]): a list of titles for each data set to be plotted and used as a savefile name.
        savedir (string): directory where resulting plots will be saved.
        size_arg (string): x axis title on plot.
    """
    if len(gflops_dicts) == 0:
        return

    gflops_dict0 = gflops_dicts[0]
    for prec, _ in gflops_dict0.items():
        colors=iter(cm.rainbow(np.linspace(0,1,len(gflops_dicts))))
        figure, axes = plt.subplots(figsize=(7,7))
        for gflops_dict, funcname in zip(gflops_dicts, titles):
            cur_color = next(colors)
            if prec not in gflops_dict:
                continue
            gflops = gflops_dict[prec]
            gflops.append((0, 0)) # I prefer having a 0 at the bottom so the performance looks more accurate
            sorted_tuples = sorted(gflops)
            sorted_sizes = [x[0] for x in sorted_tuples]
            sorted_gflops = [x[1] for x in sorted_tuples]

            axes.scatter(sorted_sizes, sorted_gflops, color=cur_color, label=funcname)
            axes.plot(sorted_sizes, sorted_gflops, '-ok', color=cur_color)

        axes.set_xlabel('='.join(size_arg)) # in case we add multiple params
        axes.set_ylabel('gflops')

        # magic numbers from performancereport.py to make plots look nice
        axes.legend(fontsize=10, bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                    mode='expand', borderaxespad=0.)
        figure.tight_layout(rect=(0,0.05,1.0,1.0))

        filename = ''
        for func in titles:
            if filename != '':
                filename += '_'
            filename += func
        filename += '_' + prec
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        figure.savefig(os.path.join(os.getcwd(), savedir, filename))

gflops, funcnames = get_data_from_directories(args.dirs, args.size_arg, args.executable_name, args.search_string)

if args.titles:
    if len(args.titles) == len(gflops):
        funcnames = args.titles
    else:
        raise RuntimeError('Must have same amount of -t parameters as -d parameters, or have no -t parameters.')

plot_data(gflops, funcnames, args.savedir, args.size_arg)
