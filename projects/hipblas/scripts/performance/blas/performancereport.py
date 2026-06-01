#!/usr/bin/env python3

"""Copyright (C) 2018-2020 Advanced Micro Devices, Inc. All rights reserved.

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
"""
import argparse
from collections import OrderedDict
import os
import re
import sys
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import math
import numpy as np
import ast
import operator

from matplotlib.ticker import (AutoMinorLocator)

import hipblas_gentest as gt

import commandrunner as cr

# TODO: Should any of these ignored arguments be passed on?
IGNORE_YAML_KEYS = [
        'KL',
        'KU',
        'incd',
        'incb',
        'alphai',
        'betai',
        'norm_check',
        'unit_check',
        'timing',
        'algo',
        'solution_index',
        'flags',
        'workspace_size',
        'initialization',
        'category',
        'known_bug_platforms',
        'name',
        'c_noalias_d',
        'samples',
        'a_type',
        'b_type',
        'c_type',
        'd_type',
        'stride_x',
        'stride_y',
        'ldd',
        'stride_a',
        'stride_b',
        'stride_c',
        'stride_d',
        ]
REGULAR_YAML_KEYS = [
        'batch_count',
        'function',
        'compute_type',
        'incx',
        'incy',
        'alpha',
        'beta',
        'iters',
        #samples', TODO: Implement this functionality at a low level
        'transA',
        'transB',
        'side',
        'uplo',
        'diag',
        ]
SWEEP_YAML_KEYS = [
        'n',
        'm',
        'k',
        'lda',
        'ldb',
        'ldc',
        ]

# Allowed operators
ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,   # unary minus
}

def _eval_node(node, variables):
    if isinstance(node, ast.Constant):  # Python 3.8+
        if isinstance(node.value, (int, float)):
            return node.value
        else:
            raise ValueError("Only int/float constants allowed")
    elif isinstance(node, ast.Name):
        if node.id in variables:
            return variables[node.id]
        else:
            raise ValueError(f"Unknown variable: {node.id}")
    elif isinstance(node, ast.BinOp):
        op = ALLOWED_OPERATORS.get(type(node.op))
        if not op:
            raise ValueError(f"Operator {type(node.op)} not allowed")
        return op(_eval_node(node.left, variables), _eval_node(node.right, variables))
    elif isinstance(node, ast.UnaryOp):
        op = ALLOWED_OPERATORS.get(type(node.op))
        if not op:
            raise ValueError(f"Unary operator {type(node.op)} not allowed")
        return op(_eval_node(node.operand, variables))
    else:
        raise ValueError(f"Unsupported expression: {ast.dump(node)}")

def safe_eval(expr, variables=None):
    """
    Evaluate a math-only expression safely.
    variables: dict of allowed variable names -> values
    """
    if variables is None:
        variables = {}

    # Parse into AST
    tree = ast.parse(expr, mode='eval')
    return _eval_node(tree.body, variables)

# If an argument is not relevant to a function, then its value is set to '*'.
# We cannot pass a '*' to subsequent commands because it will, so that flag
# needs to be removed.
class StripStarsArgument(cr.ArgumentABC):
    def __init__(self, flag):
        cr.ArgumentABC.__init__(self)
        self.flag = flag

    def get_args(self):
        if self._value is None:
            return []
            #raise RuntimeError('No value set for {}'.format(self.flag))
        if self._value == '*': # If an asterisk is specified
            return [] # Just ignore the flag entirely
        return [self.flag, str(self._value)]

# TODO: handle this better
class IgnoreArgument(cr.ArgumentABC):
    def __init__(self, flag):
        cr.ArgumentABC.__init__(self)
        self.flag = flag

    def get_args(self):
        return []

class PerfArgumentSet(cr.ArgumentSetABC):
    def _define_consistent_arguments(self):
        self.consistent_args['n'             ] = StripStarsArgument('-n'             )
        self.consistent_args['m'             ] = StripStarsArgument('-m'             )
        self.consistent_args['k'             ] = StripStarsArgument('-k'             )
        self.consistent_args['batch_count'   ] = StripStarsArgument('--batch_count'  ) #
        self.consistent_args['function'      ] = StripStarsArgument('-f'             ) #
        self.consistent_args['compute_type'  ] = StripStarsArgument('-r'             ) # precision
        self.consistent_args['incx'          ] = StripStarsArgument('--incx'         )
        self.consistent_args['incy'          ] = StripStarsArgument('--incy'         )
        self.consistent_args['alpha'         ] = StripStarsArgument('--alpha'        )
        self.consistent_args['beta'          ] = StripStarsArgument('--beta'         )
        self.consistent_args['iters'         ] = StripStarsArgument('-i'             ) #
        self.consistent_args['lda'           ] = StripStarsArgument('--lda'          )
        self.consistent_args['ldb'           ] = StripStarsArgument('--ldb'          )
        self.consistent_args['ldc'           ] = StripStarsArgument('--ldc'          )
        self.consistent_args['transA'        ] = StripStarsArgument('--transposeA'   )
        self.consistent_args['transB'        ] = StripStarsArgument('--transposeB'   )
        #self.consistent_args['initialization'] = StripStarsArgument('-initialization') # Unused?
        self.consistent_args['side'          ] = StripStarsArgument('--side'         )
        self.consistent_args['uplo'          ] = StripStarsArgument('--uplo'         )
        self.consistent_args['diag'          ] = StripStarsArgument('--diag'         )
        self.consistent_args['device'        ] = cr.DefaultArgument('--device', 0    )

    def _define_variable_arguments(self):
        self.variable_args['output_file'] = cr.PipeToArgument()

    def __init__(self, **kwargs):
        cr.ArgumentSetABC.__init__(
                self, **kwargs
                )

    def get_full_command(self, run_configuration):
        exec_name = os.path.join(run_configuration.executable_directory, 'hipblas-bench')
        if not os.path.exists(exec_name):
            raise RuntimeError('Unable to find {}!'.format(exec_name))

        #self.set('nsample', run_configuration.num_runs)
        self.set('output_file', self.get_output_file(run_configuration))

        return [exec_name] + self.get_args()

    def collect_timing(self, run_configuration, data_type='gflops', library_prefix='hipblas'):
        output_filename = self.get_output_file(run_configuration)
        rv = {}
        print('Processing {}'.format(output_filename))
        if os.path.exists(output_filename):
            lines = open(output_filename, 'r').readlines()
            us_vals = []
            gf_vals = []
            bw_vals = []
            gf_string = library_prefix + "-Gflops"
            bw_string = library_prefix + "-GB/s"
            us_string = "us"
            for i in range(0, len(lines)):
                if re.search(r"\b" + re.escape(us_string) + r"\b", lines[i]) is not None:
                    us_line = lines[i].strip().split(",")
                    index = [idx for idx, s in enumerate(us_line) if us_string in s][0] #us_line.index()
                    us_vals.append(float(re.split(r',\s*(?![^()]*\))', lines[i+1])[index]))
                if gf_string in lines[i]:
                    gf_line = lines[i].split(",")
                    index = gf_line.index(gf_string)
                    gf_vals.append(float(re.split(r',\s*(?![^()]*\))', lines[i+1])[index]))
                if bw_string in lines[i]:
                    bw_line = lines[i].split(",")
                    index = bw_line.index(bw_string)
                    bw_vals.append(float(re.split(r',\s*(?![^()]*\))', lines[i+1])[index]))
            if len(us_vals) > 0 and data_type == 'time':
                rv['Time (microseconds)'] = us_vals
            if len(bw_vals) > 0 and data_type == 'bandwidth':
                rv['Bandwidth (GB/s)'] = bw_vals
            if len(gf_vals) > 0 and data_type == 'gflops':
                rv['GFLOP/s'] = gf_vals
        else:
            print('{} does not exist'.format(output_filename))
        return rv

    def collect_timing_compare(self, run_configuration, data_type='gflops', library_prefix='hipblas'):
        output_filename = self.get_output_file_compare(run_configuration)
        rv = {}
        print('Processing {}'.format(output_filename))
        if os.path.exists(output_filename):
            lines = open(output_filename, 'r').readlines()
            us_vals = []
            gf_vals = []
            bw_vals = []
            gf_string = library_prefix + "-Gflops"
            bw_string = library_prefix + "-GB/s"
            us_string = "us"
            for i in range(0, len(lines)):
                if re.search(r"\b" + re.escape(us_string) + r"\b", lines[i]) is not None:
                    us_line = lines[i].strip().split(",")
                    index = [idx for idx, s in enumerate(us_line) if us_string in s][0] #us_line.index()
                    us_vals.append(float(re.split(r',\s*(?![^()]*\))', lines[i+1])[index]))
                if gf_string in lines[i]:
                    gf_line = lines[i].split(",")
                    index = gf_line.index(gf_string)
                    gf_vals.append(float(re.split(r',\s*(?![^()]*\))', lines[i+1])[index]))
                if bw_string in lines[i]:
                    bw_line = lines[i].split(",")
                    index = bw_line.index(bw_string)
                    bw_vals.append(float(re.split(r',\s*(?![^()]*\))', lines[i+1])[index]))
            if len(us_vals) > 0 and data_type == 'time':
                rv['Time (microseconds)'] = us_vals
            if len(bw_vals) > 0 and data_type == 'bandwidth':
                rv['Bandwidth (GB/s)'] = bw_vals
            if len(gf_vals) > 0 and data_type == 'gflops':
                rv['GFLOP/s'] = gf_vals
        else:
            print('{} does not exist'.format(output_filename))
        return rv


class YamlData:

    def __init__(self, config_file):
        self.config_file = config_file
        self.test_cases = []
        self.execute_run()

    def reorder_data(self):
        old_data = self.test_cases
        new_data = []
        names = []
        for test in old_data:
            name = test['function']
            precision = test['compute_type']
            side = test['side']
            if (name,precision) not in names: # TODO: This will always be true because "side" is not in the tuple.
                type = [ x for x in old_data if x['function']==name and x['compute_type'] == precision and x['side'] == side ]
                new_data.append(type)
                names.append((name,precision, side))
        self.test_cases = new_data

    #Monkey Patch
    def write_test(self, test):
        self.test_cases.append(test)

    #Monkey Patch
    def process_doc(self, doc):
        """Process one document in the YAML file"""

        # Ignore empty documents
        if not doc or not doc.get('Tests'):
            return

        # Clear datatypes and params from previous documents
        gt.datatypes.clear()
        gt.param.clear()

        # Return dictionary of all known datatypes
        gt.datatypes.update(gt.get_datatypes(doc))

        # Arguments structure corresponding to C/C++ structure
        gt.param['Arguments'] = type('Arguments', (gt.ctypes.Structure,),
                                {'_fields_': gt.get_arguments(doc)})

        # Special names which get expanded as lists of arguments
        gt.param['dict_lists_to_expand'] = doc.get('Dictionary lists to expand') or ()

        # Lists which are not expanded
        gt.param['lists_to_not_expand'] = doc.get('Lists to not expand') or ()

        # Defaults
        defaults = doc.get('Defaults') or {}

        default_add_ons = {'m': -1, 'M': -1, 'n': -1, 'N': -1, 'k': -1, 'K': -1, 'lda': -1, 'ldb': -1, 'ldc': -1, 'LDA': -1, 'LDB': -1, 'LDC': -1, 'iters': 1, 'flops': '', 'mem': '', 'samples': 1, 'step_mult': 0}
        defaults.update(default_add_ons)

        # Known Bugs
        gt.param['known_bugs'] = doc.get('Known bugs') or []

        # Functions
        gt.param['Functions'] = doc.get('Functions') or {}

        # Instantiate all of the tests, starting with defaults
        for test in doc['Tests']:
            case = defaults.copy()
            case.update(test)
            gt.generate(case, gt.instantiate)

    def import_data(self):
        gt.args['includes'] = []
        gt.args['infile'] = self.config_file
        gt.write_test = self.write_test
        for doc in gt.get_yaml_docs():
            self.process_doc(doc)

    def execute_run(self):
        self.import_data()
        self.reorder_data()

class PerfYamlComparison(cr.Comparison):
    def __init__(self, test_yaml, data_type, **kwargs):
        cr.Comparison.__init__(self,
            description=self.get_function_prefix(test_yaml[0]['compute_type']) + test_yaml[0]['function'].split('_')[0] + ' Performance',
            **kwargs)

        for test in test_yaml:
            argument_set = PerfArgumentSet()
            all_inputs = {key:test[key] for key in test if not key in IGNORE_YAML_KEYS} # deep copy and cast to dict
            # regular keys have a direct mapping to the benchmark executable
            for key in REGULAR_YAML_KEYS:
                argument_set.set(key, all_inputs.pop(key))
            # step_size and step_mult are special, the determine how to sweep variables
            step_size = int(all_inputs.pop('step_size')) if 'step_size' in all_inputs else 10 #backwards compatiable default
            step_mult = (int(all_inputs.pop('step_mult')) == 1) if 'step_mult' in all_inputs else False
            mem = all_inputs.pop('mem')
            flops = all_inputs.pop('flops')
            self.mem = mem
            self.flops = flops

            if step_size == 1 and step_mult:
                raise ValueError('Cannot increment by multiplying by one.')
            sweep_lists = {}
            for key in SWEEP_YAML_KEYS:
                key_min = int(all_inputs.pop(key))
                key_max = int(all_inputs.pop(key.upper()))
                if user_args.surface_plot:
                    num_comparisons = 0
                    key_minimum = key_min
                    while key_minimum <= key_max:
                        num_comparisons = num_comparisons + 1
                        key_minimum = key_minimum * step_size if step_mult else key_minimum + step_size
                    key_values = [None] * num_comparisons * num_comparisons
                    for row in range (0, num_comparisons):
                        for col in range (0, num_comparisons):
                            if key_min == -1:
                                break
                            elif key == 'm' or key == 'lda':
                                key_values[row * num_comparisons + col] = key_min
                            elif key == 'n':
                                key_values[col * num_comparisons + row] = key_min
                        key_min = key_min * step_size if step_mult else key_min + step_size
                    sweep_lists[key] = key_values
                else:
                    key_values = []
                    while key_min <= key_max:
                        key_values.append(key_min)
                        if key_min == -1:
                            break
                        key_min = key_min * step_size if step_mult else key_min + step_size
                    sweep_lists[key] = key_values
            sweep_lengths = {key:len(sweep_lists[key]) for key in sweep_lists}
            max_sweep_length = max(sweep_lengths.values())

            for key in sweep_lists:
                if sweep_lists[key][0] != -1:
                    sweep_lists[key] += [sweep_lists[key][sweep_lengths[key]-1]] * (max_sweep_length - sweep_lengths[key])
                    sweep_lengths[key] = max_sweep_length
            for sweep_idx in range(max_sweep_length):
                sweep_argument_set = argument_set.get_deep_copy()
                for key in sweep_lists:
                    if sweep_lengths[key] == max_sweep_length:
                        sweep_argument_set.set(key, sweep_lists[key][sweep_idx])

                self.add(sweep_argument_set)
            if len(all_inputs) > 0:
                print('WARNING - The following values were unused: {}'.format(all_inputs))
        self.data_type = data_type

    def get_function_prefix(self, compute_type):
        if '32_r' in compute_type:
            return 's'
        elif '64_r' in compute_type:
            return 'd'
        elif '32_c' in compute_type:
            return 'c'
        elif '64_c' in compute_type:
            return 'z'
        elif 'bf16_r' in compute_type:
            return 'bf'
        elif 'f16_r' in compute_type:
            return 'h'
        else:
            print('Error - Cannot detect precision preFix: ' + compute_type)

    def write_docx_table(self, document):
        if len(self.argument_sets) > 0:
            argument_diff = cr.ArgumentSetDifference(self.argument_sets, ignore_keys=self._get_sweep_keys())
            differences = argument_diff.get_differences()
            is_a_comparison = len(differences) > 0
            document.add_paragraph(
                 ('For all runs, ``' if is_a_comparison else 'Command: ')
                + ' '.join(self.argument_sets[0].get_args(require_keys=argument_diff.get_similarities()))
                +("'' is held constant." if is_a_comparison else '')
                )
            # if is_a_comparison:
            #     header_row = ['label'] + differences
            #     num_columns = len(header_row)
            #     sorted_argument_sets = self.sort_argument_sets(isolate_keys=self._get_sweep_keys())
            #     num_rows = len(sorted_argument_sets) + 1
            #     table_style = 'Colorful Grid' if self.user_args.docx_template is None else None
            #     table = document.add_table(num_rows, num_columns, style=table_style)
            #     row_idx = 0
            #     for col_idx, data in enumerate(header_row):
            #         table.cell(row_idx, col_idx).text = data
            #     for argument_set_hash, argument_sets in sorted_argument_sets.items():
            #         if len(argument_sets) > 0:
            #             row_idx += 1
            #             argument_set = argument_sets[0]
            #             row = [argument_set_hash]
            #             for key in differences:
            #                 argument = argument_set.get(key)
            #                 row.append(argument.get_value() if argument.is_set() else 'DEFAULT')
            #             for col_idx, data in enumerate(row):
            #                 table.cell(row_idx, col_idx).text = str(data)

    def write_latex_table(self, latex_module):
        if len(self.argument_sets) > 0:
            argument_diff = cr.ArgumentSetDifference(self.argument_sets, ignore_keys=self._get_sweep_keys())
            differences = argument_diff.get_differences()
            is_a_comparison = len(differences) > 0
            latex_module.append(
                 ('For all runs, ``' if is_a_comparison else 'Command: ')
                + ' '.join(self.argument_sets[0].get_args(require_keys=argument_diff.get_similarities()))
                +("'' is held constant." if is_a_comparison else '')
                )
            # if is_a_comparison:
            #     with latex_module.create(cr.pylatex.Center()) as centered:
            #         tabu_format = 'r|' + ''.join(['c' for key in differences])
            #         with centered.create(cr.pylatex.Tabu(tabu_format)) as data_table:
            #             header_row = ['label'] + differences
            #             data_table.add_row(header_row, mapper=[cr.pylatex.utils.bold])
            #             data_table.add_hline()
            #             sorted_argument_sets = self.sort_argument_sets(isolate_keys=self._get_sweep_keys())
            #             for argument_set_hash, argument_sets in sorted_argument_sets.items():
            #                 if len(argument_sets) > 0:
            #                     argument_set = argument_sets[0]
            #                     row = [argument_set_hash]
            #                     results = argument_set.collect_timing(run_configuration)
            #                     for metric_label in results:
            #                         if not metric_label in y_list_by_metric:
            #                             y_list_by_metric[metric_label] = []
            #                         y_list_by_metric[metric_label].extend(results[metric_label])
            #                     # For each metric, add a set of bars in the bar chart.
            #                     for metric_label, y_list in y_list_by_metric.items():
            #                         y_scatter_by_group[group_label].extend(sorted(y_list))
            #                     for key in differences:
            #                         argument = argument_set.get(key)
            #                         row.append(argument.get_value() if argument.is_set() else 'DEFAULT')
            #                     data_table.add_row(row)

data_type_classes = {}
class TimeComparison(PerfYamlComparison):
    def __init__(self, **kwargs):
        PerfYamlComparison.__init__(self, data_type='time', **kwargs)
# data_type_classes['time'] = TimeComparison

class FlopsComparison(PerfYamlComparison):
    def __init__(self, **kwargs):
        PerfYamlComparison.__init__(self, data_type='gflops', **kwargs)

    def plot(self, run_configurations, figure, axes, cuda, compare):
        num_argument_sets = len(self.argument_sets)
        if num_argument_sets == 0:
            return

        sorted_argument_sets = self.sort_argument_sets(isolate_keys=[]) # No sort applied, but labels provided
        argument_diff = cr.ArgumentSetDifference(self.argument_sets, ignore_keys=self._get_sweep_keys())
        differences = argument_diff.get_differences()
        test = []
        test_x = []
        test_y = []
        xLabel = []
        for key in differences:
            xLabel.append(key)
        for argument_set_hash, argument_sets in sorted_argument_sets.items():
            argument_set = argument_sets[0]
            precision = argument_set.get("compute_type").get_value()
            function = argument_set.get("function").get_value()
            for key in differences:
                argument = argument_set.get(key)
                if user_args.surface_plot:
                    if key == 'm':
                        test_x.append(argument.get_value() if argument.is_set() else 'DEFAULT')
                    elif key == 'n':
                        test_y.append(argument.get_value() if argument.is_set() else 'DEFAULT')
                else:
                    test.append(argument.get_value() if argument.is_set() else 'DEFAULT')
                    break

        grouped_run_configurations = run_configurations.group_by_label()

        num_groups = len(grouped_run_configurations)
        metric_labels = [key for key in self.argument_sets[0].collect_timing(run_configurations[0])]
        num_metrics = len(metric_labels)

        if num_metrics == 0:
            return

        # loop over independent outputs
        y_scatter_by_group = OrderedDict()
        # for comparison runs
        y_scatter_by_group2 = OrderedDict()
        for group_label, run_configuration_group in grouped_run_configurations.items():
            # x_scatter_by_group[group_label] = []
            y_scatter_by_group[group_label] = []
            y_scatter_by_group2[group_label] = []
            # loop over argument sets that differ other than the swept variable(s)
            for subset_label, partial_argument_sets in sorted_argument_sets.items():
                if len(partial_argument_sets) != 1:
                    raise ValueError('Assumed that sorting argument sets with no keys has a single element per sort.')
                argument_set = partial_argument_sets[0]
                y_list_by_metric = OrderedDict() # One array of y values for each metric
                y_list_by_metric2 = OrderedDict() # For comparison runs
                # loop over number of coarse grain runs and concatenate results
                for run_configuration in run_configuration_group:
                    results = argument_set.collect_timing(run_configuration)
                    for metric_label in results:
                        if not metric_label in y_list_by_metric:
                            y_list_by_metric[metric_label] = []
                        y_list_by_metric[metric_label].extend(results[metric_label])
                    if compare:
                        results2 = argument_set.collect_timing_compare(run_configuration)
                        for metric_label in results2:
                            if not metric_label in y_list_by_metric2:
                                y_list_by_metric2[metric_label] = []
                            y_list_by_metric2[metric_label].extend(results2[metric_label])
                # For each metric, add a set of bars in the bar chart.
                for metric_label, y_list in y_list_by_metric.items():
                    y_scatter_by_group[group_label].extend(sorted(y_list))
                if compare:
                    for metric_label, y_list in y_list_by_metric2.items():
                        y_scatter_by_group2[group_label].extend(sorted(y_list))

        for group_label, run_configuration_group in grouped_run_configurations.items():
            for run_configuration in run_configuration_group:
                mhz_str = "Mhz"
                mem_clk_str = "mclk"
                sys_clk_str = "sclk"
                mhz_str_cuda = "MHz"
                mem_clk_str_cuda = "memory"
                sys_clk_str_cuda = "sm"
                if cuda:
                    mhz_str = mhz_str_cuda
                    mem_clk_str = mem_clk_str_cuda
                    sys_clk_str = sys_clk_str_cuda
                # Reference: MI-100 clocks by default
                # mclk = 1200.0
                # sclk = 1087.0
                mclk = run_configuration.load_specifications()['Card0']["Start " + mem_clk_str].split(mhz_str)[0]
                sclk = run_configuration.load_specifications()['Card0']["Start " + sys_clk_str].split(mhz_str)[0]

                # Reference: V-100 clock by default
                # sclk_cuda = 1530.0
                sclk_cuda = 0
                if compare:
                    sclk_cuda = run_configuration.load_specifications_compare()['Card0']["Start " + sys_clk_str_cuda].split(mhz_str_cuda)[0]
                elif cuda:
                    sclk_cuda = run_configuration.load_specifications()['Card0']["Start " + sys_clk_str_cuda].split(mhz_str_cuda)[0]
                theoMax = 0
                theoMax_cuda = 0
                precisionBits = int(re.search(r'\d+', precision).group())
                if function == 'gemm' and precisionBits == 32: #xdlops
                    theoMax = float(sclk)/1000.00 * 256 * 120 #scaling to appropriate precision
                    theoMax_cuda = float(sclk_cuda)/1000.00 * 128 * 80
                elif function == 'trsm' or function == 'gemm':  #TODO better logic to decide memory bound vs compute bound
                    theoMax = float(sclk)/1000.00 * 128 * 120  * 32.00 / precisionBits #scaling to appropriate precision
                    theoMax_cuda = float(sclk_cuda)/1000.00 * 128 * 80 * 32.00 / precisionBits
                elif self.flops and self.mem:
                    try:
                        # TODO: Add calculation for theoMax_cuda
                        m=100000
                        n=100000
                        k=100000
                        theoFlops = safe_eval(self.flops, {'m': m, 'n': n, 'k': k})
                        theoMem = safe_eval(self.mem, {'m': m, 'n': n, 'k': k})
                        theoMax = float(mclk) / float(theoMem) * float(theoFlops) * 32 / precisionBits / 4
                    except:
                        print("flops and mem equations produce errors")

                if theoMax:
                    theoMax = round(theoMax)
                    x_co = (test[0], test[len(test)-1])
                    y_co = (theoMax, theoMax)
                    if not cuda:
                        theo_amd, = axes.plot(x_co, y_co, color='#ED1C24', label = "Theoretical Peak Performance MI-100: "+str(theoMax)+" GFLOP/s")

                if compare or cuda:
                    theoMax_cuda = round(theoMax_cuda)
                    x_co_cuda = (test[0], test[len(test)-1])
                    y_co_cuda = (theoMax_cuda, theoMax_cuda)
                    theo_cuda, = axes.plot(x_co_cuda, y_co_cuda, color='#76B900', label = "Theoretical Peak Performance V-100: "+ str(theoMax_cuda)+" GFLOP/s")
        if user_args.surface_plot:
            #===============
            #  First subplot
            #===============
            # set up the axes for the first plot
            #ax = fig.add_subplot(1, 2, 1, projection='3d')

            # plot a 3D surface like in the example mplot3d/surface3d_demo
            # copied from rocblas' performancereport.py
            X = np.array(test_x)
            X = np.reshape(X,(int(math.sqrt(X.size)), int(math.sqrt(X.size))))
            Y = np.array(test_y)
            Y = np.reshape(Y,(int(math.sqrt(Y.size)), int(math.sqrt(Y.size))))
            Z = np.array(y_scatter_by_group[group_label])
            Z = np.reshape(Z,(int(math.sqrt(Z.size)), int(math.sqrt(Z.size))))
            axes.legend()
            figure.suptitle(super().get_function_prefix(precision) + function + 'Performance', fontsize=14, fontweight='bold')
            axes.set_xlabel('m == lda', fontsize='large', fontweight='bold', labelpad=9)
            axes.set_ylabel('n', fontsize='large', fontweight='bold', labelpad=9)
            axes.zaxis.set_rotate_label(False)
            axes.set_zlabel(metric_labels[0] if len(metric_labels) == 1 else 'Time (s)', fontsize='large', fontweight='bold', rotation = 0, labelpad=36)
            surf = axes.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                                linewidth=0, antialiased=False)
            figure.colorbar(surf, shrink=0.5, aspect=10)
            plt.savefig(os.path.join(self.user_args.documentation_directory,
                                                    super().get_function_prefix(precision) + function + ' Performance' + '_auto_plot.pdf'))
            plt.show()
        else:
            if not cuda:
                for group_label in y_scatter_by_group:
                    axes.scatter(
                            # x_bar_by_group[group_label],
                            test,
                            y_scatter_by_group[group_label],
                            # gap_scalar * width,
                            color='#ED1C24',
                            label = 'MI-100 Performance'
                            # label = group_label,
                            )
                    axes.plot(
                            # x_scatter_by_group[group_label],
                            test,
                            y_scatter_by_group[group_label],
                            # 'k*',
                            '-ok',
                            color='#ED1C24',
                            )
            else:
                for group_label in y_scatter_by_group:
                    axes.scatter(
                            # x_bar_by_group[group_label],
                            test,
                            y_scatter_by_group[group_label],
                            # gap_scalar * width,
                            color='#76B900',
                            label = 'V-100 Performance'
                            # label = group_label,
                            )
                    axes.plot(
                            # x_scatter_by_group[group_label],
                            test,
                            y_scatter_by_group[group_label],
                            # 'k*',
                            '-ok',
                            color='#76B900',
                            )

            # if compare - already plotted AMD above
            if compare:
                for group_label in y_scatter_by_group:
                    axes.scatter(
                            # x_bar_by_group[group_label],
                            test,
                            y_scatter_by_group2[group_label],
                            # gap_scalar * width,
                            color='#76B900',
                            label = "V-100 Performance"
                            # label = group_label,
                            )
                    axes.plot(
                            # x_scatter_by_group[group_label],
                            test,
                            y_scatter_by_group2[group_label],
                            # 'k*',
                            '-ok',
                            color='#76B900',
                            )

            axes.xaxis.set_minor_locator(AutoMinorLocator())
            axes.yaxis.set_minor_locator(AutoMinorLocator())

            axes.set_ylabel(metric_labels[0] if len(metric_labels) == 1 else 'Time (s)' )
            axes.set_xlabel('='.join(xLabel))

        return True

class EfficiencyComparison(PerfYamlComparison):
    def __init__(self, **kwargs):
        PerfYamlComparison.__init__(self, data_type='gflops', **kwargs)

    def plot(self, run_configurations, axes, cuda, compare):
        num_argument_sets = len(self.argument_sets)
        if num_argument_sets == 0:
            return

        sorted_argument_sets = self.sort_argument_sets(isolate_keys=[]) # No sort applied, but labels provided
        argument_diff = cr.ArgumentSetDifference(self.argument_sets, ignore_keys=self._get_sweep_keys())
        differences = argument_diff.get_differences()
        test = []
        xLabel = []
        for key in differences:
            xLabel.append(key)
        for argument_set_hash, argument_sets in sorted_argument_sets.items():
            argument_set = argument_sets[0]
            precision = argument_set.get("compute_type").get_value()
            function = argument_set.get("function").get_value()
            for key in differences:
                argument = argument_set.get(key)
                test.append(argument.get_value() if argument.is_set() else 'DEFAULT')
                break

        grouped_run_configurations = run_configurations.group_by_label()

        num_groups = len(grouped_run_configurations)
        metric_labels = [key for key in self.argument_sets[0].collect_timing(run_configurations[0])]
        num_metrics = len(metric_labels)
        if num_metrics == 0:
            return

        # loop over independent outputs
        y_scatter_by_group = OrderedDict()
        # For comparison runs
        y_scatter_by_group2 = OrderedDict()
        for group_label, run_configuration_group in grouped_run_configurations.items():
            # x_scatter_by_group[group_label] = []
            y_scatter_by_group[group_label] = []
            if compare:
                y_scatter_by_group2[group_label] = []
            # loop over argument sets that differ other than the swept variable(s)
            for subset_label, partial_argument_sets in sorted_argument_sets.items():
                if len(partial_argument_sets) != 1:
                    raise ValueError('Assumed that sorting argument sets with no keys has a single element per sort.')
                argument_set = partial_argument_sets[0]
                y_list_by_metric = OrderedDict() # One array of y values for each metric
                y_list_by_metric2 = OrderedDict() # For comparison runs
                # loop over number of coarse grain runs and concatenate results
                for run_configuration in run_configuration_group:
                    results = argument_set.collect_timing(run_configuration)
                    for metric_label in results:
                        if not metric_label in y_list_by_metric:
                            y_list_by_metric[metric_label] = []
                        y_list_by_metric[metric_label].extend(results[metric_label])
                    if compare:
                        results2 = argument_set.collect_timing_compare(run_configuration)
                        for metric_label in results2:
                            if not metric_label in y_list_by_metric2:
                                y_list_by_metric2[metric_label] = []
                            y_list_by_metric2[metric_label].extend(results2[metric_label])
                # For each metric, add a set of bars in the bar chart.
                for metric_label, y_list in y_list_by_metric.items():
                    y_scatter_by_group[group_label].extend(sorted(y_list))
                if compare:
                    for metric_label, y_list in y_list_by_metric2.items():
                        y_scatter_by_group2[group_label].extend(sorted(y_list))

        for group_label, run_configuration_group in grouped_run_configurations.items():
            for run_configuration in run_configuration_group:
                mhz_str = "Mhz"
                mem_clk_str = "mclk"
                sys_clk_str = "sclk"
                mhz_str_cuda = "MHz"
                mem_clk_str_cuda = "memory"
                sys_clk_str_cuda = "sm"
                if cuda:
                    mhz_str = mhz_str_cuda
                    mem_clk_str = mem_clk_str_cuda
                    sys_clk_str = sys_clk_str_cuda
                # Reference: MI-100 clocks by default
                # mclk = 1200.0
                # sclk = 1087.0
                mclk = run_configuration.load_specifications()['Card0']["Start " + mem_clk_str].split(mhz_str)[0]
                sclk = run_configuration.load_specifications()['Card0']["Start " + sys_clk_str].split(mhz_str)[0]

                # Reference: V-100 clock by default
                # sclk_cuda = 1530.0

                sclk_cuda = 0
                if compare:
                    sclk_cuda = run_configuration.load_specifications_compare()['Card0']["Start " + sys_clk_str_cuda].split(mhz_str_cuda)[0]
                elif cuda:
                    sclk_cuda = run_configuration.load_specifications()['Card0']["Start " + sys_clk_str_cuda].split(mhz_str_cuda)[0]
                theoMax = 0
                theoMax_cuda = 0
                precisionBits = int(re.search(r'\d+', precision).group())
                if function == 'gemm' and precisionBits == 32: #xdlops
                    theoMax = float(sclk)/1000.00 * 256 * 120 #scaling to appropriate precision
                    theoMax_cuda = float(sclk_cuda)/1000.00 * 128 * 80
                elif function == 'trsm' or function == 'gemm':  #TODO better logic to decide memory bound vs compute bound
                    theoMax = float(sclk)/1000.00 * 128 * 120  * 32.00 / precisionBits #scaling to appropriate precision
                    theoMax_cuda = float(sclk_cuda)/1000.00 * 128 * 80 * 32.00 / precisionBits
                elif self.flops and self.mem:
                    # TODO: cuda here
                    try:
                        m=100000
                        n=100000
                        k=100000
                        theoFlops = safe_eval(self.flops, {'m': m, 'n': n, 'k': k})
                        theoMem = safe_eval(self.mem, {'m': m, 'n': n, 'k': k})
                        theoMax = float(mclk) / float(theoMem) * float(theoFlops) * 32 / precisionBits / 4
                    except:
                        print("flops and mem equations produce errors")

                # Comparing efficiency
                amd_performance_eff = OrderedDict()
                cuda_performance_eff = OrderedDict()


                if not cuda:
                    amd_perf_list = [x / theoMax for x in y_scatter_by_group[group_label]]
                    axes.plot(test, amd_perf_list, color='#ED1C24', label = "MI-100")
                else:
                    cuda_perf_list = [x / theoMax_cuda for x in y_scatter_by_group[group_label]]
                    axes.plot(test, cuda_perf_list, color='#76B900', label = "V-100")

                # Already plotted AMD, use second list for CUDA results
                if compare:
                    cuda_perf_list = [x / theoMax_cuda for x in y_scatter_by_group2[group_label]]
                    axes.plot(test, cuda_perf_list, color='#76B900', label = "V-100")
                axes.grid(True, which='major')
                axes.grid(True, which='minor')
                axes.yaxis.set_minor_locator(AutoMinorLocator(2))
                axes.set_ylim([0, 1])

        axes.set_ylabel('Efficiency')
        axes.set_xlabel('='.join(xLabel))
        return True

class BandwidthComparison(PerfYamlComparison):
    def __init__(self, **kwargs):
        PerfYamlComparison.__init__(self, data_type='bandwidth', **kwargs)

    def plot(self, run_configurations, axes, cuda, compare):
        num_argument_sets = len(self.argument_sets)
        if num_argument_sets == 0:
            return

        sorted_argument_sets = self.sort_argument_sets(isolate_keys=[]) # No sort applied, but labels provided
        #print(sorted_argument_sets)
        argument_diff = cr.ArgumentSetDifference(self.argument_sets, ignore_keys=self._get_sweep_keys())
        differences = argument_diff.get_differences()
        test = []
        xLabel = []
        for key in differences:
            xLabel.append(key)
        for argument_set_hash, argument_sets in sorted_argument_sets.items():
            argument_set = argument_sets[0]
            precision = argument_set.get("compute_type").get_value()
            function = argument_set.get("function").get_value()
            for key in differences:
                argument = argument_set.get(key)
                test.append(argument.get_value() if argument.is_set() else 'DEFAULT')
                break

        grouped_run_configurations = run_configurations.group_by_label()

        num_groups = len(grouped_run_configurations)
        metric_labels = [key for key in self.argument_sets[0].collect_timing(run_configurations[0])]
        num_metrics = len(metric_labels)
        if num_metrics == 0:
            return

        # loop over independent outputs
        y_scatter_by_group = OrderedDict()
        # for comparison runs
        y_scatter_by_group2 = OrderedDict()
        for group_label, run_configuration_group in grouped_run_configurations.items():
            # x_scatter_by_group[group_label] = []
            print(group_label)
            y_scatter_by_group[group_label] = []
            y_scatter_by_group2[group_label] = []
            # loop over argument sets that differ other than the swept variable(s)
            for subset_label, partial_argument_sets in sorted_argument_sets.items():
                if len(partial_argument_sets) != 1:
                    raise ValueError('Assumed that sorting argument sets with no keys has a single element per sort.')
                argument_set = partial_argument_sets[0]
                y_list_by_metric = OrderedDict() # One array of y values for each metric
                y_list_by_metric2 = OrderedDict() # For comparison runs
                # loop over number of coarse grain runs and concatenate results
                for run_configuration in run_configuration_group:
                    results = argument_set.collect_timing(run_configuration)
                    for metric_label in results:
                        if not metric_label in y_list_by_metric:
                            y_list_by_metric[metric_label] = []
                        y_list_by_metric[metric_label].extend(results[metric_label])
                    if compare:
                        results2 = argument_set.collect_timing_compare(run_configuration)
                        for metric_label in results2:
                            if not metric_label in y_list_by_metric2:
                                y_list_by_metric2[metric_label] = []
                            y_list_by_metric2[metric_label].extend(results2[metric_label])
                # For each metric, add a set of bars in the bar chart.
                for metric_label, y_list in y_list_by_metric.items():
                    y_scatter_by_group[group_label].extend(sorted(y_list))
                if compare:
                    for metric_label, y_list in y_list_by_metric2.items():
                        y_scatter_by_group2[group_label].extend(sorted(y_list))

        for group_label, run_configuration_group in grouped_run_configurations.items():
            for run_configuration in run_configuration_group:
                mhz_str = "Mhz"
                mem_clk_str = "mclk"
                sys_clk_str = "sclk"
                mhz_str_cuda = "MHz"
                mem_clk_str_cuda = "memory"
                sys_clk_str_cuda = "sm"
                if cuda:
                    mhz_str = mhz_str_cuda
                    mem_clk_str = mem_clk_str_cuda
                    sys_clk_str = sys_clk_str_cuda
                # Reference: MI-100 theoretical memory bandwidth by default
                tmb_MI100 = 1200
                # Reference: radeon 7 theoretical memory bandwidth by default
                tmb_radeon7 = 1000
                # Reference: Volta V100 theoretical memory bandwidth by default
                tmb_V100 = 900

                # Reference: V-100 clock by default
                # sclk_cuda = 1530.0
                if compare:
                    sclk_cuda = run_configuration.load_specifications_compare()['Card0']["Start " + sys_clk_str_cuda].split(mhz_str_cuda)[0]
                elif cuda:
                    sclk_cuda = run_configuration.load_specifications()['Card0']["Start " + sys_clk_str_cuda].split(mhz_str_cuda)[0]
                theoMax = 0
                theoMax_cuda = 0
                precisionBits = int(re.search(r'\d+', precision).group())
                if function == 'gemm' and precisionBits == 32: #xdlops
                    theoMax = tmb_radeon7
                    theoMax_cuda = tmb_V100
                elif function == 'trsm' or function == 'gemm':  #TODO better logic to decide memory bound vs compute bound
                    theoMax = tmb_radeon7 #scaling to appropriate precision
                    theoMax_cuda = tmb_V100
                elif function == 'copy' and precisionBits == 32:
                    theoMax = tmb_radeon7
                    theoMax_cuda = tmb_V100
                elif function == 'swap' and precisionBits == 32:
                    theoMax = tmb_radeon7
                    theoMax_cuda = tmb_V100
                elif self.flops and self.mem:
                    try:
                        # TODO: Add calculation for theoMax_cuda
                        theoMax = tmb_radeon7
                        theoMax_cuda = tmb_V100
                    except:
                        print("flops and mem equations produce errors")
                if theoMax:
                    print(theoMax)
                    theoMax = round(theoMax)
                    x_co = (test[0], test[len(test)-1])
                    y_co = (theoMax, theoMax)
                    if not cuda:
                        theo_amd, = axes.plot(x_co, y_co, color='#ED1C24', label = "Theoretical Peak Performance MI-100: "+str(theoMax)+" GB/s")

                if compare or cuda:
                    theoMax_cuda = round(theoMax_cuda)
                    x_co_cuda = (test[0], test[len(test)-1])
                    y_co_cuda = (theoMax_cuda, theoMax_cuda)
                    theo_cuda, = axes.plot(x_co_cuda, y_co_cuda, color='#76B900', label = "Theoretical Peak Performance V-100: "+ str(theoMax_cuda)+" GB/s")

        if not cuda:
            for group_label in y_scatter_by_group:
                #print(y_scatter_by_group[group_label])
                axes.scatter(
                        # x_bar_by_group[group_label],
                        test,
                        y_scatter_by_group[group_label],
                        # gap_scalar * width,
                        color='#ED1C24',
                        label = 'MI-100 Performance'
                        # label = group_label,
                        )
                axes.plot(
                        # x_scatter_by_group[group_label],
                        test,
                        y_scatter_by_group[group_label],
                        # 'k*',
                        '-ok',
                        color='#ED1C24',
                        )
        else:
            for group_label in y_scatter_by_group:
                axes.scatter(
                        # x_bar_by_group[group_label],
                        test,
                        y_scatter_by_group[group_label],
                        # gap_scalar * width,
                        color='#76B900',
                        label = 'V-100 Performance'
                        # label = group_label,
                        )
                axes.plot(
                        # x_scatter_by_group[group_label],
                        test,
                        y_scatter_by_group[group_label],
                        # 'k*',
                        '-ok',
                        color='#76B900',
                        )

        # if compare - already plotted AMD above
        if compare:
            for group_label in y_scatter_by_group:
                axes.scatter(
                        # x_bar_by_group[group_label],
                        test,
                        y_scatter_by_group2[group_label],
                        # gap_scalar * width,
                        color='#76B900',
                        label = "V-100 Performance"
                        # label = group_label,
                        )
                axes.plot(
                        # x_scatter_by_group[group_label],
                        test,
                        y_scatter_by_group2[group_label],
                        # 'k*',
                        '-ok',
                        color='#76B900',
                        )

        axes.xaxis.set_minor_locator(AutoMinorLocator())
        axes.yaxis.set_minor_locator(AutoMinorLocator())

        axes.set_ylabel(metric_labels[0] if len(metric_labels) == 1 else 'Time (s)' )
        axes.set_xlabel('='.join(xLabel))
        return True

data_type_classes['gflops'] = FlopsComparison
data_type_classes['efficiency'] = EfficiencyComparison
data_type_classes['bandwidth'] = BandwidthComparison

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-N', '--num-runs', default=10, type=int,
                        help='Number of times to run each test.')
    parser.add_argument('--data-types', default=data_type_classes.keys(), nargs='+',
                        choices = data_type_classes.keys(),
                        help='Types of data to generate plots for.')
    parser.add_argument('-I', '--input-yaml', required=True,
                        help='hipBLAS input yaml config.')
    user_args = cr.parse_input_arguments(parser)
    command_runner = cr.CommandRunner(user_args)
    command_runner.setup_system()

    #load yaml then create fig for every test
    with open(user_args.input_yaml, 'r') as f:
        data = YamlData(f)
        f.close()

    comparisons = []

    #setup tests sorted by their respective figures
    for test_yaml in data.test_cases:
        for data_type in user_args.data_types:
            print(data_type)
            data_type_cls = data_type_classes[data_type]
            comparison = data_type_cls(test_yaml = test_yaml)
            comparisons.append(comparison)

    command_runner.add_comparisons(comparisons)
    command_runner.main()
