#!/usr/bin/python3
# ########################################################################
# Copyright (C) 2022-2023 Advanced Micro Devices, Inc.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ########################################################################

#Expand hipBLASLt YAML test data file into binary Arguments records"""

import re
import sys
import os
import argparse
import ctypes
from fnmatch import fnmatchcase
from typing import List, Tuple
try:  # Import either the C or pure-Python YAML parser
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader
import yaml

# Regex for type names in the YAML file. Optional *nnn indicates array.
TYPE_RE = re.compile(r'[a-z_A-Z]\w*(:?\s*\*\s*\d+)?$')

# Regex for integer ranges A..B[..C]
INT_RANGE_RE = re.compile(
    r'\s*(-?\d+)\s*\.\.\s*(-?\d+)\s*(?:\.\.\s*(-?\d+)\s*)?$')

# Regex for include: YAML extension
INCLUDE_RE = re.compile(r'include\s*:\s*([-.\w/]+)')

args = {}
testcases = set()
datatypes = {}
param = {}


def main():
    args.update(parse_args().__dict__)
    for doc in get_yaml_docs():
        process_doc(doc)


def process_doc(doc):
    """Process one document in the YAML file"""

    # Ignore empty documents
    if not doc or not doc.get('Tests'):
        return

    # Clear datatypes and params from previous documents
    datatypes.clear()
    param.clear()

    # Return dictionary of all known datatypes
    datatypes.update(get_datatypes(doc))

    # Arguments structure corresponding to C/C++ structure
    param['Arguments'] = type('Arguments', (ctypes.Structure,),
                              {'_fields_': get_arguments(doc)})

    # Special names which get expanded as lists of arguments
    param['dict_lists_to_expand'] = doc.get('Dictionary lists to expand') or ()

    # Lists which are not expanded
    param['lists_to_not_expand'] = doc.get('Lists to not expand') or ()

    # Defaults
    defaults = doc.get('Defaults') or {}

    # Known Bugs
    param['known_bugs'] = doc.get('Known bugs') or []

    # Functions
    param['Functions'] = doc.get('Functions') or {}

    # Instantiate all of the tests, starting with defaults
    for test in doc['Tests']:
        case = defaults.copy()
        case.update(test)
        generate(case, instantiate)


def parse_args():
    """Parse command-line arguments, returning input and output files"""
    parser = argparse.ArgumentParser(description="""
Expand hipBLASLt YAML test data file into binary Arguments records
""")
    parser.add_argument('infile',
                        nargs='?',
                        type=argparse.FileType('r'),
                        default=sys.stdin)
    parser.add_argument('-o', '--out',
                        dest='outfile',
                        type=argparse.FileType('wb'),
                        default=sys.stdout)
    parser.add_argument('-I',
                        help="Add include path",
                        action='append',
                        dest='includes',
                        default=[])
    parser.add_argument('-t', '--template',
                        type=argparse.FileType('r'))
    return parser.parse_args()


def read_yaml_file(file):
    """Read the YAML file, processing include: lines as an extension"""
    file_dir = os.path.dirname(file.name) or os.getcwd()
    source = []
    for line_no, line in enumerate(file, start=1):
        # Keep track of file names and line numbers for each line of YAML
        match = line.startswith('include') and INCLUDE_RE.match(line)
        if not match:
            source.append([line, file.name, line_no])
        else:
            include_file = match.group(1)
            include_dirs = [file_dir] + args['includes']
            for path in include_dirs:
                path = os.path.join(path, include_file)
                if os.path.exists(path):
                    source.extend(read_yaml_file(open(path, 'r')))
                    break
            else:
                sys.exit("In file " + file.name + ", line " +
                         str(line_no) + ", column " + str(match.start(1)+1) +
                         ":\n" + line.rstrip() + "\n" + " " * match.start(1) +
                         "^\nCannot open " + include_file +
                         "\n\nInclude paths:\n" + "\n".join(include_dirs))
    file.close()
    return source


def get_yaml_docs():
    """Parse the YAML file"""
    source = read_yaml_file(args['infile'])

    if args.get('template'):
        source = read_yaml_file(args['template']) + source

    source_str = ''.join([line[0] for line in source])

    def mark_str(mark):
        line = source[mark.line]
        return("In file " + line[1] + ", line " + str(line[2]) + ", column " +
               str(mark.column + 1) + ":\n" + line[0].rstrip() + "\n" +
               ' ' * mark.column + "^\n")

    # We iterate through all of the documents to properly diagnose errors,
    # because the load_all generator does not handle exceptions correctly.
    docs = []
    load = Loader(source_str)
    while load.check_data():
        try:
            doc = load.get_data()
        except yaml.YAMLError as err:
            sys.exit((mark_str(err.problem_mark) if err.problem_mark else "") +
                     (err.problem + "\n" if err.problem else "") +
                     (err.note + "\n" if err.note else ""))
        else:
            docs.append(doc)
    return docs


def get_datatypes(doc):
    """ Get datatypes from YAML doc"""
    dt = ctypes.__dict__.copy()
    for declaration in doc.get('Datatypes') or ():
        for name, decl in declaration.items():
            if isinstance(decl, dict):
                # Create derived class type based on bases and attr entries
                dt[name] = type(name,
                                tuple([eval(t, dt)
                                       for t in decl.get('bases') or ()
                                       if TYPE_RE.match(t)]
                                      ), decl.get('attr') or {})
                # Import class' attributes into the datatype namespace
                for subtype in decl.get('attr') or {}:
                    if TYPE_RE.match(subtype):
                        dt[subtype] = eval(name+'.'+subtype, dt)
            elif isinstance(decl, str) and TYPE_RE.match(decl):
                dt[name] = dt[decl]
            else:
                sys.exit("Unrecognized data type "+name+": "+repr(decl))
    return dt


def get_arguments(doc):
    """The kernel argument list, with argument names and types"""
    return [(var, eval(decl[var], datatypes))
            for decl in doc.get('Arguments') or ()
            if len(decl) == 1
            for var in decl
            if TYPE_RE.match(decl[var])]


def setkey_product(test, key, vals):
    """Helper for setdefaults. Tests that all values in vals is present
    in test, if so then sets test[key] to product of all test[vals]."""
    if all(x in test for x in vals):
        result = 1
        for x in vals:
            if x in ('incx', 'incy'):
                result *= abs(test[x])
            else:
                result *= test[x]
        test[key] = int(result)


def setdefaults(test):
    """Set default values for parameters"""
    # Do not put constant defaults here -- use hipblaslt_common.yaml for that.
    # These are only for dynamic defaults

    if test['transA'] == '*' or test['transB'] == '*':
        test.setdefault('lda', 0)
        test.setdefault('ldb', 0)
        test.setdefault('ldc', 0)
        test.setdefault('ldd', 0)
        test.setdefault('lde', 0)
    elif not test['grouped_gemm']: #catered to gemm default behavior
        M = test['M'][0] if isinstance(test['M'], list) else test['M']
        N = test['N'][0] if isinstance(test['N'], list) else test['N']
        K = test['K'][0] if isinstance(test['K'], list) else test['K']
        test.setdefault('lda', (M if M != 0 else 1) if test['transA'].upper() == 'N'
                        else K if K != 0 else 1)
        test.setdefault('ldb', (K if K != 0 else 1) if test['transB'].upper() == 'N'
                        else N if N != 0 else 1)
        test.setdefault('ldc', M if M != 0 else 1)
        test.setdefault('ldd', M if M != 0 else 1)
        test.setdefault('lde', M if M != 0 else 1)
        if test['batch_count'] > 0:
            test.setdefault('stride_a', test['lda'] *
                            (K if test['transA'].upper() == 'N' else
                             M))
            test.setdefault('stride_b', test['ldb'] *
                            (N if test['transB'].upper() == 'N' else
                             K))
            test.setdefault('stride_c', test['ldc'] * N)
            test.setdefault('stride_d', test['ldd'] * N)
            test.setdefault('stride_e', test['lde'] * N)
            return
    else:
        Ms = test['M']
        Ns = test['N']
        Ks = test['K']
        test.setdefault('lda', [(M if M != 0 else 1) if test['transA'].upper() == 'N'
                        else K if K != 0 else 1 for M, K in zip(Ms, Ks)])
        test.setdefault('ldb', [(K if K != 0 else 1) if test['transB'].upper() == 'N'
                        else N if N != 0 else 1 for K, N in zip(Ks, Ns)])
        test.setdefault('ldc', [M if M != 0 else 1 for M in Ms])
        test.setdefault('ldd', [M if M != 0 else 1 for M in Ms])
        test.setdefault('lde', [M if M != 0 else 1 for M in Ms])

        if test['batch_count'] > 0:
            test.setdefault('stride_a', [lda * K if test['transA'].upper() == 'N' else lda * M for K, M, lda in zip(Ks, Ms, test['lda'])])
            test.setdefault('stride_b',
                            [N * ldb if test['transB'].upper() == 'N' else
                             K * ldb for N, K, ldb in zip(Ns, Ks, test['ldb'])])
            test.setdefault('stride_c', [ldc * N for ldc, N in zip(test['ldc'], Ns)])
            test.setdefault('stride_d', [ldd * N for ldd, N in zip(test['ldd'], Ns)])
            test.setdefault('stride_e', [lde * N for lde, N in zip(test['lde'], Ns)])
            return


    test.setdefault('stride_a', 0)
    test.setdefault('stride_b', 0)
    test.setdefault('stride_c', 0)
    test.setdefault('stride_d', 0)
    test.setdefault('stride_e', 0)


def write_signature(out):
    """Write the signature used to verify binary file compatibility"""
    if 'signature_written' not in args:
        sig = 0
        byt = bytearray("hipBLASLt", 'utf_8')
        byt.append(0)
        last_ofs = 0
        for (name, ctype) in param['Arguments']._fields_:
            member = getattr(param['Arguments'], name)
            for i in range(0, member.offset - last_ofs):
                byt.append(0)
            for i in range(0, member.size):
                byt.append(sig ^ i)
            sig = (sig + 89) % 256
            last_ofs = member.offset + member.size
        for i in range(0, ctypes.sizeof(param['Arguments']) - last_ofs):
            byt.append(0)
        byt.extend(bytes("HIPblaslT", 'utf_8'))
        byt.append(0)
        out.write(byt)
        args['signature_written'] = True

def gen_shape_combinations(M: List, N: List, K: List) -> Tuple[List, List, List]:
    """Generate all shape combinations. This suitable for grouped gemm case."""
    from itertools import product

    if not isinstance(M, List):
        M = [M,]

    if not isinstance(N, List):
        N = [N,]

    if not isinstance(K, List):
        K = [K,]

    problem_sizes = list(product(M, N, K))
    return tuple(tuple(s[i] for s in problem_sizes) for i in range(len(problem_sizes[0])))

def write_test(test):
    """Write the test case out to the binary file if not seen already"""

    # For each argument declared in arguments, we generate a positional
    # argument in the Arguments constructor. For strings, we pass the
    # value of the string directly. For arrays, we unpack their contents
    # into the ctype array constructor and pass the ctype array. For
    # scalars, we coerce the string/numeric value into ctype.
    arg = []
    for name, ctype in param['Arguments']._fields_:
        try:
            if issubclass(ctype, ctypes.Array):
                if issubclass(ctype._type_, ctypes.c_char):
                    arg.append(bytes(test[name], 'utf_8'))
                else:
                    try:
                        arg.append(ctype(*test[name]))
                    except TypeError:
                        arg.append(ctype(test[name]))
            elif issubclass(ctype, ctypes.c_char):
                arg.append(bytes(test[name], 'utf_8'))
            else:
                arg.append(ctype(test[name]))
        except TypeError as err:
            sys.exit("TypeError: " + str(err) + " for " + name +
                     ", which has type " + str(type(test[name])) + "\n")
    byt = bytes(param['Arguments'](*arg))
    if byt not in testcases:
        testcases.add(byt)
        write_signature(args['outfile'])
        args['outfile'].write(byt)


def instantiate(test):
    """Instantiate a given test case"""
    test = test.copy()

    if test['grouped_gemm']:
        if 'matrix_size' in test:
            sizes = test.pop('matrix_size')
            for s in ('M', 'N', 'K'):
                test[s] = tuple(l[s] for l in sizes)
        else:
            M, N, K = gen_shape_combinations(test['M'], test['N'], test['K'])
            test['M'] = M
            test['N'] = N
            test['K'] = K

    gsu_vector = [-1 for _ in range(32)]
    if 'gsu_vector' in test:
      gsu_vector[0] = test['gsu_vector']
    else:
      gsu_vector[0] = 0
    test['gsu_vector'] = gsu_vector

    wgm_vector = [-1 for _ in range(32)]
    if 'wgm_vector' in test:
      wgm_vector[0] = test['wgm_vector']
    else:
      wgm_vector[0] = 0
    test['wgm_vector'] = wgm_vector


    # Any Arguments fields declared as enums (a_type, b_type, etc.)
    enum_args = [decl[0] for decl in param['Arguments']._fields_
                 if decl[1].__module__ == '__main__']
    array_value_args = ["gsu_vector", "wgm_vector"]

    try:
        setdefaults(test)

        # For enum arguments, replace name with value
        for typename in enum_args:
            if test[typename] in datatypes:
                test[typename] = datatypes[test[typename]]

        known_bug_platforms = set()

        # Match known bugs
        if test['category'] not in ('known_bug'):
            for bug in param['known_bugs']:
                for key, value in bug.items():
                    if key == 'known_bug_platforms' or key == 'category':
                        continue
                    if key not in test:
                        break
                    if key == 'function':
                        if not fnmatchcase(test[key], value):
                            break
                    # For keys declared as enums, compare resulting values
                    if key in enum_args:
                        if test[key] != datatypes.get(value, value):
                            break
                    # For array value type, treat as known bug if empty array was given
                    elif key in array_value_args:
                        stripped_val = [i for i in test[key] if i >= 0]

                        # default case in test
                        if stripped_val == [0,]:
                            break

                        if len(value) == 0:
                            continue
                        else:
                            if stripped_val != value:
                                break
                    elif test[key] != value:
                        break
                else:
                    # All values specified in known bug match the test case
                    platforms = bug.get('known_bug_platforms', '')

                    # If at least one known_bug_platforms is specified, add
                    # each platform in platforms to known_bug_platforms set
                    if platforms.strip(' :,\f\n\r\t\v'):
                        known_bug_platforms |= set(re.split('[ :,\f\n\r\t\v]+',
                                                   platforms))
                    else:
                        test['category'] = 'known_bug'
                    break

        # Unless category is already set to known_bug or disabled, set
        # known_bug_platforms to a space-separated list of platforms
        test['known_bug_platforms'] = ' ' . join(known_bug_platforms) if test[
            'category'] not in ('known_bug') else ''

        write_test(test)

    except KeyError as err:
        sys.exit("Undefined value " + str(err) + "\n" + str(test))


def generate(test, function):
    """Generate test combinations by iterating across lists recursively"""
    test = test.copy()

    def should_expand(key):
        grouped_gemm_cond = (test['grouped_gemm'] and key not in ('M', 'N', 'K'))
        normal_cond = type(test[key]) in (tuple, list) and\
                      key not in param['lists_to_not_expand'] and\
                      key != 'matrix_size'

        if test['grouped_gemm']:
            return grouped_gemm_cond and normal_cond
        return normal_cond

    def should_expand_dict_lists(argname):
        normal_cond = argname in test and type(test[argname]) in (tuple, list, dict)
        grouped_gemm_cond = argname != 'matrix_size'

        if test['grouped_gemm']:
            return normal_cond and grouped_gemm_cond
        return normal_cond

    # For specially named lists, they are expanded and merged into the test
    # argument list. When the list name is a dictionary of length 1, its pairs
    # indicate that the argument named by its key takes on values paired with
    # the argument named by its value, which is another dictionary list. We
    # process the value dictionaries' keys in alphabetic order, to ensure
    # deterministic test ordering.
    for argname in param['dict_lists_to_expand']:
        if type(argname) == dict:
            if len(argname) == 1:
                arg, target = list(argname.items())[0]
                if arg in test and type(test[arg]) == dict:
                    pairs = sorted(list(test[arg].items()), key=lambda x: x[0])
                    for test[arg], test[target] in pairs:
                        generate(test, function)
                    return
        elif should_expand_dict_lists(argname):#argname in test and type(test[argname]) in (tuple, list, dict):
            # Pop the list and iterate across it
            ilist = test.pop(argname)

            # For a bare dictionary, wrap it in a list and apply it once
            for item in [ilist] if type(ilist) == dict else ilist:
                try:
                    case = test.copy()
                    case.update(item)  # original test merged with each item
                    generate(case, function)
                except TypeError as err:
                    sys.exit("TypeError: " + str(err) + " for " + argname +
                            ", which has type " + str(type(item)) +
                            "\nA name listed in \"Dictionary lists to "
                            "expand\" must be a defined as a dictionary.\n")
            return

    for key in sorted(list(test)):
        # Integer arguments which are ranges (A..B[..C]) are expanded
        if type(test[key]) == str:
            match = INT_RANGE_RE.match(str(test[key]))
            if match:
                for test[key] in range(int(match.group(1)),
                                       int(match.group(2))+1,
                                       int(match.group(3) or 1)):
                    generate(test, function)
                return

        # For sequence arguments, they are expanded into scalars
        elif should_expand(key):
            for test[key] in test[key]:
                generate(test, function)
            return

    # Replace typed function names with generic functions and types
    if 'hipblaslt_function' in test:
        func = test.pop('hipblaslt_function')
        if func in param['Functions']:
            test.update(param['Functions'][func])
        else:
            test['function'] = func.rpartition('hipblaslt_')[2]
        generate(test, function)
        return

    function(test)


if __name__ == '__main__':
    main()
