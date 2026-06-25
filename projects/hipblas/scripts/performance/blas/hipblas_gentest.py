#!/usr/bin/python3
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

import re
import sys
import os
import argparse
import ctypes
from fnmatch import fnmatchcase
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
INCLUDE_RE = re.compile(r'include\s*:\s*(.*)')

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
Expand hipBLAS YAML test data file into binary Arguments records
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
    # Do not put constant defaults here -- use hipblas_common.yaml for that.
    # These are only for dynamic defaults
    # TODO: This should be ideally moved to YAML file, with eval'd expressions.

    if test['function'] in ('asum_strided_batched', 'nrm2_strided_batched',
                            'scal_strided_batched', 'swap_strided_batched',
                            'copy_strided_batched', 'dot_strided_batched',
                            'dotc_strided_batched', 'dot_strided_batched_ex',
                            'dotc_strided_batched_ex', 'rot_strided_batched',
                            'rot_strided_batched_ex',
                            'rotm_strided_batched', 'iamax_strided_batched',
                            'iamin_strided_batched', 'axpy_strided_batched',
                            'axpy_strided_batched_ex', 'nrm2_strided_batched_ex',
                            'scal_strided_batched_ex'):
        setkey_product(test, 'stride_x', ['N', 'incx', 'stride_scale'])
        setkey_product(test, 'stride_y', ['N', 'incy', 'stride_scale'])
        # we are using stride_c for param in rotm
        if all([x in test for x in ('stride_scale')]):
            test.setdefault('stride_c', int(test['stride_scale']) * 5)

    elif test['function'] in ('tpmv_strided_batched'):
        setkey_product(test, 'stride_x', ['M', 'incx', 'stride_scale'])
# Let's use M * M (> (M * (M+1)) / 2) as a 'stride' size for the packed format.
        setkey_product(test, 'stride_a', ['M', 'M', 'stride_scale'])

    elif test['function'] in ('trmv_strided_batched'):
        setkey_product(test, 'stride_x', ['M', 'incx', 'stride_scale'])
        setkey_product(test, 'stride_a', ['M', 'lda', 'stride_scale'])

    elif test['function'] in ('gemv_strided_batched', 'gbmv_strided_batched',
                              'ger_strided_batched', 'geru_strided_batched',
                              'gerc_strided_batched', 'trsv_strided_batched'):
        if test['function'] in ('ger_strided_batched', 'geru_strided_batched',
                                'gerc_strided_batched', 'trsv_strided_batched'
                                ) or test['transA'] in ('T', 'C'):
            setkey_product(test, 'stride_x', ['M', 'incx', 'stride_scale'])
            setkey_product(test, 'stride_y', ['N', 'incy', 'stride_scale'])
        else:
            setkey_product(test, 'stride_x', ['N', 'incx', 'stride_scale'])
            setkey_product(test, 'stride_y', ['M', 'incy', 'stride_scale'])
        if test['function'] in ('gbmv_strided_batched'):
            setkey_product(test, 'stride_a', ['lda', 'N', 'stride_scale'])

    elif test['function'] in ('hemv_strided_batched', 'hbmv_strided_batched'):
        if all([x in test for x in ('N', 'incx', 'incy', 'stride_scale')]):
            setkey_product(test, 'stride_x', ['N', 'incx', 'stride_scale'])
            setkey_product(test, 'stride_y', ['N', 'incy', 'stride_scale'])
            setkey_product(test, 'stride_a', ['N', 'lda', 'stride_scale'])

    elif test['function'] in ('hpmv_strided_batched'):
        if all([x in test for x in ('N', 'incx', 'incy', 'stride_scale')]):
            setkey_product(test, 'stride_x', ['N', 'incx', 'stride_scale'])
            setkey_product(test, 'stride_y', ['N', 'incy', 'stride_scale'])
            ldN = int((test['N'] * (test['N'] + 1) * test['stride_scale']) / 2)
            test.setdefault('stride_a', ldN)

    elif test['function'] in ('spr_strided_batched', 'spr2_strided_batched',
                              'hpr_strided_batched', 'hpr2_strided_batched',
                              'tpsv_strided_batched'):
        setkey_product(test, 'stride_x', ['N', 'incx', 'stride_scale'])
        setkey_product(test, 'stride_y', ['N', 'incy', 'stride_scale'])
        setkey_product(test, 'stride_a', ['N', 'N', 'stride_scale'])

    elif test['function'] in ('her_strided_batched', 'her2_strided_batched',
                              'syr2_strided_batched'):
        setkey_product(test, 'stride_x', ['N', 'incx', 'stride_scale'])
        setkey_product(test, 'stride_y', ['N', 'incy', 'stride_scale'])
        setkey_product(test, 'stride_a', ['N', 'lda', 'stride_scale'])

    # we are using stride_c for arg c and stride_d for arg s in rotg
    # these are are single values for each batch
    elif test['function'] in ('rotg_strided_batched'):
        if 'stride_scale' in test:
            test.setdefault('stride_a', int(test['stride_scale']))
            test.setdefault('stride_b', int(test['stride_scale']))
            test.setdefault('stride_c', int(test['stride_scale']))
            test.setdefault('stride_d', int(test['stride_scale']))

    # we are using stride_a for d1, stride_b for d2, and stride_c for param in
    # rotmg. These are are single values for each batch, except param which is
    # a 5 element array
    elif test['function'] in ('rotmg_strided_batched'):
        if 'stride_scale' in test:
            test.setdefault('stride_a', int(test['stride_scale']))
            test.setdefault('stride_b', int(test['stride_scale']))
            test.setdefault('stride_c', int(test['stride_scale']) * 5)
            test.setdefault('stride_x', int(test['stride_scale']))
            test.setdefault('stride_y', int(test['stride_scale']))

    elif test['function'] in ('dgmm_strided_batched'):
        setkey_product(test, 'stride_c', ['N', 'ldc', 'stride_scale'])
        setkey_product(test, 'stride_a', ['N', 'lda', 'stride_scale'])
        if test['side'].upper() == 'L':
            setkey_product(test, 'stride_x', ['M', 'incx', 'stride_scale'])
        else:
            setkey_product(test, 'stride_x', ['N', 'incx', 'stride_scale'])

    elif test['function'] in ('geam_strided_batched'):
        setkey_product(test, 'stride_c', ['N', 'ldc', 'stride_scale'])

        if test['transA'].upper() == 'N':
            setkey_product(test, 'stride_a', ['N', 'lda', 'stride_scale'])
        else:
            setkey_product(test, 'stride_a', ['M', 'lda', 'stride_scale'])

        if test['transB'].upper() == 'N':
            setkey_product(test, 'stride_b', ['N', 'ldb', 'stride_scale'])
        else:
            setkey_product(test, 'stride_b', ['M', 'ldb', 'stride_scale'])

    elif test['function'] in ('trmm_strided_batched'):
        setkey_product(test, 'stride_b', ['N', 'ldb', 'stride_scale'])

        if test['side'].upper() == 'L':
            setkey_product(test, 'stride_a', ['M', 'lda', 'stride_scale'])
        else:
            setkey_product(test, 'stride_a', ['N', 'lda', 'stride_scale'])

    elif test['function'] in ('trsm_strided_batched',
                              'trsm_strided_batched_ex'):
        setkey_product(test, 'stride_b', ['N', 'ldb', 'stride_scale'])

        if test['side'].upper() == 'L':
            setkey_product(test, 'stride_a', ['M', 'lda', 'stride_scale'])
        else:
            setkey_product(test, 'stride_a', ['N', 'lda', 'stride_scale'])

    elif test['function'] in ('tbmv_strided_batched'):
        if all([x in test for x in ('M', 'lda', 'stride_scale')]):
            ldM = int(test['M'] * test['lda'] * test['stride_scale'])
            test.setdefault('stride_a', ldM)
        if all([x in test for x in ('M', 'incx', 'stride_scale')]):
            ldx = int(test['M'] * abs(test['incx']) * test['stride_scale'])
            test.setdefault('stride_x', ldx)

    elif test['function'] in ('tbsv_strided_batched'):
        setkey_product(test, 'stride_a', ['N', 'lda', 'stride_scale'])
        setkey_product(test, 'stride_x', ['N', 'incx', 'stride_scale'])

    test.setdefault('stride_x', 0)
    test.setdefault('stride_y', 0)

    if test['transA'] == '*' or test['transB'] == '*':
        test.setdefault('lda', 0)
        test.setdefault('ldb', 0)
        test.setdefault('ldc', 0)
        test.setdefault('ldd', 0)
    else:
        test.setdefault('lda', test['M'] if test['transA'].upper() == 'N'
                        else test['K'] if test['K'] != 0 else 1)
        test.setdefault('ldb', test['K'] if test['K'] != 0 else 1
                        if test['transB'].upper() == 'N' else test['N'])
        test.setdefault('ldc', test['M'])
        test.setdefault('ldd', test['M'])
        if test['batch_count'] > 0:
            test.setdefault('stride_a', test['lda'] *
                            (test['K'] if test['transA'].upper() == 'N' else
                             test['M']))
            test.setdefault('stride_b', test['ldb'] *
                            (test['N'] if test['transB'].upper() == 'N' else
                             test['K']))
            test.setdefault('stride_c', test['ldc'] * test['N'])
            test.setdefault('stride_d', test['ldd'] * test['N'])
            return

    test.setdefault('stride_a', 0)
    test.setdefault('stride_b', 0)
    test.setdefault('stride_c', 0)
    test.setdefault('stride_d', 0)


def write_signature(out):
    """Write the signature used to verify binary file compatibility"""
    if 'signature_written' not in args:
        sig = 0
        byt = bytearray("hipBLAS", 'utf_8')
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
        byt.extend(bytes("HIPblas", 'utf_8'))
        byt.append(0)
        out.write(byt)
        args['signature_written'] = True


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
                    arg.append(ctype(*test[name]))
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

    # Any Arguments fields declared as enums (a_type, b_type, etc.)
    enum_args = [decl[0] for decl in param['Arguments']._fields_
                 if decl[1].__module__ == '__main__']
    try:
        setdefaults(test)

        # For enum arguments, replace name with value
        for typename in enum_args:
            if test[typename] in datatypes:
                test[typename] = datatypes[test[typename]]

        known_bug_platforms = set()

        # Match known bugs
        if test['category'] not in ('known_bug', 'disabled'):
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
                    elif test[key] != (datatypes.get(value, value)
                                       if key in enum_args else value):
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
            'category'] not in ('known_bug', 'disabled') else ''

        write_test(test)

    except KeyError as err:
        sys.exit("Undefined value " + str(err) + "\n" + str(test))


def generate(test, function):
    """Generate test combinations by iterating across lists recursively"""
    test = test.copy()

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
        elif argname in test and type(test[argname]) in (tuple, list, dict):
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
        elif (type(test[key]) in (tuple, list) and
              key not in param['lists_to_not_expand']):
            for test[key] in test[key]:
                generate(test, function)
            return

    # Replace typed function names with generic functions and types
    if 'hipblas_function' in test:
        func = test.pop('hipblas_function')
        if func in param['Functions']:
            test.update(param['Functions'][func])
        else:
            test['function'] = func.rpartition('hipblas_')[2]
        generate(test, function)
        return

    function(test)


if __name__ == '__main__':
    main()
