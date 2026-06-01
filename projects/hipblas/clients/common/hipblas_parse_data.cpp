/* ************************************************************************
 * Copyright (C) 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * ************************************************************************ */

#include "hipblas_parse_data.hpp"
#include "hipblas_data.hpp"
#include "utility.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <string>
#include <sys/types.h>

// Parse YAML data
static std::string hipblas_parse_yaml(const std::string& yaml)
{
#ifdef WIN32
    // Explicitly run via `python.exe`, without relying on the .py file being
    // treated as an executable that should be run via the python interpreter.
    std::string python_command_launcher = "python ";
#else
    // Rely on the shebang in the file, e.g. `#!/usr/bin/env python3`.
    std::string python_command_launcher = "";
#endif

    std::string tmp     = hipblas_tempname();
    auto        exepath = hipblas_exepath();
    auto        cmd = python_command_launcher + exepath + "hipblas_gentest.py --template " + exepath
               + "hipblas_template.yaml -o " + tmp + " " + yaml;
    std::cerr << cmd << std::endl;

#ifdef WIN32
    int status = std::system(cmd.c_str());
    if(status == -1)
        exit(EXIT_FAILURE);
#else
    int         status                  = system(cmd.c_str());
    if(status == -1 || !WIFEXITED(status) || WEXITSTATUS(status))
        exit(EXIT_FAILURE);
#endif

    return tmp;
}

// Parse --data and --yaml command-line arguments
bool hipblas_parse_data(int& argc, char** argv, const std::string& default_file)
{
    std::string filename;
    char**      argv_p = argv + 1;
    bool        help = false, yaml = false;

    // Scan, process and remove any --yaml or --data options
    for(int i = 1; argv[i]; ++i)
    {
        if(!strcmp(argv[i], "--data") || !strcmp(argv[i], "--yaml"))
        {
            if(!strcmp(argv[i], "--yaml"))
            {
                yaml = true;
            }

            if(filename != "")
            {
                std::cerr << "Only one of the --yaml and --data options may be specified"
                          << std::endl;
                exit(EXIT_FAILURE);
            }

            if(!argv[i + 1] || !argv[i + 1][0])
            {
                std::cerr << "The " << argv[i] << " option requires an argument" << std::endl;
                exit(EXIT_FAILURE);
            }
            filename = argv[++i];
        }
        else
        {
            *argv_p++ = argv[i];
            if(!help && (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")))
            {
                help = true;
                std::cout << "\n"
                          << argv[0] << " [ --data <path> | --yaml <path> ] <options> ...\n"
                          << std::endl;
            }
        }
    }

    // argc and argv contain remaining options and non-option arguments
    *argv_p = nullptr;
    argc    = argv_p - argv;

    if(filename == "-")
        filename = "/dev/stdin";
    else if(filename == "")
        filename = default_file;

    if(yaml)
        filename = hipblas_parse_yaml(filename);

    if(filename != "")
    {
        HipBLAS_TestData::set_filename(filename, yaml);
        return true;
    }

    return false;
}
