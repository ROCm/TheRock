// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>
#include <functional>
#include <tuple>
#include <exception>
#include <sstream>
#include <vector>
#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "gemm_streamk_profiler.hpp"
#include "gemm_streamk_common.hpp"

// The kernel header is included via the compile command line with -include flag
// It defines SelectedKernel struct and KERNEL_NAME
// DataTypeTraits are now defined in gemm_streamk_common.hpp

// Create argument parser
inline auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("m", "3840", "The value for m dimension.")
        .insert("n", "4096", "The value for n dimension.")
        .insert("k", "2048", "The value for k dimension.")
        .insert("stride_a", "0", "The stride value for tensor A.")
        .insert("stride_b", "0", "The stride value for tensor B.")
        .insert("stride_c", "0", "The stride value for tensor C.")
        .insert("split_k", "1", "The split value for k dimension.")
        .insert("verify",
                "0",
                "The type of validation. Set to 0 for no validation, 1 for validation on CPU, or 2 "
                "for validation on GPU.")
        .insert("log",
                "false",
                "Whether output kernel instance information or not. Possible values are true or "
                "false.")
        .insert("warmup", "50", "The number of iterations before benchmark the kernel.")
        .insert("repeat", "100", "The number of iterations to benchmark the kernel.")
        .insert("timer",
                "true",
                "Whether if the timer is gpu timer or not. Possible values are false or true. "
                "")
        .insert("init",
                "0",
                "The method of tensor initialization. Set to 0 for random, to 1 for linear, or 2 "
                "for constant(1).")
        .insert("flush_cache", "true", "To flush cache, possible values are true or false.")
        .insert("rotating_count", "1000", "number of iterations to rotate the cache. default is 5.")
        .insert("metric",
                "0",
                "Metric with which to measure kernel performance. Set to 0 for latency, 1 for "
                "tflops, or 2 for bandwidth.")
        .insert("csv_filename",
                "",
                "The filename of benchmark result. Default is empty (no CSV output).")
        .insert("structured_sparsity",
                "false",
                "Whether use sparsity kernel or not. Possible values are true or false.")
        .insert(
            "json_output",
            "false",
            "Whether to output results in JSON format only. Possible values are true or false.");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

void repeat_once_if_verify(Setting& setting)
{
    // The output buffer will be reset after each run, which means the gemm result will be
    // accumulated in the output buffer. So limit the repeat to 1 if verify is true.
    if(setting.verify_)
    {
        setting.n_repeat_ = 1;
        setting.n_warmup_ = 0;
    }
}

void benchmark_gemm_single(const ck_tile::ArgParser& arg_parser)
{
    // Use DataTypeTraits to get the actual type names from the generated header
    // The generated header defines ADataType, BDataType, AccDataType, CDataType
    std::string dtype_a   = ck_tile::DataTypeTraits<ADataType>::name;
    std::string dtype_b   = ck_tile::DataTypeTraits<BDataType>::name;
    std::string dtype_acc = ck_tile::DataTypeTraits<AccDataType>::name;
    std::string dtype_c   = ck_tile::DataTypeTraits<CDataType>::name;

    // Layout names from the layout types
    std::string layout_a = ALayout::name;
    std::string layout_b = BLayout::name;
    std::string layout_c = CLayout::name;

    // Create GemmProblem struct
    GemmProblem gemm_problem{arg_parser.get_int("split_k"),
                             arg_parser.get_int("m"),
                             arg_parser.get_int("n"),
                             arg_parser.get_int("k"),
                             arg_parser.get_int("stride_a"),
                             arg_parser.get_int("stride_b"),
                             arg_parser.get_int("stride_c"),
                             dtype_a,
                             dtype_b,
                             dtype_acc,
                             dtype_c,
                             layout_a,
                             layout_b,
                             layout_c,
                             arg_parser.get_bool("structured_sparsity")};

    // Create Setting struct
    Setting setting{arg_parser.get_int("warmup"),
                    arg_parser.get_int("repeat"),
                    arg_parser.get_bool("timer"),
                    arg_parser.get_int("verify"),
                    arg_parser.get_int("init"),
                    arg_parser.get_bool("log"),
                    arg_parser.get_str("csv_filename"),
                    arg_parser.get_bool("flush_cache"),
                    arg_parser.get_int("rotating_count"),
                    arg_parser.get_bool("json_output")};

    repeat_once_if_verify(setting);

    // Get the profiler instance
    auto& profiler = GemmProfiler::instance(setting);

    try
    {
        // Create a lambda that wraps the kernel launch
        auto kernel_func = [](const ck_tile::StreamKHostArgs& args,
                              const ck_tile::stream_config& stream) {
            return SelectedKernel::launch(args, stream);
        };

        // Benchmark the kernel
        profiler.benchmark(gemm_problem, kernel_func);

        // Select best instance based on metric
        profiler.select_best_instance(static_cast<Metric>(arg_parser.get_int("metric")));
    }
    catch(const std::exception& e)
    {
        std::cerr << "Benchmark failed: " << e.what() << std::endl;
    }
}

int main(int argc, char* argv[])
{
    try
    {
        auto [result, parser] = create_args(argc, argv);
        if(!result)
        {
            parser.print();
            return EXIT_FAILURE;
        }

        benchmark_gemm_single(parser);
        return EXIT_SUCCESS;
    }
    catch(const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
