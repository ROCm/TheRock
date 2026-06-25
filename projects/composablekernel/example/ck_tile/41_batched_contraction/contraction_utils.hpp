// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"

struct AddDs
{
    template <typename E, typename C, typename... Ds>
    CK_TILE_HOST_DEVICE auto operator()(E& e, const C& c, const Ds&... ds) const -> void
    {
        const float x0_f =
            ck_tile::type_convert<float>(c) + (ck_tile::type_convert<float>(ds) + ...);

        e = ck_tile::type_convert<E>(x0_f);
    }
};

#define GEMM_PIPELINE ck_tile::GemmPipelineAgBgCrCompV3
#define UNIVERSAL_GEMM_PIPELINE ck_tile::BaseGemmPipelineAgBgCrCompV3
#define GEMM_PIPELINE_SCHEDULER ck_tile::GemmPipelineScheduler::Intrawave

template <typename DataType>
struct BatchedContractionTypeConfig
{
    using ADataType   = DataType;
    using BDataType   = DataType;
    using AccDataType = float;
    using EDataType   = DataType;
    using DDataType   = DataType;
};

using ContractionTypes = BatchedContractionTypeConfig<ck_tile::half_t>;

using ADataType   = ContractionTypes::ADataType;
using BDataType   = ContractionTypes::BDataType;
using AccDataType = ContractionTypes::AccDataType;
using EDataType   = ContractionTypes::EDataType;
using DDataType   = ContractionTypes::DDataType;

void print_help(const char* program_name)
{
    std::cout << "\n";
    std::cout << "Batched Tensor Contraction with element-wise fusion\n";
    std::cout << "E[G,M,N] = element_wise_op(contraction(A[G,M,K], B[G,N,K]), D0, D1, ...)\n";
    std::cout << "(Supports multiple D tensors with configurable element-wise operations)\n\n";

    std::cout << "Usage: " << program_name << " [OPTIONS]\n\n";

    std::cout << "Dimension Arguments (comma-separated, no spaces):\n";
    std::cout << "  -g_dims=<dims>   Batch dimensions      (default: \"1,2\")\n";
    std::cout << "  -m_dims=<dims>   M (row) dimensions    (default: \"4,256\")\n";
    std::cout << "  -n_dims=<dims>   N (column) dimensions (default: \"16,128\")\n";
    std::cout << "  -k_dims=<dims>   K (contract) dims     (default: \"64\")\n";
    std::cout << "  -num_d=<int>     Number of D tensors   (default: 2, range: 0-4)\n\n";

    std::cout << "Custom Stride Arguments (for testing non-contiguous tensors):\n";
    std::cout << "  -strides_a=<s>   A tensor strides (comma-separated, empty = auto)\n";
    std::cout << "  -strides_b=<s>   B tensor strides (comma-separated, empty = auto)\n";
    std::cout << "  -strides_e=<s>   E tensor strides (comma-separated, empty = auto)\n";
    std::cout << "  -strides_ds=<s>  D tensors strides (semicolon-separated, empty = same as E)\n";
    std::cout << "  Example: -strides_a=\"32768,128,1\" -strides_ds=\"512,2,1;1024,4,1\"\n\n";

    std::cout << "Layout Arguments:\n";
    std::cout
        << "  -a_layout=<R|C>  A tensor layout (R=Row-major, C=Column-major, default: \"R\")\n";
    std::cout << "  -b_layout=<R|C>  B tensor layout (default: \"C\")\n";
    std::cout << "  -e_layout=<R|C>  E tensor layout (default: \"R\")\n\n";

    std::cout << "Examples:\n";
    std::cout << "  Single batch (12 batches of 256×128):\n";
    std::cout << "    " << program_name
              << " -g_dims=\"12\" -m_dims=\"256\" -n_dims=\"128\" -k_dims=\"64\"\n\n";

    std::cout << "  2D batch grid (2×3=6 batches):\n";
    std::cout << "    " << program_name
              << " -g_dims=\"2,3\" -m_dims=\"128\" -n_dims=\"128\" -k_dims=\"64\"\n\n";

    std::cout << "  Multi-dimensional (flattened to M=128, N=128, K=128):\n";
    std::cout << "    " << program_name
              << " -g_dims=\"4\" -m_dims=\"8,16\" -n_dims=\"32,4\" -k_dims=\"16,8\"\n\n";

    std::cout << "Other Options:\n";
    std::cout << "  -v=<0|1>         Validation (0=off, 1=on, default: 1)\n";
    std::cout << "  -split_k=<int>   Split-K value (default: 1)\n";
    std::cout << "  -warmup=<int>    Warmup iterations (default: 5)\n";
    std::cout << "  -repeat=<int>    Benchmark iterations (default: 10)\n";
    std::cout << "  -log=<0|1>       Logging level (default: 1)\n";
    std::cout << "  -help            Show this help\n\n";
}

auto create_args(int argc, char* argv[])
{
    // Check for --help flag
    for(int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if(arg == "--help" || arg == "-h" || arg == "-help")
        {
            print_help(argv[0]);
            std::exit(0);
        }
    }

    ck_tile::ArgParser arg_parser;
    arg_parser.insert("m_dims", "4,256", "M dimensions separated by comma (e.g., '16,32' for 2D M)")
        .insert("n_dims", "16,128", "N dimensions separated by comma (e.g., '32,32' for 2D N)")
        .insert("k_dims", "64", "K dimensions separated by comma (e.g., '64,32' for 2D K)")
        .insert(
            "g_dims", "1,2", "G dimensions separated by comma (e.g., '4,2' for 2D, '2,3,4' for 3D)")
        .insert("num_d", "2", "Number of D (auxiliary input) tensors")
        .insert("strides_a", "", "A tensor strides (comma-separated, empty = auto/contiguous)")
        .insert("strides_b", "", "B tensor strides (comma-separated, empty = auto/contiguous)")
        .insert("strides_e", "", "E tensor strides (comma-separated, empty = auto/contiguous)")
        .insert("strides_ds",
                "",
                "D tensors strides (semicolon-separated for multiple, empty = same as E)")
        .insert("a_layout", "R", "A tensor data layout - Row by default")
        .insert("b_layout", "C", "B tensor data layout - Col by default")
        .insert("e_layout", "R", "E tensor data layout - Row by default")
        .insert("v", "1", "0. No validation, 1. Validation on CPU")
        .insert("prec", "fp16", "data type. fp32/fp16/bf16")
        .insert("warmup", "5", "number of iterations before benchmark the kernel")
        .insert("repeat", "10", "number of iterations to benchmark the kernel")
        .insert("timer", "gpu", "gpu:gpu timer, cpu:cpu timer")
        .insert("split_k", "1", "splitK value")
        .insert("log", "1", "log level for debugging");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

// Helper function to parse G, M, N, K dimensions from string
std::vector<ck_tile::index_t> parse_dimensions(const std::string& dims_str)
{
    std::vector<ck_tile::index_t> dims;
    std::stringstream ss(dims_str);
    std::string token;

    while(std::getline(ss, token, ','))
    {
        dims.push_back(std::stoi(token));
    }

    if(dims.empty())
    {
        throw std::invalid_argument("Dimensions cannot be empty");
    }

    return dims;
}

// Helper function to Calculate total elements from multi-dimensional vector
ck_tile::index_t calculate_total_elements(const std::vector<ck_tile::index_t>& dims)
{
    ck_tile::index_t total = 1;
    for(auto dim : dims)
    {
        total *= dim;
    }
    return total;
}

/**
 * @brief Flattens a list of tensor dimension components into a single dimension vector.
 *
 * This function takes a list of dimension vectors (e.g., representing different components
 * such as G, M, N, or K dimensions) and concatenates them into a single vector.
 *
 * Example:
 * Input: {{G0, G1}, {M0, M1}, {K0}}
 * Output: {G0, G1, M0, M1, K0}
 *
 * @param dim_components A vector of vectors, where each inner vector represents a set of tensor
 * dimensions.
 * @return A single vector containing all dimensions concatenated in order.
 */
std::vector<ck_tile::index_t>
concatenate_dim_components(const std::vector<std::vector<ck_tile::index_t>>& dim_components)
{
    std::vector<ck_tile::index_t> result;

    // Concatenate all dimension components into a single vector
    for(const auto& component : dim_components)
    {
        result.insert(result.end(), component.begin(), component.end());
    }

    return result;
}

// Helper function for printing dimensions
void print_dims(const std::string& name,
                const std::vector<ck_tile::index_t>& dims,
                ck_tile::index_t total)
{
    std::cout << name << ": [";
    for(size_t i = 0; i < dims.size(); ++i)
    {
        std::cout << dims[i];
        if(i < dims.size() - 1)
            std::cout << ",";
    }
    std::cout << "] ";
    if(total != 0)
        std::cout << "(total=" << total << ")";
    std::cout << std::endl;
}
