// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/pk_int4.hpp"

//[TODO] This can be moved to commons
// DataTypeTraits for all supported types
template <typename T>
struct DataTypeTraits;

template <>
struct DataTypeTraits<float>
{
    static constexpr const char* name = "fp32";
};

template <>
struct DataTypeTraits<double>
{
    static constexpr const char* name = "fp64";
};

template <>
struct DataTypeTraits<ck_tile::half_t>
{
    static constexpr const char* name = "fp16";
};

template <>
struct DataTypeTraits<ck_tile::bf16_t>
{
    static constexpr const char* name = "bf16";
};

template <>
struct DataTypeTraits<ck_tile::fp8_t>
{
    static constexpr const char* name = "fp8";
};

template <>
struct DataTypeTraits<ck_tile::bf8_t>
{
    static constexpr const char* name = "bf8";
};

template <>
struct DataTypeTraits<ck_tile::int8_t>
{
    static constexpr const char* name = "int8";
};

template <>
struct DataTypeTraits<ck_tile::int32_t>
{
    static constexpr const char* name = "int32";
};

template <>
struct DataTypeTraits<ck_tile::pk_int4_t>
{
    static constexpr const char* name = "pk_int4_t";
};

// Helper function to determine if a layout is row-major
template <typename Layout>
constexpr auto is_row_major(Layout)
{
    return ck_tile::bool_constant<std::is_same_v<Layout, ck_tile::tensor_layout::gemm::RowMajor>>{};
}

// Structure to hold kernel traits for dispatcher
struct KernelTraits
{
    std::string pipeline;  // preshufflev2
    std::string scheduler; // intrawave, interwave, default
    std::string epilogue;  // cshuffle, default
    bool pad_m;
    bool pad_n;
    bool pad_k;
    bool persistent;

    // Constructor with defaults
    KernelTraits()
        : pipeline("preshufflev2"),
          scheduler("default"),
          epilogue("default"),
          pad_m(false),
          pad_n(false),
          pad_k(false),
          persistent(false)
    {
    }
};

// Helper to extract traits from kernel name
inline KernelTraits extract_traits_from_name(const std::string& kernel_name)
{
    KernelTraits traits;

    // Extract pipeline
    if(kernel_name.find("preshufflev2") != std::string::npos)
    {
        traits.pipeline = "preshufflev2";
    }

    // Extract scheduler
    if(kernel_name.find("interwave") != std::string::npos)
    {
        traits.scheduler = "interwave";
    }
    else if(kernel_name.find("intrawave") != std::string::npos)
    {
        traits.scheduler = "intrawave";
    }
    else
    {
        traits.scheduler = "default";
    }

    // Extract epilogue
    if(kernel_name.find("default") != std::string::npos &&
       kernel_name.find("default_") == std::string::npos)
    {
        traits.epilogue = "default";
    }
    else
    {
        traits.epilogue = "cshuffle";
    }

    // Padding flags would need to be extracted from the kernel configuration
    // For now, we'll leave them as false

    return traits;
}

template <typename T>
auto shuffle_b(const ck_tile::HostTensor<T>& t,
               ck_tile::index_t N_Warp_Tile,
               ck_tile::index_t K_Warp_Tile)
{
    assert(t.get_lengths().size() == 2);
    int n_      = t.get_lengths()[1];
    int k_      = t.get_lengths()[0];
    int divisor = N_Warp_Tile == 32 ? 2 : 4;
    ck_tile::HostTensor<T> t_view(
        {n_ / N_Warp_Tile, N_Warp_Tile, k_ / K_Warp_Tile, divisor, K_Warp_Tile / divisor});
    std::copy(t.begin(), t.end(), t_view.begin());
    return ck_tile::reference_permute(t_view, {0, 2, 3, 1, 4});
}

template <typename T>
auto shuffle_b_permuteN(const ck_tile::HostTensor<T>& t,
                        ck_tile::index_t N_Warp_Tile,
                        ck_tile::index_t K_Warp_Tile,
                        ck_tile::index_t N_Tile,
                        ck_tile::index_t N_Warp)
{
    assert(t.get_lengths().size() == 2);

    int n_      = t.get_lengths()[1];
    int k_      = t.get_lengths()[0];
    int divisor = N_Warp_Tile == 32 ? 2 : 4;
    int NRepeat = N_Tile / N_Warp_Tile / N_Warp;
    ck_tile::HostTensor<T> t_view({n_ / N_Tile,
                                   N_Warp,
                                   N_Warp_Tile,
                                   NRepeat,
                                   k_ / K_Warp_Tile,
                                   divisor,
                                   K_Warp_Tile / divisor});
    std::copy(t.begin(), t.end(), t_view.begin());
    return ck_tile::reference_permute(t_view, {0, 3, 1, 4, 5, 2, 6});
}
