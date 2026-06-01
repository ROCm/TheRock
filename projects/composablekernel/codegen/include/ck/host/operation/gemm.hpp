// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

namespace ck {
namespace host {
namespace operation {

struct TileDesc
{
    int block_size               = 0;
    int m_per_block              = 0;
    int n_per_block              = 0;
    int k_per_block              = 0;
    int ak1                      = 0;
    int bk1                      = 0;
    int m_per_XDL                = 0;
    int n_per_XDL                = 0;
    int m_Xdl_per_wave           = 0;
    int n_Xdl_per_wave           = 0;
    int num_gemmk_prefetch_stage = 0;
};

struct TileDescGemmGemm
{
    int block_size               = 0;
    int gemm01_m_per_block       = 0;
    int gemm0_n_per_block        = 0;
    int gemm0_k_per_block        = 0;
    int gemm1_n_per_block        = 0;
    int gemm1_k_per_block        = 0;
    int ak1                      = 0;
    int bk1                      = 0;
    int b1k1                     = 0;
    int m_per_XDL                = 0;
    int n_per_XDL                = 0;
    int gemm0_m_Xdl_per_wave     = 0;
    int gemm0_n_Xdl_per_wave     = 0;
    int gemm1_n_Xdl_per_wave     = 0;
    int num_gemmk_prefetch_stage = 0;
};

struct BlockTransferDesc
{
    std::string thread_cluster_length        = "";
    std::string thread_cluster_arrange_order = "";
    std::string src_access_order             = "";
    int src_vec_dim                          = 0;
    int src_scalar_per_vector                = 0;
    int dst_scalar_per_vector_k1             = 0;
    int lds_add_extra_dim                    = 0;
};
struct CShuffleDesc
{
    int m_Xdl_per_wave_per_shuffle = 0;
    int n_Xdl_per_wave_per_shuffle = 0;
};
struct CBlockTransferDesc
{
    std::string cluster_lengths_m_block_m_wave_m_per_Xdl_n_block_n_wave_n_per_Xdl = "";
    int scalar_per_vector_n_wave_n_per_Xdl                                        = 0;
};

} // namespace operation
} // namespace host
} // namespace ck
