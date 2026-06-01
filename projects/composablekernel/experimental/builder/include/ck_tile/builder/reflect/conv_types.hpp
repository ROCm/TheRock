// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/// @file
/// @brief Type definitions for convolution reflection
///
/// This file contains the type definitions used by both conv_traits.hpp and conv_description.hpp
/// to avoid circular dependencies.

#pragma once

#include <array>

namespace ck_tile::reflect::conv {

/// @brief Data tile dimensions processed by a workgroup.
/// @details This struct defines the M, N, and K dimensions of the data tile
/// that a single workgroup (thread block) is responsible for processing in the
/// underlying GEMM computation.
struct DataTileInfo
{
    int m; ///< M dimension of the tile processed by the workgroup (MPerBlock).
    int n; ///< N dimension of the tile processed by the workgroup (NPerBlock).
    int k; ///< K dimension of the tile processed by the workgroup (KPerBlock).
};

/// @brief Dimensions for an input data tile transfer.
/// @details Defines the shape of the input tile (A or B matrix) as it is
/// transferred from global memory to LDS. The tile is conceptually divided
/// into k0 and k1 dimensions.
struct InputTileTransferDimensions
{
    int k0;     ///< The outer dimension of K, where K = k0 * k1.
    int m_or_n; ///< The M dimension for the A matrix transfer, or the N dimension for the B matrix.
    int k1; ///< The inner dimension of K, often corresponding to the vector load size from global
            ///< memory.
};

/// @brief Parameters governing the transfer of an input tile.
/// @details This struct holds configuration details for how an input tile is
/// loaded from global memory into LDS, including thread clustering, memory
/// access patterns, and vectorization settings.
struct InputTileTransferParams
{
    int k1; ///< The inner K dimension size, often matching the vectorization width.
    std::array<int, 3>
        thread_cluster_dims; ///< Spatial thread distribution over the input data tile; defines how
                             ///< many threads are arranged on each axis.
    std::array<int, 3> thread_cluster_order; ///< The order of thread spatial distribution over the
                                             ///< input tensor dimensions.
    std::array<int, 3> src_access_order; ///< The order of accessing input tensor axes (e.g., which
                                         ///< dimension to read first).
    int src_vector_dim; ///< The index of the axis on which vectorized memory access is performed
                        ///< (the contiguous dimension).
    int src_scalar_per_vector;    ///< The size of the vector access instruction; the number of
                                  ///< elements accessed per thread per instruction.
    int dst_scalar_per_vector_k1; ///< The size of the vectorized store into LDS memory along the K1
                                  ///< dimension.
    bool lds_padding; ///< Flag indicating if padding is used for the LDS tensor to prevent bank
                      ///< conflicts.
};

/// @brief Complete information for an input tile transfer.
/// @details Combines the dimensional information and transfer parameters for
/// a full description of an input tile's journey from global memory to LDS.
struct InputTileTransferInfo
{
    InputTileTransferDimensions tile_dimensions; ///< The shape and layout of the tile.
    InputTileTransferParams transfer_params; ///< The parameters for the memory transfer operation.
};

/// @brief Parameters for the warp-level GEMM computation.
/// @details Defines the configuration of the GEMM operation performed by each
/// warp using hardware MFMA (Matrix Fused Multiply-Add) instructions.
struct WarpGemmParams
{
    int gemm_m; ///< The M dimension of a single MFMA instruction (MPerXdl).
    int gemm_n; ///< The N dimension of a single MFMA instruction (NPerXdl).
    int m_iter; ///< The number of MFMA iterations along the M dimension of the output tile per
                ///< wavefront (MXdlPerWave).
    int n_iter; ///< The number of MFMA iterations along the N dimension of the output tile per
                ///< wavefront (NXdlPerWave).
};

/// @brief Parameters for shuffling data between warps (CShuffle optimization).
/// @details Configures how many MFMA instruction results are processed per
/// wave in each iteration of the CShuffle routine.
struct WarpShuffleParams
{
    int m_gemms_per_shuffle; ///< Number of MFMA results along the M dimension to process per wave
                             ///< per shuffle iteration.
    int n_gemms_per_shuffle; ///< Number of MFMA results along the N dimension to process per wave
                             ///< per shuffle iteration.
};

/// @brief Information for the output tile transfer (CShuffle).
/// @details Describes how the final computed tile (C matrix) is written out from
/// LDS to global memory, including shuffling, thread clustering, and vectorization.
struct OutputTileTransferInfo
{
    WarpShuffleParams shuffle_params; ///< Configuration for cross-warp data shuffling.
    // m_block, m_wave_per_xdl, n_block, n_wave_per_xdl
    std::array<int, 4> thread_cluster_dims; ///< The spatial thread distribution used for storing
                                            ///< data into the output tensor.
    int scalar_per_vector; ///< The size of the vectorized memory access when storing data to the
                           ///< output tensor.
};

} // namespace ck_tile::reflect::conv
