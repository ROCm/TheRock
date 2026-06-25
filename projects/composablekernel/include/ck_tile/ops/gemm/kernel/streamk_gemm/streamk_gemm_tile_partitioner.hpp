// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"

namespace ck_tile {

/**
 * @brief Stream-K tile partitioner base class.
 *
 * This partitioner is responsible for mapping workgroups to tiles in the C tensor
 * for the Stream-K algorithm.
 *
 *  @tparam BlockGemmShapeType     A class providing basic GEMM parameters.
 *  @tparam ReductionStrategyType  An enum that defines the reduction strategy for the results in
 * the C Tensor.
 */
template <typename BlockGemmShapeType,
          StreamKReductionStrategy ReductionStrategyType = StreamKReductionStrategy::Atomic>
struct StreamKTilePartitionerBase
{

    static constexpr index_t MPerBlock                          = BlockGemmShapeType::kM;
    static constexpr index_t NPerBlock                          = BlockGemmShapeType::kN;
    static constexpr index_t KPerBlock                          = BlockGemmShapeType::kK;
    static constexpr StreamKReductionStrategy ReductionStrategy = ReductionStrategyType;
    static constexpr auto MemoryOperation = (ReductionStrategy == StreamKReductionStrategy::Atomic)
                                                ? memory_operation_enum::atomic_add
                                                : memory_operation_enum::set;

    StreamKTilePartitionerBase(index_t m, index_t n, index_t k, index_t grid);

    /**
     * @brief Calculates the total space needed for the partials buffer.
     *
     * @param acc_element_bytes  The number of bytes for the accumulator data type used in the GEMM.
     * @return index_t           The number of bytes needed for the partials buffer.
     */
    CK_TILE_HOST_DEVICE index_t get_partials_buffer_size(index_t acc_element_bytes) const noexcept;

    /**
     * @brief Calculates the total space needed for the flags buffer.
     *
     * @return index_t The number of bytes needed for the flags buffer.
     */
    CK_TILE_HOST_DEVICE index_t get_flags_buffer_size() const noexcept;

    public:
    /**
     * @brief Calculates the start iteration for the given the cta_idx.
     * @param cta_idx     The current Stream-K workgroup's index.
     * @return index_t    The start iteration.
     * @note It is assumed that the first Stream-K workgroup has a `cta_idx` of zero. If a
     * non-persistent DP section is used, then a Stream-K workgroup's `cta_idx` should be something
     * like `blockIdx.x` minus number of DP workgroups.
     */
    CK_TILE_DEVICE index_t get_start_iter(index_t cta_idx) const noexcept;

    /**
     * @brief Calculates the start and end iteration given the cta_idx.
     *
     * @param iter_start  Reference to an index_t; will be set to the starting iteration by the
     * function.
     * @param iter_end    Reference to an index_t; will be set to the non-inclusive end iteration by
     * the function.
     * @param cta_idx     The current Stream-K workgroup's index.
     * @note It is assumed that the first Stream-K workgroup has a `cta_idx` of zero. If a
     * non-persistent DP section is used, then a Stream-K workgroup's `cta_idx` should be something
     * like `blockIdx.x` minus number of DP workgroups.
     */
    CK_TILE_DEVICE void
    get_iter_boundaries(index_t& iter_start, index_t& iter_end, index_t cta_idx) const noexcept;

    /**
     * @brief Calculates the 1D tile index in the C tensor for a workgroup.
     *
     * @param iter_start  The starting iteration.
     * @return index_t    The 1D tile index.
     */
    CK_TILE_DEVICE index_t get_tile_index(index_t iter_start) const noexcept;

    /**
     * @brief Calculates the starting and ending tile boundaries for the given 1D tile index.
     *
     * @param tile_iter_start  Reference to an index_t; will be set to the tile's start iteration by
     * the function.
     * @param tile_iter_end    Reference to an index_t; will be set to the non-inclusive tile's end
     * iteration by the function.
     * @param tile_idx       The 1D C tensor tile index for the workgroup.
     */
    CK_TILE_DEVICE void get_tile_boundaries(index_t& tile_iter_start,
                                            index_t& tile_iter_end,
                                            index_t tile_idx) const noexcept;

    /**
     * @brief Calculates the workgroup's starting iteration that is local to a tile.
     *
     * @param iter_start       The starting iteration.
     * @param tile_iter_start  The starting iteration of the tile (i.e., the tile's starting
     * boundary).
     * @return index_t  The local starting iteration. The value is in range [0, `iters_per_tile_`).
     * @note  Assumes `iter_start` >= `tile_iter_start`.
     */
    CK_TILE_DEVICE static index_t get_local_iter(index_t iter_start,
                                                 index_t tile_iter_start) noexcept;

    /**
     * @brief Calculates the workgroup's non-inclusive end iteration that is local to a tile.
     *
     * @param tile_iter_start  The starting tile iteration.
     * @param iter_end         The non-inclusive end iteration.
     * @param tile_iter_end    The non-inclusive end iteration of the tile.
     * @return index_t         The local non-inclusive end iteration.
     * @note  Assumes `iter_end` >= `tile_iter_start` and `tile_iter_end` >= `tile_iter_start`.
     */
    CK_TILE_DEVICE static index_t
    get_local_iter_end(index_t tile_iter_start, index_t iter_end, index_t tile_iter_end) noexcept;

    /**
     * @brief Calculates the workgroup's local CTA idx within the given tile.
     *
     * @param tile_iter_start  The starting tile iteration.
     * @param cta_idx          The Stream-K workgroup index.
     * @return index_t         The tile local workgroup index in the tile.
     */
    CK_TILE_DEVICE index_t get_tile_local_cta_index(index_t tile_iter_start,
                                                    index_t cta_idx) const noexcept;

    /**
     * @brief Calculates the workgroup's 2D tile index in the C tensor given the 1D tile index.
     *
     * @param tile_idx  The 1D tile index in the C tensor for the workgroup.
     * @return index_t  The corresponding 2D tile index in the C tensor for the workgroup.
     */
    CK_TILE_DEVICE auto
    get_output_tile_index(index_t tile_idx) const noexcept -> tuple<index_t, index_t>;

    /**
     * @brief Calculates the total space needed for the partials and flags buffers.
     *
     * @param acc_element_bytes  The number of bytes for the accumulator data type used in the GEMM.
     * @return index_t           The number of bytes needed for the partials and flags buffers.
     */
    CK_TILE_HOST_DEVICE index_t get_workspace_size(index_t acc_element_bytes) const noexcept;

    /**
     * @brief Returns the number of macro tiles in the C tensor.
     */
    CK_TILE_HOST_DEVICE index_t get_num_tiles() const noexcept;

    /**
     * @brief Returns the maximum number of active workgroups; this is assumed to be number of CUs *
     * occupancy.
     */
    CK_TILE_HOST_DEVICE index_t get_grid() const noexcept;

    /**
     * @brief Returns the number of tiles in the C tensor that will use the data-parallel (DP)
     * approach.
     */
    CK_TILE_HOST_DEVICE index_t get_dp_tiles() const noexcept;

    /**
     * @brief Returns the number of tiles in the C tensor that will use the Stream-K approach.
     */
    CK_TILE_HOST_DEVICE index_t get_sk_tiles() const noexcept;

    /**
     * @brief Returns the number of workgroups that will participate in Stream-K in the `sk_tiles_`.
     */
    CK_TILE_HOST_DEVICE index_t get_sk_ctas() const noexcept;

    /**
     * @brief Returns the total number of Stream-K iterations.
     */
    CK_TILE_HOST_DEVICE index_t get_total_sk_iters() const noexcept;

    /**
     * @brief Returns the total number of iterations per tile in the C tensor. In other words, this
     * is the total number of macro tiles along the K dimension of A and B.
     */
    CK_TILE_HOST_DEVICE index_t get_iters_per_tile() const noexcept;

    /**
     * @brief Returns the total number of Stream-K iterations for each `sk_cta`. This is the lower
     * bound (i.e., all `sk_ctas_` are guaranteed to perform at least this many iterations).
     */
    CK_TILE_HOST_DEVICE index_t get_iters_per_sk_cta() const noexcept;

    /**
     * @brief Returns the remainder resulting from `total_sk_iters_` divided by `sk_ctas_`. When
     * this is non-zero, the first `extra_iters_` `sk_ctas_` will get one additional iteration
     * assigned to them; such work groups will perform (`iters_per_sk_cta_` + 1) iterations.
     */
    CK_TILE_HOST_DEVICE index_t get_extra_iters() const noexcept;

    /**
     * @brief Returns the total number of DP iterations.
     */
    CK_TILE_HOST_DEVICE index_t get_total_dp_iters() const noexcept;

    /**
     * @brief Returns the n dimension for the GEMM problem.
     */
    CK_TILE_HOST_DEVICE index_t get_n() const noexcept;

    /**
     * @brief Returns an estimate of the number of workgroups writing to the same macro tile in C.
     */
    CK_TILE_HOST index_t estimate_num_wgs_per_tile() const noexcept;

    protected:
    index_t num_tiles_;
    index_t grid_;
    index_t dp_tiles_;

    private:
    /**
     * @brief The number of full tiles assigned to each `sk_cta` when performing DP + 2 Tile
     * Stream-K.
     */
    index_t full_tiles_ = 1;
    index_t sk_tiles_;
    index_t sk_ctas_;
    index_t total_sk_iters_;
    index_t iters_per_tile_;
    index_t iters_per_sk_cta_;
    index_t extra_iters_;
    index_t total_dp_iters_;
    index_t n_;
};

/**
 * @brief Template for the Stream-K tile partitioner derived struct.
 *
 * This partitioner is responsible for mapping workgroups to tiles in the C tensor
 * for the Stream-K algorithm. This struct is derived from
 * StreamKTilePartitionerBase<BlockGemmShapeType, ReductionStrategyType>. Behavior of the
 * StreamKTilePartitioner based on persistency will be in the template specializations.
 *
 *  @tparam BlockGemmShapeType     A class providing basic GEMM parameters.
 *  @tparam ReductionStrategyType  An enum that defines the reduction strategy for the results in
 * the C Tensor.
 *  @tparam Persistent          A bool that indicates whether to use a Persistent approach
 */
template <typename BlockGemmShapeType,
          StreamKReductionStrategy ReductionStrategyType,
          bool Persistent>
struct StreamKTilePartitioner;

/**
 * @brief Persistent Stream-K tile partitioner derived struct.
 *
 * This partitioner is responsible for mapping workgroups to tiles in the C tensor
 * for the Stream-K algorithm when using a Persistent approach where no extra workgroups
 * are allocated for data parallel.
 *
 *  @tparam BlockGemmShapeType      A class providing basic GEMM parameters.
 *  @tparam ReductionStrategyType   An enum that defines the reduction strategy for the results in
 * the C Tensor.
 */
template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategyType>
struct StreamKTilePartitioner<BlockGemmShapeType, ReductionStrategyType, true>
    : StreamKTilePartitionerBase<BlockGemmShapeType, ReductionStrategyType>
{
    StreamKTilePartitioner(ck_tile::index_t m,
                           ck_tile::index_t n,
                           ck_tile::index_t k,
                           ck_tile::index_t grid);

    public:
    static constexpr bool PERSISTENT = true;
    /**
     * @brief Calculates the launching grid size for the Stream-K kernel. In the Persistent
     * case, no extra workgroups are allocated for the data parallel section, making the grid
     * size num_cu * occupancy.
     *
     * @return dim_3           The launching grid size for the kernel.
     */
    CK_TILE_HOST auto grid_size() const noexcept -> dim3;

    /**
     * @brief Returns the total number of DP tiles per workgroup.
     */
    CK_TILE_HOST_DEVICE index_t get_dp_tiles_per_cta() const noexcept;

    /**
     * @brief Returns the total number of DP tiles left over when `dp_tiles_` is not evenly
     * divisible by `grid_`.
     */
    CK_TILE_HOST_DEVICE index_t get_extra_dp_tiles() const noexcept;

    protected:
    index_t dp_tiles_per_cta_;
    index_t extra_dp_tiles_;
};

/**
 * @brief Non-Persistent Stream-K tile partitioner derived struct.
 *
 * This partitioner is responsible for mapping workgroups to tiles in the C tensor
 * for the Stream-K algorithm when using a Non-Persistent approach where extra workgroups
 * are allocated for the data parallel section.
 *
 *  @tparam BlockGemmShapeType  A class providing basic GEMM parameters.
 *  @tparam ReductionStrategyType   An enum that defines the reduction strategy for the results in
 * the C Tensor.
 */
template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategyType>
struct StreamKTilePartitioner<BlockGemmShapeType, ReductionStrategyType, false>
    : StreamKTilePartitionerBase<BlockGemmShapeType, ReductionStrategyType>
{
    StreamKTilePartitioner(ck_tile::index_t m,
                           ck_tile::index_t n,
                           ck_tile::index_t k,
                           ck_tile::index_t grid);

    public:
    static constexpr bool PERSISTENT = false;
    /**
     * @brief Calculates the launching grid size for the Stream-K kernel. In the Non-Persistent
     * case, extra workgroups are allocated for the data parallel section, making the grid
     * size the total number of Stream-K and data parallel workgroups.
     *
     * @return dim_3           The launching grid size for the kernel.
     */
    CK_TILE_HOST auto grid_size() const noexcept -> dim3;

    /**
     * @brief Returns the total number of DP workgroups.
     */
    CK_TILE_HOST_DEVICE index_t get_dp_ctas() const noexcept;

    /**
     * @brief Returns starting DP workgroup index. It is always zero.
     */
    CK_TILE_HOST_DEVICE index_t get_dp_start_block_idx() const noexcept;

    /**
     * @brief The index that starts the Stream-K workgroups. It is set to the number of `dp_tiles_`.
     */
    CK_TILE_HOST_DEVICE index_t get_sk_start_block_idx() const noexcept;

    protected:
    index_t dp_ctas_;
    index_t dp_start_block_idx_;
    index_t sk_start_block_idx_;
};

} // namespace ck_tile

#include "streamk_gemm_tile_partitioner_impl.hpp"
