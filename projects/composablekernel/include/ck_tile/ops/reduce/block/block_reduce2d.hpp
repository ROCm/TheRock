// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/core/utility/reduce_operator_accumulate.hpp"

namespace ck_tile {

// BlockReduce2d implements a hierarchical 2D reduction operator that reduces data along the second
// dimension using a user-specified reduction function.
//
// The reduction is performed in a three-stage hierarchical approach:
//
// STAGE 1: Thread-level reduction (BlockReduce2d)
// ===============================================
// - Each thread processes multiple elements from the input tensor within its assigned data
// partition
// - Reduction is performed locally within each thread by iterating over assigned elements
// - ReducePacksPerXDim controls how many elements sweep_tile processes in one iteration per
// dimension
//   (e.g., {1,1} = 1 element at a time from each dimension, {2,4} = 2 from dim0, 4 from dim1)
// - Results are accumulated into a thread-local output tensor stored in registers
// - The output tensor distribution is derived from the input tensor's distribution using
//   make_reduce_tile_distribution_encoding() to handle dimension reduction
//
// STAGE 2: Warp-level reduction (BlockReduce2dSync)
// ================================================
// - Performs inter-thread reduction within each warp
// - Uses warp shuffle operations to exchange data between threads in the same warp
// - Implements a tree-reduction pattern with power-of-2 stages
// - Only reduces along dimensions that map to lane IDs within the warp
//
// STAGE 3: Cross-warp reduction (BlockReduce2dCrossWarpSync)
// ========================================================
// - Performs reduction across multiple warps within the same thread block
// - Uses shared memory (LDS) to facilitate data exchange between warps
// - Each warp's lane-0 thread stores its partial results to shared memory
// - All threads participate in loading and reducing data from shared memory
// - Implements block-level synchronization to ensure memory consistency

// BlockReduce2d: Thread-level reduction (Stage 1)
template <typename Problem_, typename Policy_ = void>
struct BlockReduce2d
{
    // Thread-level reduction implementation
    using Problem         = remove_cvref_t<Problem_>;
    using XDataType       = typename Problem::XDataType;
    using ComputeDataType = typename Problem::ComputeDataType;

    CK_TILE_DEVICE constexpr BlockReduce2d() {}

    private:
    template <bool kProcessIndex,
              typename XDistributedTensor_,
              typename YDistributedTensor_,
              typename YIndexDistributedTensor_,
              typename ReduceFunc,
              typename IndexCalculatorFunc,
              typename ReducePacksPerXDim>
    CK_TILE_DEVICE void reduce_impl(const XDistributedTensor_& x_tensor,
                                    YDistributedTensor_& y_tensor,
                                    YIndexDistributedTensor_& y_index_tensor,
                                    const ReduceFunc& reduce_func,
                                    const IndexCalculatorFunc& index_calculator,
                                    ReducePacksPerXDim)
    {
        sweep_tile<XDistributedTensor_>(
            [&](auto... idx_) {
                constexpr auto idx_0 = make_tuple(make_tuple(idx_[number<0>{}]...)[number<0>{}]);

                (..., [&](auto idx) {
                    auto val = ck_tile::type_convert<ComputeDataType>(x_tensor[idx]);

                    if constexpr(kProcessIndex)
                    {

                        const auto x_indices = get_x_indices_from_distributed_indices(
                            XDistributedTensor_::get_tile_distribution(), idx);
                        const auto new_idx = index_calculator(x_indices);
                        auto current_idx   = y_index_tensor(idx_0);

                        AccumulateWithIndex{}(
                            reduce_func, y_tensor(idx_0), current_idx, val, new_idx);

                        y_index_tensor(idx_0) =
                            type_convert<typename YIndexDistributedTensor_::DataType>(current_idx);
                    }
                    else
                    {
                        Accumulate{}(reduce_func, y_tensor(idx_0), val);
                    }
                }(idx_));
            },
            ReducePacksPerXDim{});
    }

    public:
    // Overload for non-index tracking
    template <
        typename XDistributedTensor_,
        typename YDistributedTensor_,
        typename ReduceFunc,
        typename ReducePacksPerXDim =
            uniform_sequence_gen_t<2, 1>> // {1,1} = process 1 element at a time from each dimension
    CK_TILE_DEVICE void operator()(const XDistributedTensor_& x_tensor,
                                   YDistributedTensor_& y_tensor,
                                   const ReduceFunc& reduce_func,
                                   ReducePacksPerXDim = {})
    {
        reduce_impl<false>(
            x_tensor,
            y_tensor,
            y_tensor, // dummy
            reduce_func,
            [](auto) { return 0; }, // dummy
            ReducePacksPerXDim{});
    }

    // Overload for index tracking
    template <typename XDistributedTensor_,
              typename YDistributedTensor_,
              typename YIndexDistributedTensor_,
              typename ReduceFunc,
              typename IndexCalculatorFunc,
              typename ReducePacksPerXDim = uniform_sequence_gen_t<2, 1>>
    CK_TILE_DEVICE void operator()(const XDistributedTensor_& x_tensor,
                                   YDistributedTensor_& y_tensor,
                                   YIndexDistributedTensor_& y_index_tensor,
                                   const ReduceFunc& reduce_func,
                                   const IndexCalculatorFunc& index_calculator,
                                   ReducePacksPerXDim = {})
    {
        reduce_impl<Problem::kOutputIndex>(x_tensor,
                                           y_tensor,
                                           y_index_tensor,
                                           reduce_func,
                                           index_calculator,
                                           ReducePacksPerXDim{});
    }

#if 0
        constexpr auto I0 = number<0>{};
        constexpr auto I1 = number<1>{};
        constexpr auto spans = XDistributedTensor_::get_distributed_spans();

        // FIXME: hard coded to reduce 2nd axis
        sweep_tile_span(spans[I0], [&](auto dstr_idx_i0) {
            constexpr auto y_dstr_idx = make_tuple(dstr_idx_i0);

            auto y = y_tensor[y_dstr_idx];

            sweep_tile_span(spans[I1], [&](auto dstr_idx_i1) {
                constexpr auto in_dstr_idx = make_tuple(dstr_idx_i0, dstr_idx_i1);
                const auto x = ck_tile::type_convert<ComputeDataType>(x_tensor[in_dstr_idx]);

                y = reduce_func(y, x);
            });

            y_tensor(y_dstr_idx) = y;
        });
#endif

    template <typename XDistributedTensor_>
    CK_TILE_DEVICE static auto MakeYBlockTile()
    {
        // FIXME: hard coded to reduce 2nd axis
        constexpr auto reduce_dims = sequence<1>{};

        constexpr auto dstr =
            make_static_tile_distribution(detail::make_reduce_tile_distribution_encoding(
                XDistributedTensor_::get_tile_distribution()
                    .get_static_tile_distribution_encoding(),
                reduce_dims));

        auto tensor = make_static_distributed_tensor<ComputeDataType>(dstr);

        return tensor;
    }

    template <typename XDistributedTensor_, typename IndexDataType = index_t>
    CK_TILE_DEVICE static auto MakeYIndexBlockTile()
    {
        static_assert(std::is_same_v<XDataType, typename XDistributedTensor_::DataType>, "wrong!");

        // FIXME: hard coded to reduce 2nd axis
        constexpr auto reduce_dims = sequence<1>{};

        constexpr auto dstr =
            make_static_tile_distribution(detail::make_reduce_tile_distribution_encoding(
                XDistributedTensor_::get_tile_distribution()
                    .get_static_tile_distribution_encoding(),
                reduce_dims));

        auto tensor = make_static_distributed_tensor<IndexDataType>(dstr);

        return tensor;
    }

    // uniform_sequence_gen_t<NSize, Value> generates sequence of NSize elements filled with Value
    // e.g., uniform_sequence_gen_t<2, 1> → {1, 1} and uniform_sequence_gen_t<3, 4> → {4, 4, 4}
    template <typename XDistributedTensor_,
              typename ReduceFunc,
              typename ReducePacksPerXDim = uniform_sequence_gen_t<2, 1>>
    CK_TILE_DEVICE auto operator()(const XDistributedTensor_& x_tensor,
                                   const ComputeDataType& reduce_init,
                                   const ReduceFunc& reduce_func,
                                   ReducePacksPerXDim = {})
    {
        auto y_tensor = MakeYBlockTile<XDistributedTensor_>();
        set_tile(y_tensor, reduce_init);
        (*this)(x_tensor, y_tensor, reduce_func, ReducePacksPerXDim{});

        return y_tensor;
    }
};

// BlockReduce2dSync: Warp-level reduction (Stage 2)
template <typename Problem_, typename Policy_ = void>
struct BlockReduce2dSync
{
    using Problem = remove_cvref_t<Problem_>;

    private:
    template <bool kProcessIndex,
              typename YDistributedTensor_,
              typename YIndexDistributedTensor_,
              typename ReduceFunc>
    CK_TILE_DEVICE void reduce_impl(YDistributedTensor_& y_tensor,
                                    YIndexDistributedTensor_& y_index_tensor,
                                    const ReduceFunc& reduce_func)
    {
        using Dstr             = typename YDistributedTensor_::StaticTileDistribution;
        using DstrEncode       = typename Dstr::DstrEncode;
        using DstrEncodeDetail = typename DstrEncode::detail;

        constexpr index_t NDimP = Dstr::get_num_of_dimension_p();
        constexpr index_t NDimR = Dstr::get_num_of_dimension_r();

        constexpr index_t idim_p_lane = NDimP - 1;

        // const auto ps_idx = make_array<index_t>(get_warp_id(), get_lane_id());
        // const auto rs_idx =
        //     y_tensor.get_tile_distribution().calculate_rs_index_from_ps_index(ps_idx);

        constexpr index_t thread_buf_size = YDistributedTensor_::get_thread_buffer_size();

        // loop over thread data
        static_for<0, thread_buf_size, 1>{}([&](auto i) {
            auto v_local = y_tensor.get_thread_buffer()[i];

            using IndexDataType = typename YIndexDistributedTensor_::DataType;
            IndexDataType idx_local{};

            if constexpr(kProcessIndex)
            {
                idx_local = y_index_tensor.get_thread_buffer()[i];
            }

            // cross-lane reduce for replication
            // only reduce on R dimension correspond to lane
            // (lane id maps to this R dimension)
            static_for<0, NDimR, 1>{}([&](auto idim_r) {
                // FIXME: nasty to use does_p_own_r_
                if constexpr(DstrEncodeDetail::does_p_own_r_[idim_p_lane][idim_r])
                {
                    constexpr index_t r_length = DstrEncode::rs_lengths_[idim_r];

                    constexpr index_t lid_over_rid_derivative =
                        DstrEncodeDetail::ps_over_rs_derivative_[idim_p_lane][idim_r];

                    static_assert(is_power_of_two_integer(r_length),
                                  "wrong! only support power of 2 reduction");

                    constexpr index_t nstage = integer_log2_floor(r_length);

                    // reduction sweep forward
                    static_for<0, nstage, 1>{}([&](auto istage) {
                        // xor
                        index_t src_lane =
                            (__lane_id()) ^
                            (number<lid_over_rid_derivative << istage.value>{}.value);

                        // pull data from remote lane
                        const auto v_remote = warp_shuffle(v_local, src_lane);

                        if constexpr(kProcessIndex)
                        {
                            const auto idx_remote = warp_shuffle(idx_local, src_lane);

                            AccumulateWithIndex{}(
                                reduce_func, v_local, idx_local, v_remote, idx_remote);
                        }
                        else
                        {
                            Accumulate{}(reduce_func, v_local, v_remote);
                        }
                    });
                }
            });

            // TODO - Do we need to broadcast to other lane?
            y_tensor.get_thread_buffer()(i) = v_local;

            if constexpr(kProcessIndex)
            {
                y_index_tensor.get_thread_buffer()(i) = idx_local;
            }
        });
    }

    public:
    template <typename YDistributedTensor_, typename ReduceFunc>
    CK_TILE_DEVICE void operator()(YDistributedTensor_& y_tensor, const ReduceFunc& reduce_func)
    {
        reduce_impl<false>(y_tensor, y_tensor, reduce_func);
    }

    template <typename YDistributedTensor_, typename YIndexDistributedTensor_, typename ReduceFunc>
    CK_TILE_DEVICE void operator()(YDistributedTensor_& y_tensor,
                                   YIndexDistributedTensor_& y_index_tensor,
                                   const ReduceFunc& reduce_func)
    {
        reduce_impl<Problem::kOutputIndex>(y_tensor, y_index_tensor, reduce_func);
    }
};

// BlockReduce2dCrossWarpSync: Cross-warp reduction (Stage 3)
template <typename Problem_, typename Policy_ = void>
struct BlockReduce2dCrossWarpSync
{
    using Problem    = remove_cvref_t<Problem_>;
    using BlockShape = typename Problem::BlockShape;

    template <typename YDistributedTensor_>
    CK_TILE_DEVICE static constexpr index_t GetReduceWarps()
    {
        constexpr index_t num_reduce_warps = [&]() {
            using Dstr             = typename YDistributedTensor_::StaticTileDistribution;
            using DstrEncode       = typename Dstr::DstrEncode;
            using DstrEncodeDetail = typename DstrEncode::detail;

            constexpr index_t NDimR = Dstr::get_num_of_dimension_r();

            constexpr index_t idim_p_warp = 0;

            index_t len_ = 1;
            static_for<0, NDimR, 1>{}([&](auto idim_r) {
                if constexpr(DstrEncodeDetail::does_p_own_r_[idim_p_warp][idim_r])
                {
                    constexpr index_t r_length = DstrEncode::rs_lengths_[idim_r];
                    len_ *= r_length;
                }
            });
            return len_;
        }();
        return num_reduce_warps;
    }

    // return in byte
    template <typename YDistributedTensor_>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        using DataType                    = typename YDistributedTensor_::DataType;
        constexpr index_t thread_buf_size = YDistributedTensor_::get_thread_buffer_size();

        // we need to store all data from every wave into smem
        // e.g. 2x2 reduce along N
        //     -------------> reduce N
        //    | w0 | w1 |   ___>      | w01 |
        //    | w2 | w3 |             | w23 |
        //
        //   -> store data from every wave into LDS
        //
        //
        //     -------------> reduce N
        //    | w0 | w1 | w2 | w3 |   ----->  | w0123 |
        //
        //   -> also store data from every wave into LDS
        constexpr index_t num_warps = BlockShape::BlockSize / get_warp_size();
        return num_warps * thread_buf_size * sizeof(DataType);
    }

    // return in byte - separate shared memory size calculation for indices
    template <typename YIndexDistributedTensor_>
    CK_TILE_HOST_DEVICE static constexpr index_t GetIndicesSmemSize()
    {
        using IndexDataType               = typename YIndexDistributedTensor_::DataType;
        constexpr index_t thread_buf_size = YIndexDistributedTensor_::get_thread_buffer_size();
        constexpr index_t num_warps       = BlockShape::BlockSize / get_warp_size();
        return num_warps * thread_buf_size * sizeof(IndexDataType);
    }

    private:
    template <bool kProcessIndex,
              typename YDistributedTensor_,
              typename YIndexDistributedTensor_,
              typename ReduceFunc>
    CK_TILE_DEVICE void reduce_impl(YDistributedTensor_& y_tensor,
                                    YIndexDistributedTensor_& y_index_tensor,
                                    void* smem,
                                    void* smem_indices_ptr,
                                    const ReduceFunc& reduce_func)
    {
        using DataType      = typename YDistributedTensor_::DataType;
        using IndexDataType = typename YIndexDistributedTensor_::DataType;

        constexpr index_t thread_buf_size = YDistributedTensor_::get_thread_buffer_size();

        DataType* smem_ptr          = reinterpret_cast<DataType*>(smem);
        IndexDataType* smem_indices = nullptr;
        if constexpr(kProcessIndex)
        {
            smem_indices = reinterpret_cast<IndexDataType*>(smem_indices_ptr);
        }

        const index_t lane_id = get_lane_id();
        const index_t warp_id = get_warp_id();

        constexpr index_t num_warps        = BlockShape::BlockSize / get_warp_size();
        constexpr index_t num_reduce_warps = GetReduceWarps<YDistributedTensor_>();

        if constexpr(num_reduce_warps == 1)
            return;
        block_sync_lds();
        // Each warp's lane 0 writes its partial results to shared memory
        const index_t smem_offset = warp_id;
        if(lane_id == 0)
        {
            static_for<0, thread_buf_size, 1>{}([&](auto i) {
                // Store the i-th element of this warp's thread_buffer into SMEM
                smem_ptr[smem_offset + i * num_warps] = y_tensor.get_thread_buffer()[i];
                if constexpr(kProcessIndex)
                {
                    smem_indices[smem_offset + i * num_warps] =
                        y_index_tensor.get_thread_buffer()[i];
                }
            });
        }
        block_sync_lds();

        // We let each warp holds a duplication to do reduction.
        const index_t local_warp_id = warp_id / num_reduce_warps;
        const index_t local_smem_os = local_warp_id * num_reduce_warps;

        static_for<0, thread_buf_size, 1>{}([&](auto i) {
            DataType v[num_reduce_warps];
            [[maybe_unused]] std::
                conditional_t<kProcessIndex, IndexDataType[num_reduce_warps], IndexDataType> idx_v;

            static_for<0, num_reduce_warps, 1>{}([&](auto idx) {
                v[idx] = smem_ptr[i * num_warps + local_smem_os + idx];
                if constexpr(kProcessIndex)
                {
                    idx_v[idx] = smem_indices[i * num_warps + local_smem_os + idx];
                }
            });

            static_assert(is_power_of_two_integer(num_reduce_warps),
                          "wrong! only support power of 2 reduction");

            constexpr index_t nstage = integer_log2_floor(num_reduce_warps);

            static_for<0, nstage, 1>{}([&](auto istage) {
                constexpr index_t stride = 1 << istage.value;
                static_for<0, num_reduce_warps, stride * 2>{}([&](auto idx_) {
                    constexpr index_t i0 = idx_();
                    constexpr index_t i1 = idx_ + stride;
                    if constexpr(i1 < num_reduce_warps)
                    {
                        if constexpr(kProcessIndex)
                        {
                            AccumulateWithIndex{}(reduce_func, v[i0], idx_v[i0], v[i1], idx_v[i1]);
                        }
                        else
                        {
                            Accumulate{}(reduce_func, v[i0], v[i1]);
                        }
                    }
                });
            });

            y_tensor.get_thread_buffer()(i) = v[0];
            if constexpr(kProcessIndex)
            {
                y_index_tensor.get_thread_buffer()(i) = idx_v[0];
            }
        });
    }

    public:
    template <typename YDistributedTensor_, typename ReduceFunc>
    CK_TILE_DEVICE void
    operator()(YDistributedTensor_& y_tensor, void* smem, const ReduceFunc& reduce_func)
    {
        reduce_impl<false>(y_tensor, y_tensor, smem, nullptr, reduce_func);
    }

    template <typename YDistributedTensor_, typename YIndexDistributedTensor_, typename ReduceFunc>
    CK_TILE_DEVICE void operator()(YDistributedTensor_& y_tensor,
                                   YIndexDistributedTensor_& y_index_tensor,
                                   void* smem,
                                   void* smem_indices,
                                   const ReduceFunc& reduce_func)
    {
        reduce_impl<Problem::kOutputIndex>(
            y_tensor, y_index_tensor, smem, smem_indices, reduce_func);
    }
};

template <typename Problem_, typename Policy_ = void>
struct BlockReduce2dLinearCrossWarpSync
{
    using Problem    = remove_cvref_t<Problem_>;
    using BlockShape = typename Problem::BlockShape;

    template <typename YDistributedTensor_>
    CK_TILE_DEVICE static constexpr index_t GetReduceWarps()
    {
        constexpr index_t num_reduce_warps = [&]() {
            using Dstr             = typename YDistributedTensor_::StaticTileDistribution;
            using DstrEncode       = typename Dstr::DstrEncode;
            using DstrEncodeDetail = typename DstrEncode::detail;

            constexpr index_t NDimR = Dstr::get_num_of_dimension_r();

            constexpr index_t idim_p_warp = 0;

            index_t len_ = 1;
            static_for<0, NDimR, 1>{}([&](auto idim_r) {
                if constexpr(DstrEncodeDetail::does_p_own_r_[idim_p_warp][idim_r])
                {
                    constexpr index_t r_length = DstrEncode::rs_lengths_[idim_r];
                    len_ *= r_length;
                }
            });
            return len_;
        }();
        return num_reduce_warps;
    }

    // return in byte
    template <typename YDistributedTensor_>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        using DataType                    = typename YDistributedTensor_::DataType;
        constexpr index_t thread_buf_size = YDistributedTensor_::get_thread_buffer_size();

        // we need to store all data from every wave into smem
        // e.g. 2x2 reduce along N
        //     -------------> reduce N
        //    | w0 | w1 |   ___>      | w01 |
        //    | w2 | w3 |             | w23 |
        //
        //   -> store data from every wave into LDS
        //
        //
        //     -------------> reduce N
        //    | w0 | w1 | w2 | w3 |   ----->  | w0123 |
        //
        //   -> also store data from every wave into LDS
        constexpr index_t num_warps = BlockShape::BlockSize / get_warp_size();
        return num_warps * thread_buf_size * sizeof(DataType);
    }

    // return in byte - separate shared memory size calculation for indices
    template <typename YIndexDistributedTensor_>
    CK_TILE_HOST_DEVICE static constexpr index_t GetIndicesSmemSize()
    {
        using IndexDataType               = typename YIndexDistributedTensor_::DataType;
        constexpr index_t thread_buf_size = YIndexDistributedTensor_::get_thread_buffer_size();
        constexpr index_t num_warps       = BlockShape::BlockSize / get_warp_size();
        return num_warps * thread_buf_size * sizeof(IndexDataType);
    }

    private:
    template <bool kProcessIndex,
              typename YDistributedTensor_,
              typename YIndexDistributedTensor_,
              typename ReduceFunc>
    CK_TILE_DEVICE void reduce_impl(YDistributedTensor_& y_tensor,
                                    YIndexDistributedTensor_& y_index_tensor,
                                    void* smem,
                                    void* smem_indices_ptr,
                                    const ReduceFunc& reduce_func)
    {
        using DataType      = typename YDistributedTensor_::DataType;
        using IndexDataType = typename YIndexDistributedTensor_::DataType;

        constexpr index_t thread_buf_size = YDistributedTensor_::get_thread_buffer_size();

        DataType* smem_ptr          = reinterpret_cast<DataType*>(smem);
        IndexDataType* smem_indices = nullptr;
        if constexpr(kProcessIndex)
        {
            smem_indices = reinterpret_cast<IndexDataType*>(smem_indices_ptr);
        }

        const index_t lane_id           = get_lane_id();
        const index_t warp_id           = get_warp_id();
        constexpr auto num_reduce_warps = GetReduceWarps<YDistributedTensor_>();
        constexpr index_t num_warps     = BlockShape::BlockSize / get_warp_size();
        const index_t smem_offset       = warp_id;

        // skip if nonthing to do
        if constexpr(num_reduce_warps == 1)
            return;

        // store into smem only for lane-0 within one warp
        if(lane_id == 0)
        {
            static_for<0, thread_buf_size, 1>{}([&](auto i) {
                smem_ptr[smem_offset + i * num_warps] = y_tensor.get_thread_buffer()[i];
                if constexpr(kProcessIndex)
                {
                    smem_indices[smem_offset + i * num_warps] =
                        y_index_tensor.get_thread_buffer()[i];
                }
            });
        }
        block_sync_lds();

        // load from smem. here we let everythread to do compute :)
        index_t local_warp_id = warp_id / num_reduce_warps;
        index_t local_smem_os = local_warp_id * num_reduce_warps;

        DataType all_scratch[thread_buf_size * num_reduce_warps];
        [[maybe_unused]] std::conditional_t<kProcessIndex,
                                            IndexDataType[thread_buf_size * num_reduce_warps],
                                            IndexDataType> all_indices;

        // Load data from shared memory
        static_for<0, thread_buf_size, 1>{}([&](auto i_0) {
            static_for<0, num_reduce_warps, 1>{}([&](auto i_1) {
                all_scratch[i_0 * num_reduce_warps + i_1] =
                    smem_ptr[i_0 * num_warps + local_smem_os + i_1];

                if constexpr(kProcessIndex)
                {
                    all_indices[i_0 * num_reduce_warps + i_1] =
                        smem_indices[i_0 * num_warps + local_smem_os + i_1];
                }
            });
        });
        block_sync_lds(); // TODO: we don't need sync here

        // Perform reduction
        static_for<0, thread_buf_size, 1>{}([&](auto i_0) {
            // TODO: use descriptor for this
            auto v_local = all_scratch[i_0 * num_reduce_warps];

            IndexDataType idx_local{};
            if constexpr(kProcessIndex)
            {
                idx_local = all_indices[i_0 * num_reduce_warps];
            }

            // further reduce mean/var
            static_for<0, num_reduce_warps - 1, 1>{}([&](auto i_1_n1) {
                constexpr auto i_1      = number<i_1_n1 + 1>{};
                const DataType v_remote = all_scratch[i_0 * num_reduce_warps + i_1];

                if constexpr(kProcessIndex)
                {
                    const IndexDataType idx_remote = all_indices[i_0 * num_reduce_warps + i_1];

                    bool changed = false;
                    v_local      = reduce_func(v_local, v_remote, changed);
                    if(changed)
                    {
                        idx_local = idx_remote;
                    }
                }
                else
                {
                    v_local = reduce_func(v_local, v_remote);
                }
            });

            y_tensor.get_thread_buffer()(i_0) = v_local;
            if constexpr(kProcessIndex)
            {
                y_index_tensor.get_thread_buffer()(i_0) = idx_local;
            }
        });
    }

    public:
    template <typename YDistributedTensor_, typename ReduceFunc>
    CK_TILE_DEVICE void
    operator()(YDistributedTensor_& y_tensor, void* smem, const ReduceFunc& reduce_func)
    {
        reduce_impl<false>(y_tensor, y_tensor, smem, nullptr, reduce_func);
    }

    template <typename YDistributedTensor_, typename YIndexDistributedTensor_, typename ReduceFunc>
    CK_TILE_DEVICE void operator()(YDistributedTensor_& y_tensor,
                                   YIndexDistributedTensor_& y_index_tensor,
                                   void* smem,
                                   void* smem_indices,
                                   const ReduceFunc& reduce_func)
    {
        reduce_impl<Problem::kOutputIndex>(
            y_tensor, y_index_tensor, smem, smem_indices, reduce_func);
    }
};

} // namespace ck_tile
