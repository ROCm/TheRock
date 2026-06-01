// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/core/algorithm/coordinate_transform.hpp"
#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/container/tuple.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/math.hpp"
#include "ck_tile/host/concat.hpp"
#include "ck_tile/ops/gemm_quant/pipeline/tile_gemm_quant_traits.hpp"

namespace ck_tile {

namespace detail {
// Helper templates for safe type extraction
template <typename, typename Default, typename = void>
struct get_aq_layout_or
{
    using type = Default;
};

template <typename T, typename Default>
struct get_aq_layout_or<T, Default, std::void_t<typename T::AQLayout>>
{
    using type = typename T::AQLayout;
};

template <typename, typename Default, typename = void>
struct get_bq_layout_or
{
    using type = Default;
};

template <typename T, typename Default>
struct get_bq_layout_or<T, Default, std::void_t<typename T::BQLayout>>
{
    using type = typename T::BQLayout;
};

template <typename, typename Default, typename = void>
struct get_aq_data_type_or
{
    using type = Default;
};

template <typename T, typename Default>
struct get_aq_data_type_or<T, Default, std::void_t<typename T::AQDataType>>
{
    using type = typename T::AQDataType;
};

template <typename, typename Default, typename = void>
struct get_bq_data_type_or
{
    using type = Default;
};

template <typename T, typename Default>
struct get_bq_data_type_or<T, Default, std::void_t<typename T::BQDataType>>
{
    using type = typename T::BQDataType;
};

template <typename, typename = void>
struct is_quantpreshuffle_enabled
{
    static constexpr bool value = false;
};

template <typename T>
struct is_quantpreshuffle_enabled<T, std::void_t<decltype(T::PreshuffleQuant)>>
{
    static constexpr bool value = T::PreshuffleQuant;
};

template <typename, typename = void>
struct is_preshuffleB_enabled
{
    static constexpr bool value = false;
};

template <typename T>
struct is_preshuffleB_enabled<T, std::void_t<decltype(T::PreshuffleB)>>
{
    static constexpr bool value = T::PreshuffleB;
};
} // namespace detail

struct QuantGemmProblem
{
    CK_TILE_HOST QuantGemmProblem() = default;
    CK_TILE_HOST QuantGemmProblem(index_t M_,
                                  index_t N_,
                                  index_t K_,
                                  index_t QK_A_,
                                  index_t QK_B_,
                                  index_t stride_A_,
                                  index_t stride_B_,
                                  index_t stride_C_,
                                  index_t stride_AQ_,
                                  index_t stride_BQ_)
        : M(M_),
          N(N_),
          K(K_),
          QK_A(QK_A_),
          QK_B(QK_B_),
          stride_A(stride_A_),
          stride_B(stride_B_),
          stride_C(stride_C_),
          stride_AQ(stride_AQ_),
          stride_BQ(stride_BQ_)
    {
    }

    index_t M;
    index_t N;
    index_t K;
    index_t QK_A;
    index_t QK_B;
    index_t stride_A;
    index_t stride_B;
    index_t stride_C;
    index_t stride_AQ;
    index_t stride_BQ;
};

struct QuantGemmHostArgs : public QuantGemmProblem
{
    CK_TILE_HOST QuantGemmHostArgs() = default;
    CK_TILE_HOST QuantGemmHostArgs(const void* a_ptr_,
                                   const void* b_ptr_,
                                   void* c_ptr_,
                                   const void* aq_ptr_,
                                   const void* bq_ptr_,
                                   index_t k_batch_,
                                   index_t M_,
                                   index_t N_,
                                   index_t K_,
                                   index_t QK_A_,
                                   index_t QK_B_,
                                   index_t stride_A_,
                                   index_t stride_B_,
                                   index_t stride_C_,
                                   index_t stride_AQ_,
                                   index_t stride_BQ_)
        : QuantGemmProblem(
              M_, N_, K_, QK_A_, QK_B_, stride_A_, stride_B_, stride_C_, stride_AQ_, stride_BQ_),
          a_ptr(a_ptr_),
          b_ptr(b_ptr_),
          aq_ptr(aq_ptr_),
          bq_ptr(bq_ptr_),
          c_ptr(c_ptr_),
          k_batch(k_batch_)
    {
    }

    const void* a_ptr  = nullptr;
    const void* b_ptr  = nullptr;
    const void* aq_ptr = nullptr;
    const void* bq_ptr = nullptr;
    void* c_ptr        = nullptr;
    index_t k_batch    = 0;
};

struct QuantGemmKernelArgs
{
    const void* a_ptr;
    const void* b_ptr;
    const void* aq_ptr;
    const void* bq_ptr;
    void* c_ptr;
    index_t M;
    index_t N;
    index_t K;
    index_t QK_A;
    index_t QK_B;
    index_t stride_A;
    index_t stride_B;
    index_t stride_C;
    index_t stride_AQ;
    index_t stride_BQ;
    index_t k_batch;
};

template <typename TilePartitioner_,
          typename GemmPipeline_,
          typename EpiloguePipeline_,
          QuantType QuantType_>
struct QuantGemmKernel
{
    using TilePartitioner  = remove_cvref_t<TilePartitioner_>;
    using GemmPipeline     = remove_cvref_t<GemmPipeline_>;
    using EpiloguePipeline = remove_cvref_t<EpiloguePipeline_>;
    using ALayout          = remove_cvref_t<typename GemmPipeline::ALayout>;
    using BLayout          = remove_cvref_t<typename GemmPipeline::BLayout>;
    using CLayout          = remove_cvref_t<typename GemmPipeline::CLayout>;

    using AQLayout = remove_cvref_t<
        typename detail::get_aq_layout_or<GemmPipeline, typename GemmPipeline::ALayout>::type>;
    using BQLayout = remove_cvref_t<
        typename detail::get_bq_layout_or<GemmPipeline, typename GemmPipeline::BLayout>::type>;

    static constexpr index_t kBlockSize = GemmPipeline::BlockSize;
    static constexpr bool PreshuffleQuant =
        detail::is_quantpreshuffle_enabled<GemmPipeline_>::value;
    static constexpr bool PreshuffleB = detail::is_preshuffleB_enabled<GemmPipeline_>::value;

    using ADataType   = remove_cvref_t<typename GemmPipeline::ADataType>;
    using BDataType   = remove_cvref_t<typename GemmPipeline::BDataType>;
    using CDataType   = remove_cvref_t<typename EpiloguePipeline::ODataType>;
    using AccDataType = remove_cvref_t<typename EpiloguePipeline::AccDataType>;

    using AQDataType =
        remove_cvref_t<typename detail::get_aq_data_type_or<GemmPipeline, AccDataType>::type>;
    using BQDataType =
        remove_cvref_t<typename detail::get_bq_data_type_or<GemmPipeline, AccDataType>::type>;

    static constexpr auto I0 = number<0>(); // A Tensor
    static constexpr auto I1 = number<1>(); // AQ Tensor
    static constexpr auto I2 = number<2>(); // B Tensor
    static constexpr auto I3 = number<3>(); // BQ Tensor
    static constexpr auto I4 = number<4>(); // C Tensor

    static constexpr auto kQuantType = QuantType_;

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        // clang-format off
        return concat('_', "gemm_quant", gemm_prec_str<ADataType, BDataType>, GemmPipeline::GetName());
        // clang-format on
    }

    CK_TILE_HOST static constexpr auto GridSize(index_t M, index_t N, index_t KBatch)
    {
        return dim3(TilePartitioner::GridSize(M, N), 1, KBatch);
    }

    CK_TILE_HOST static auto BlockSize()
    {
        return is_wave32() ? dim3(kBlockSize / 2) : dim3(kBlockSize);
    }

    CK_TILE_HOST static constexpr QuantGemmKernelArgs
    MakeKernelArgs(const QuantGemmHostArgs& hostArgs)
    {
        return QuantGemmKernelArgs{hostArgs.a_ptr,
                                   hostArgs.b_ptr,
                                   hostArgs.aq_ptr,
                                   hostArgs.bq_ptr,
                                   hostArgs.c_ptr,
                                   hostArgs.M,
                                   hostArgs.N,
                                   hostArgs.K,
                                   hostArgs.QK_A,
                                   hostArgs.QK_B,
                                   hostArgs.stride_A,
                                   hostArgs.stride_B,
                                   hostArgs.stride_C,
                                   hostArgs.stride_AQ,
                                   hostArgs.stride_BQ,
                                   hostArgs.k_batch};
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return max(GemmPipeline::GetSmemSize(), EpiloguePipeline::GetSmemSize());
    }

    private:
    CK_TILE_DEVICE static constexpr index_t get_padding_size(index_t length, index_t alignment)
    {
        return ck_tile::integer_least_multiple(length, alignment) - length;
    };
    // ===================================================================
    // Helper: Create Pre-shuffled Quantization Tensor Descriptor
    // ===================================================================
    template <index_t KPerBlockBQ,
              index_t NPerBlockBQ,
              index_t NPerBlock,
              index_t WarpTileN,
              index_t GetVectorSizeBQ,
              typename BQDataType_>
    CK_TILE_DEVICE static auto
    MakePreshuffledQuantTensorView(const BQDataType_* bq_ptr, index_t N, index_t QN_B, index_t QK_B)
    {
        // Step 1: Calculate base BQ tensor dimensions
        // ----------------------------------------------------------
        // bq_x: Number of quantization groups in N dimension
        //       = N * KPerBlockBQ, where KPerBlockBQ is the number of
        //       K-dimension groups per block
        // bq_y: Number of quantization groups in K dimension
        //       = Total K groups (QK_B) / groups per block
        const auto bq_x = N * KPerBlockBQ;
        const auto bq_y = QK_B / KPerBlockBQ;

        const auto bq_desc = make_naive_tensor_descriptor(
            make_tuple(bq_y, bq_x), make_tuple(bq_x, 1), number<GetVectorSizeBQ>{}, number<1>{});

        // Step 2: First padding transformation (block-level alignment)
        // ----------------------------------------------------------
        // Pad the X dimension to be a multiple of block_tile_size to ensure
        // each thread block can process complete tiles without edge cases
        const auto block_tile_size = NPerBlockBQ * KPerBlockBQ;

        const auto bq_pad0_desc = transform_tensor_descriptor(
            bq_desc,
            make_tuple(make_pass_through_transform(bq_y),
                       make_right_pad_transform(bq_x, get_padding_size(bq_x, block_tile_size))),
            make_tuple(sequence<0>{}, sequence<1>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        // Step 3: Unmerge transformation (wave-level decomposition)
        // ----------------------------------------------------------
        // Split the X dimension into [wave_tile_count_x, wave_tile_size]
        // This separates the work into tiles that can be processed by
        // individual warps/waves
        const auto pad_bq_x          = bq_pad0_desc.get_lengths()[I1];
        const auto wave_tile_size    = ((QN_B <= WarpTileN) ? (WarpTileN / QN_B) : 1) * KPerBlockBQ;
        const auto wave_tile_count_x = ck_tile::integer_divide_ceil(pad_bq_x, wave_tile_size);

        const auto bq_unmerge_pad0_desc = transform_tensor_descriptor(
            bq_pad0_desc,
            make_tuple(make_pass_through_transform(bq_y),
                       make_unmerge_transform(make_tuple(wave_tile_count_x, wave_tile_size))),
            make_tuple(sequence<0>{}, sequence<1>{}),
            make_tuple(sequence<0>{}, sequence<1, 2>{}));

        // Step 4: Second padding transformation (warp-level alignment)
        // ----------------------------------------------------------
        // Pad wave_tile_size to be a multiple of warp_size (typically 32 or 64)
        // This ensures coalesced memory accesses within each warp
        const auto bq_pad1_desc = transform_tensor_descriptor(
            bq_unmerge_pad0_desc,
            make_tuple(make_pass_through_transform(bq_y),
                       make_pass_through_transform(wave_tile_count_x),
                       make_right_pad_transform(wave_tile_size,
                                                get_padding_size(wave_tile_size, get_warp_size()))),
            make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}));

        // Step 5: Final merge transformation (prepare for indexing)
        // ----------------------------------------------------------
        // Merge [bq_y, wave_tile_count_x] into a single outer dimension
        // This creates a 2D layout: [merged_outer_dim, pad_wave_size]
        // where merged_outer_dim = bq_y * wave_tile_count_x
        // This layout facilitates efficient block-to-data mapping
        const auto pad_wave_size = ck_tile::integer_least_multiple(wave_tile_size, get_warp_size());
        const auto bq_merge_pad1_desc = transform_tensor_descriptor(
            bq_pad1_desc,
            make_tuple(make_merge_transform(make_tuple(bq_y, wave_tile_count_x)),
                       make_pass_through_transform(pad_wave_size)),
            make_tuple(sequence<0, 1>{}, sequence<2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return make_tensor_view<address_space_enum::global>(bq_ptr, bq_merge_pad1_desc);
    }

    public:
    struct SplitKBatchOffset
    {
        __device__ SplitKBatchOffset(const QuantGemmKernelArgs& kargs,
                                     const std::size_t k_id = blockIdx.z)
        {
            constexpr auto K1   = GemmPipeline::BlockGemmShape::WarpTile::at(I2);
            const index_t K_t   = amd_wave_read_first_lane(kargs.k_batch * K1);
            const index_t KRead = amd_wave_read_first_lane((kargs.K + K_t - 1) / K_t * K1);

            if constexpr(std::is_same_v<tensor_layout::gemm::RowMajor, ALayout>)
            {
                a_k_split_offset = amd_wave_read_first_lane(k_id * KRead);
            }
            else if constexpr(std::is_same_v<tensor_layout::gemm::ColumnMajor, ALayout>)
            {
                a_k_split_offset = amd_wave_read_first_lane(k_id * KRead * kargs.stride_A);
            }

            if constexpr(std::is_same_v<tensor_layout::gemm::RowMajor, BLayout>)
            {
                b_k_split_offset = amd_wave_read_first_lane(k_id * KRead * kargs.stride_B);
            }
            else if constexpr(std::is_same_v<tensor_layout::gemm::ColumnMajor, BLayout>)
            {
                b_k_split_offset = amd_wave_read_first_lane(k_id * KRead);
            }

            if(k_id < static_cast<uint32_t>(kargs.k_batch - 1))
            {
                splitted_k = amd_wave_read_first_lane(KRead);
            }
            else
            {
                splitted_k = amd_wave_read_first_lane(kargs.K - KRead * (kargs.k_batch - 1));
            }
        }

        index_t a_k_split_offset;
        index_t b_k_split_offset;
        index_t splitted_k;
    };

    CK_TILE_DEVICE static auto MakeABlockWindow(const ADataType* a_ptr,
                                                const QuantGemmKernelArgs& kargs,
                                                const index_t k_size,
                                                const index_t i_m)
    {
        // Step 1: Create tensor view for A
        const auto& a_tensor_view = [&]() {
            if constexpr(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>)
            {
                return make_naive_tensor_view<address_space_enum::global>(
                    a_ptr,
                    make_tuple(kargs.M, k_size),
                    make_tuple(kargs.stride_A, 1),
                    number<GemmPipeline::GetVectorSizeA()>{},
                    number<1>{});
            }
            else
            {
                return make_naive_tensor_view<address_space_enum::global>(
                    a_ptr,
                    make_tuple(k_size, kargs.M),
                    make_tuple(kargs.stride_A, 1),
                    number<GemmPipeline::GetVectorSizeA()>{},
                    number<1>{});
            }
        }();

        // Step 2: Create padded view
        const auto& a_pad_view = [&]() {
            if constexpr(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>)
            {
                return pad_tensor_view(a_tensor_view,
                                       make_tuple(number<TilePartitioner::MPerBlock>{},
                                                  number<TilePartitioner::KPerBlock>{}),
                                       sequence<false, GemmPipeline::kPadK>{});
            }
            else
            {
                return pad_tensor_view(a_tensor_view,
                                       make_tuple(number<TilePartitioner::KPerBlock>{},
                                                  number<TilePartitioner::MPerBlock>{}),
                                       sequence<false, GemmPipeline::kPadM>{});
            }
        }();

        // Step 3: Create tile window
        const auto& a_block_window = [&]() {
            if constexpr(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>)
            {
                return make_tile_window(a_pad_view,
                                        make_tuple(number<TilePartitioner::MPerBlock>{},
                                                   number<TilePartitioner::KPerBlock>{}),
                                        {i_m, 0});
            }
            else
            {
                return make_tile_window(a_pad_view,
                                        make_tuple(number<TilePartitioner::KPerBlock>{},
                                                   number<TilePartitioner::MPerBlock>{}),
                                        {0, i_m});
            }
        }();

        return a_block_window;
    }

    CK_TILE_DEVICE static auto MakeAQBlockWindow(const AQDataType* aq_ptr,
                                                 const QuantGemmKernelArgs& kargs,
                                                 const index_t i_m,
                                                 const index_t i_n)
    {
        // Step 1: Create tensor view for AQ
        const auto& aq_tensor_view = [&]() {
            if constexpr(kQuantType == QuantType::AQuantGrouped && PreshuffleQuant)
            {
                static_assert(std::is_same_v<AQLayout, tensor_layout::gemm::RowMajor>);
                const auto aq_x = kargs.M * GemmPipeline::KPerBlockAQ;
                const auto aq_y = kargs.QK_A / GemmPipeline::KPerBlockAQ;
                const auto aq_desc =
                    make_naive_tensor_descriptor(make_tuple(aq_y, aq_x),
                                                 make_tuple(aq_x, 1),
                                                 number<GemmPipeline::GetVectorSizeAQ()>{},
                                                 number<1>{});

                const auto block_tile_size = GemmPipeline::MPerBlock * GemmPipeline::KPerBlockAQ;
                const auto aq_pad0_desc    = transform_tensor_descriptor(
                    aq_desc,
                    make_tuple(
                        make_pass_through_transform(aq_y),
                        make_right_pad_transform(aq_x, get_padding_size(aq_x, block_tile_size))),
                    make_tuple(sequence<0>{}, sequence<1>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));

                const auto pad_aq_x = aq_pad0_desc.get_lengths()[I1];
                const auto wave_tile_size =
                    GemmPipeline::BlockGemmShape::WarpTile::at(I0) * GemmPipeline::KPerBlockAQ;
                const auto wave_tile_count_x =
                    ck_tile::integer_divide_ceil(pad_aq_x, wave_tile_size);

                const auto aq_unmerge_pad0_desc = transform_tensor_descriptor(
                    aq_pad0_desc,
                    make_tuple(
                        make_pass_through_transform(aq_y),
                        make_unmerge_transform(make_tuple(wave_tile_count_x, wave_tile_size))),
                    make_tuple(sequence<0>{}, sequence<1>{}),
                    make_tuple(sequence<0>{}, sequence<1, 2>{}));

                const auto aq_pad1_desc = transform_tensor_descriptor(
                    aq_unmerge_pad0_desc,
                    make_tuple(
                        make_pass_through_transform(aq_y),
                        make_pass_through_transform(wave_tile_count_x),
                        make_right_pad_transform(
                            wave_tile_size, get_padding_size(wave_tile_size, get_warp_size()))),
                    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}));

                const auto pad_wave_size =
                    ck_tile::integer_least_multiple(wave_tile_size, get_warp_size());
                const auto aq_merge_pad1_desc = transform_tensor_descriptor(
                    aq_pad1_desc,
                    make_tuple(make_merge_transform(make_tuple(aq_y, wave_tile_count_x)),
                               make_pass_through_transform(pad_wave_size)),
                    make_tuple(sequence<0, 1>{}, sequence<2>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));

                return make_tensor_view<address_space_enum::global>(aq_ptr, aq_merge_pad1_desc);
            }
            else if constexpr((kQuantType == QuantType::AQuantGrouped ||
                               kQuantType == QuantType::ABQuantGrouped) &&
                              !PreshuffleQuant)
            {
                if constexpr(std::is_same_v<AQLayout, tensor_layout::gemm::RowMajor>)
                {
                    return make_naive_tensor_view<address_space_enum::global>(
                        aq_ptr,
                        make_tuple(kargs.M, kargs.QK_A),
                        make_tuple(kargs.stride_AQ, 1),
                        number<GemmPipeline::GetVectorSizeAQ()>{},
                        number<1>{});
                }
                else // Column major AQ
                {
                    return make_naive_tensor_view<address_space_enum::global>(
                        aq_ptr,
                        make_tuple(kargs.QK_A, kargs.M),
                        make_tuple(kargs.stride_AQ, 1),
                        number<GemmPipeline::GetVectorSizeAQ()>{},
                        number<1>{});
                }
            }
            else if constexpr(kQuantType == QuantType::RowColQuant)
            {
                return make_naive_tensor_view<address_space_enum::global>(
                    aq_ptr,
                    make_tuple(kargs.M, kargs.N),
                    make_tuple(1, 0), // broadcasting over n
                    number<1>{},
                    number<1>{});
            }
            else
            {
                return nullptr;
            }
        }();

        // Step 2: Create tile window (no padding for AQ)
        const auto& aq_block_window = [&]() {
            if constexpr(kQuantType == QuantType::AQuantGrouped && PreshuffleQuant)
            {
                static_assert(std::is_same_v<AQLayout, tensor_layout::gemm::RowMajor>);
                using QuantGroupSize   = remove_cvref_t<typename GemmPipeline::QuantGroupSize>;
                constexpr auto block_m = TilePartitioner::MPerBlock;
                constexpr auto warp_m  = GemmPipeline::BlockGemmShape::WarpTile::at(I0);
                constexpr auto aqk_per_block = TilePartitioner::KPerBlock / QuantGroupSize::kK;
                constexpr auto tile_window_width =
                    ck_tile::integer_least_multiple(warp_m * aqk_per_block, get_warp_size());
                constexpr auto tile_window_height = block_m / warp_m;
                auto block_m_idx                  = i_m / block_m;
                return make_tile_window(
                    aq_tensor_view,
                    make_tuple(number<tile_window_height>{}, number<tile_window_width>{}),
                    {block_m_idx * tile_window_height, 0});
            }
            else if constexpr(kQuantType == QuantType::AQuantGrouped && !PreshuffleQuant)
            {
                using QuantGroupSize = remove_cvref_t<typename GemmPipeline::QuantGroupSize>;
                constexpr auto aqk_per_block = TilePartitioner::KPerBlock / QuantGroupSize::kK;
                constexpr auto block_m       = TilePartitioner::MPerBlock;
                if constexpr(std::is_same_v<AQLayout, tensor_layout::gemm::RowMajor>)
                {
                    return make_tile_window(aq_tensor_view,
                                            make_tuple(number<block_m>{}, number<aqk_per_block>{}),
                                            {i_m, 0});
                }
                else // Column major AQ
                {
                    return make_tile_window(aq_tensor_view,
                                            make_tuple(number<aqk_per_block>{}, number<block_m>{}),
                                            {0, i_m});
                }
            }
            else if constexpr(kQuantType == QuantType::ABQuantGrouped && !PreshuffleQuant)
            {
                static_assert(std::is_same_v<AQLayout, tensor_layout::gemm::RowMajor>);
                using QuantGroupSize   = remove_cvref_t<typename GemmPipeline::AQuantGroupSize>;
                constexpr auto block_m = TilePartitioner::MPerBlock;
                constexpr auto block_k = TilePartitioner::KPerBlock;
                return make_tile_window(
                    aq_tensor_view,
                    make_tuple(number<block_m>{}, number<block_k / QuantGroupSize::kK>{}),
                    {i_m, 0});
            }
            else if constexpr(kQuantType == QuantType::RowColQuant)
            {
                return make_tile_window(aq_tensor_view,
                                        make_tuple(number<TilePartitioner::MPerBlock>{},
                                                   number<TilePartitioner::NPerBlock>{}),
                                        {i_m, i_n});
            }
            else
            {
                return nullptr;
            }
        }();

        return aq_block_window;
    }

    CK_TILE_DEVICE static auto MakeBBlockWindow(const BDataType* b_ptr,
                                                const QuantGemmKernelArgs& kargs,
                                                const index_t k_size,
                                                const index_t i_n)
    {
        // Step 1: Create tensor view for B
        const auto& b_tensor_view = [&]() {
            if constexpr(std::is_same_v<BLayout, tensor_layout::gemm::RowMajor>)
            {
                if constexpr(GemmPipeline::BlockGemmShape::PermuteB)
                {
                    constexpr index_t K1          = GemmPipeline::GetSmemPackB();
                    const index_t K0              = k_size / K1;
                    constexpr index_t VectorSizeB = std::min(K1, GemmPipeline::GetVectorSizeB());
                    const auto b_k0_n_k1_desc =
                        make_naive_tensor_descriptor(make_tuple(K0, kargs.N, K1),
                                                     make_tuple(kargs.N * K1, K1, I1),
                                                     number<VectorSizeB>{},
                                                     number<1>{});
                    const auto b_n_k_desc = transform_tensor_descriptor(
                        b_k0_n_k1_desc,
                        make_tuple(make_merge_transform(make_tuple(K0, K1)),
                                   make_pass_through_transform(kargs.N)),
                        make_tuple(sequence<0, 2>{}, sequence<1>{}),
                        make_tuple(sequence<0>{}, sequence<1>{}));
                    return make_tensor_view<address_space_enum::global>(b_ptr, b_n_k_desc);
                }
                else
                {
                    return make_naive_tensor_view<address_space_enum::global>(
                        b_ptr,
                        make_tuple(k_size, kargs.N),
                        make_tuple(kargs.stride_B, 1),
                        number<GemmPipeline::GetVectorSizeB()>{},
                        number<1>{});
                }
            }
            else
            {
                if constexpr(GemmPipeline::BlockGemmShape::PermuteB)
                {
                    constexpr index_t K1          = GemmPipeline::GetSmemPackB();
                    const index_t K0              = k_size / K1;
                    constexpr index_t VectorSizeB = std::min(K1, GemmPipeline::GetVectorSizeB());
                    const auto b_k0_n_k1_desc =
                        make_naive_tensor_descriptor(make_tuple(K0, kargs.N, K1),
                                                     make_tuple(kargs.N * K1, K1, I1),
                                                     number<VectorSizeB>{},
                                                     number<1>{});
                    const auto b_n_k_desc = transform_tensor_descriptor(
                        b_k0_n_k1_desc,
                        make_tuple(make_merge_transform(make_tuple(K0, K1)),
                                   make_pass_through_transform(kargs.N)),
                        make_tuple(sequence<0, 2>{}, sequence<1>{}),
                        make_tuple(sequence<1>{}, sequence<0>{}));
                    return make_tensor_view<address_space_enum::global>(b_ptr, b_n_k_desc);
                }
                else
                {
                    if constexpr(PreshuffleB)
                    {
                        index_t kFlatK =
                            GemmPipeline::flatKPerWarp *
                            (k_size / GemmPipeline::BlockGemmShape::WarpTile::at(number<2>{}));
                        index_t kFlatN = kargs.N * kargs.K / kFlatK;
                        return make_naive_tensor_view<address_space_enum::global>(
                            b_ptr,
                            make_tuple(kFlatN, kFlatK),
                            make_tuple(kFlatK, 1),
                            number<GemmPipeline::GetVectorSizeB()>{},
                            number<1>{});
                    }
                    else
                    {
                        if constexpr(std::is_same_v<BDataType, pk_fp4_raw_t>)
                            return make_naive_tensor_view<address_space_enum::global>(
                                b_ptr,
                                make_tuple(kargs.N, k_size / 2),
                                make_tuple(kargs.stride_B, 1),
                                number<GemmPipeline::GetVectorSizeB()>{},
                                number<1>{});
                        else
                            return make_naive_tensor_view<address_space_enum::global>(
                                b_ptr,
                                make_tuple(kargs.N, k_size),
                                make_tuple(kargs.stride_B, 1),
                                number<GemmPipeline::GetVectorSizeB()>{},
                                number<1>{});
                    }
                }
            }
        }();

        // Step 2: Create padded view (or flat view for PreshuffleB)
        const auto& b_pad_view = [&]() {
            if constexpr(PreshuffleB)
            {
                return b_tensor_view; // no padding for preshuffle
            }
            else if constexpr(std::is_same_v<BLayout, tensor_layout::gemm::ColumnMajor>)
            {
                if constexpr(std::is_same_v<BDataType, pk_fp4_raw_t>)
                    return pad_tensor_view(b_tensor_view,
                                           make_tuple(number<TilePartitioner::NPerBlock>{},
                                                      number<TilePartitioner::KPerBlock / 2>{}),
                                           sequence<false, GemmPipeline::kPadK>{});
                else
                    return pad_tensor_view(b_tensor_view,
                                           make_tuple(number<TilePartitioner::NPerBlock>{},
                                                      number<TilePartitioner::KPerBlock>{}),
                                           sequence<false, GemmPipeline::kPadK>{});
            }
            else
            {
                return pad_tensor_view(b_tensor_view,
                                       make_tuple(number<TilePartitioner::KPerBlock>{},
                                                  number<TilePartitioner::NPerBlock>{}),
                                       sequence<false, GemmPipeline::kPadN>{});
            }
        }();

        // Step 3: Create tile window
        const auto& b_block_window = [&]() {
            if constexpr(PreshuffleB)
            {
                return make_tile_window(
                    b_pad_view,
                    make_tuple(number<GemmPipeline::flatNPerWarp>{},
                               number<GemmPipeline::flatKPerWarp>{}),
                    {static_cast<int>(i_n / GemmPipeline::BlockGemmShape::WarpTile::at(I1)), 0});
            }
            else
            {
                if constexpr(std::is_same_v<BLayout, tensor_layout::gemm::ColumnMajor>)
                {
                    if constexpr(std::is_same_v<BDataType, pk_fp4_raw_t>)
                        return make_tile_window(
                            b_pad_view,
                            make_tuple(number<TilePartitioner::NPerBlock>{},
                                       number<TilePartitioner::KPerBlock / 2>{}),
                            {i_n, 0});
                    else
                        return make_tile_window(b_pad_view,
                                                make_tuple(number<TilePartitioner::NPerBlock>{},
                                                           number<TilePartitioner::KPerBlock>{}),
                                                {i_n, 0});
                }
                else
                {
                    return make_tile_window(b_pad_view,
                                            make_tuple(number<TilePartitioner::KPerBlock>{},
                                                       number<TilePartitioner::NPerBlock>{}),
                                            {0, i_n});
                }
            }
        }();

        return b_block_window;
    }

    CK_TILE_DEVICE static auto MakeBQBlockWindow(const BQDataType* bq_ptr,
                                                 const QuantGemmKernelArgs& kargs,
                                                 const index_t i_m,
                                                 const index_t i_n)
    {
        // Step 1: Create tensor view for BQ
        const auto& bq_tensor_view = [&]() {
            if constexpr(kQuantType == QuantType::RowColQuant)
            {
                return make_naive_tensor_view<address_space_enum::global>(
                    bq_ptr,
                    make_tuple(kargs.M, kargs.N),
                    make_tuple(0, 1), // broadcasting over m
                    number<1>{},
                    number<1>{});
            }
            else if constexpr(kQuantType == QuantType::BQuantGrouped)
            {
                if constexpr(PreshuffleQuant)
                {
                    static_assert(std::is_same_v<BQLayout, tensor_layout::gemm::ColumnMajor>,
                                  "PreshuffleQuant with BQuantGrouped currently only supports "
                                  "ColumnMajor BQ layout");
                    using QuantGroupSize = remove_cvref_t<typename GemmPipeline::QuantGroupSize>;

                    return MakePreshuffledQuantTensorView<
                        GemmPipeline::KPerBlockBQ,
                        GemmPipeline::NPerBlockBQ,
                        GemmPipeline::NPerBlock,
                        TilePartitioner::BlockGemmShape::WarpTile::at(I1),
                        GemmPipeline::GetVectorSizeBQ()>(
                        bq_ptr,
                        ck_tile::integer_divide_ceil(kargs.N, QuantGroupSize::kN),
                        QuantGroupSize::kN,
                        kargs.QK_B);
                }
                else
                {
                    using QuantGroupSize = remove_cvref_t<typename GemmPipeline::QuantGroupSize>;

                    if constexpr(std::is_same_v<BQLayout, tensor_layout::gemm::RowMajor>)
                    {
                        return make_naive_tensor_view<address_space_enum::global>(
                            bq_ptr,
                            make_tuple(integer_divide_ceil(kargs.K, QuantGroupSize::kK),
                                       integer_divide_ceil(kargs.N, QuantGroupSize::kN)),
                            make_tuple(integer_divide_ceil(kargs.N, QuantGroupSize::kN), 1),
                            number<GemmPipeline::GetVectorSizeBQ()>{},
                            number<1>{});
                    }
                    else
                    {
                        static_assert(std::is_same_v<BQLayout, tensor_layout::gemm::ColumnMajor>);
                        return make_naive_tensor_view<address_space_enum::global>(
                            bq_ptr,
                            make_tuple(integer_divide_ceil(kargs.N, QuantGroupSize::kN),
                                       integer_divide_ceil(kargs.K, QuantGroupSize::kK)),
                            make_tuple(integer_divide_ceil(kargs.K, QuantGroupSize::kK), 1),
                            number<GemmPipeline::GetVectorSizeBQ()>{},
                            number<1>{});
                    }
                }
            }
            else if constexpr(kQuantType == QuantType::ABQuantGrouped)
            {
                static_assert(std::is_same_v<BQLayout, tensor_layout::gemm::ColumnMajor>);
                using QuantGroupSize = remove_cvref_t<typename GemmPipeline::BQuantGroupSize>;
                return make_naive_tensor_view<address_space_enum::global>(
                    bq_ptr,
                    make_tuple(integer_divide_ceil(kargs.N, QuantGroupSize::kN), kargs.QK_B),
                    make_tuple(kargs.stride_BQ, 1),
                    number<GemmPipeline::GetVectorSizeBQ()>{},
                    number<1>{});
            }
            else
            {
                return nullptr;
            }
        }();

        // Step 2: Create tile window (no padding for BQ)
        const auto& bq_block_window = [&]() {
            if constexpr(kQuantType == QuantType::RowColQuant)
            {
                return make_tile_window(bq_tensor_view,
                                        make_tuple(number<TilePartitioner::MPerBlock>{},
                                                   number<TilePartitioner::NPerBlock>{}),
                                        {i_m, i_n});
            }
            else if constexpr(kQuantType == QuantType::BQuantGrouped)
            {
                using QuantGroupSize = remove_cvref_t<typename GemmPipeline::QuantGroupSize>;
                if constexpr(PreshuffleQuant)
                {
                    static_assert(std::is_same_v<BQLayout, tensor_layout::gemm::ColumnMajor>);
                    constexpr auto block_n =
                        TilePartitioner::NPerBlock /
                        QuantGroupSize::kN; // Number of N-dimension quantization groups per block
                    constexpr auto warp_n = TilePartitioner::BlockGemmShape::WarpTile::at(
                        I1); // Number of N-dimension elements per warp
                    constexpr auto warp_per_group =
                        (QuantGroupSize::kN <
                         warp_n) // Determine how many warps share the same scale in N-dimension
                            ? (warp_n / QuantGroupSize::kN)
                            : (QuantGroupSize::kN / warp_n);
                    constexpr auto bqk_per_block =
                        TilePartitioner::KPerBlock /
                        QuantGroupSize::kK; // Number of K-dimension quantization groups per block
                    constexpr auto
                        tile_window_width = // The pre-shuffled layout flattens warp_n ×
                                            // bqk_per_block scales per row, Padded up to warp_size
                                            // to ensure coalesced memory access.
                        ck_tile::integer_least_multiple(warp_n * bqk_per_block, get_warp_size());

                    // Adapts based on fine vs coarse quantization granularity:
                    //   - Fine-grained (QuantGroupSize::kN < warp_n):
                    //       Multiple quant groups per warp → fewer rows needed per block.
                    //       height = block_n / warp_per_group
                    //
                    //   - Coarse-grained (QuantGroupSize::kN >= warp_n):
                    //       Each row represents one quant group.
                    //       height = block_n
                    constexpr auto tile_window_height =
                        (QuantGroupSize::kN < warp_n) ? block_n / warp_per_group : block_n;
                    auto block_n_idx =
                        i_n / TilePartitioner::NPerBlock; // Converts the global N-index (i_n) to a
                                                          // block index.

                    return make_tile_window(
                        bq_tensor_view,
                        make_tuple(number<tile_window_height>{}, number<tile_window_width>{}),
                        {block_n_idx * tile_window_height, 0});
                }
                else
                {
                    if constexpr(std::is_same_v<BQLayout, tensor_layout::gemm::RowMajor>)
                    {
                        return make_tile_window(
                            bq_tensor_view,
                            make_tuple(number<TilePartitioner::KPerBlock / QuantGroupSize::kK>{},
                                       number<TilePartitioner::NPerBlock / QuantGroupSize::kN>{}),
                            {0, i_n / QuantGroupSize::kN});
                    }
                    else
                    {
                        static_assert(std::is_same_v<BQLayout, tensor_layout::gemm::ColumnMajor>);
                        return make_tile_window(
                            bq_tensor_view,
                            make_tuple(number<TilePartitioner::NPerBlock / QuantGroupSize::kN>{},
                                       number<TilePartitioner::KPerBlock / QuantGroupSize::kK>{}),
                            {i_n / QuantGroupSize::kN, 0});
                    }
                }
            }
            else if constexpr(kQuantType == QuantType::ABQuantGrouped)
            {
                static_assert(std::is_same_v<BQLayout, tensor_layout::gemm::ColumnMajor>);
                using QuantGroupSize = remove_cvref_t<typename GemmPipeline::BQuantGroupSize>;
                return make_tile_window(
                    bq_tensor_view,
                    make_tuple(number<TilePartitioner::NPerBlock / QuantGroupSize::kN>{},
                               number<TilePartitioner::KPerBlock / QuantGroupSize::kK>{}),
                    {i_n / QuantGroupSize::kN, 0});
            }
            else
            {
                return nullptr;
            }
        }();

        return bq_block_window;
    }

    template <memory_operation_enum DstInMemOp = memory_operation_enum::set>
    CK_TILE_DEVICE static auto MakeCBlockWindow(CDataType* c_ptr,
                                                const QuantGemmKernelArgs& kargs,
                                                const index_t i_m,
                                                const index_t i_n)
    {
        // Step 1: Create tensor view for C
        const auto& c_tensor_view = [&]() {
            if constexpr(std::is_same_v<CLayout, tensor_layout::gemm::RowMajor>)
            {
                return make_naive_tensor_view<address_space_enum::global, DstInMemOp>(
                    c_ptr,
                    make_tuple(kargs.M, kargs.N),
                    make_tuple(kargs.stride_C, 1),
                    number<EpiloguePipeline::GetVectorSizeC()>{},
                    number<1>{});
            }
            else
            {
                return make_naive_tensor_view<address_space_enum::global, DstInMemOp>(
                    c_ptr,
                    make_tuple(kargs.M, kargs.N),
                    make_tuple(1, kargs.stride_C),
                    number<1>{},
                    number<1>{});
            }
        }();

        // Step 2: Create padded view
        const auto& c_pad_view = [&]() {
            if constexpr(std::is_same_v<CLayout, tensor_layout::gemm::RowMajor>)
            {
                return pad_tensor_view(c_tensor_view,
                                       make_tuple(number<TilePartitioner::MPerBlock>{},
                                                  number<TilePartitioner::NPerBlock>{}),
                                       sequence<false, GemmPipeline::kPadN>{});
            }
            else
            {
                return pad_tensor_view(c_tensor_view,
                                       make_tuple(number<TilePartitioner::MPerBlock>{},
                                                  number<TilePartitioner::NPerBlock>{}),
                                       sequence<GemmPipeline::kPadM, false>{});
            }
        }();

        // Step 3: Create tile window
        auto c_block_window = make_tile_window(
            c_pad_view,
            make_tuple(number<TilePartitioner::MPerBlock>{}, number<TilePartitioner::NPerBlock>{}),
            {i_m, i_n});

        return c_block_window;
    }

    CK_TILE_HOST static bool IsSupportedArgument(const QuantGemmKernelArgs& kargs)
    {
        if(kargs.k_batch != 1)
        {
            if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
            {
                CK_TILE_ERROR("Conditions not met for Kbatch >1 !");
            }
            return false;
        }

        if constexpr(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>)
        {
            if(kargs.K % (TilePartitioner::KPerBlock * kargs.k_batch) != 0 &&
               GemmPipeline::kPadK == false)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR("Can't support K that is not a multiple of k_batch * KPerBlock "
                                  "without padding!");
                }
                return false;
            }
            if(kargs.K % GemmPipeline::GetVectorSizeA() != 0)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR("K is not a multiple of vector load size for A tensor!");
                }
                return false;
            }
        }
        else
        {
            if(kargs.M % TilePartitioner::MPerBlock != 0 && GemmPipeline::kPadM == false)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR(
                        "Can't support M that is not a multiple of MPerBlock without padding!");
                }
                return false;
            }
            if(kargs.M % GemmPipeline::GetVectorSizeA() != 0)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR("M is not a multiple of vector load size for A tensor!");
                }
                return false;
            }
        }

        if constexpr(std::is_same_v<BLayout, tensor_layout::gemm::RowMajor>)
        {
            if(kargs.N % TilePartitioner::NPerBlock != 0 && GemmPipeline::kPadN == false)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR(
                        "Can't support N that is not a multiple of NPerBlock without padding!");
                }
                return false;
            }
            if(kargs.N % GemmPipeline::GetVectorSizeB() != 0)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR("N is not a multiple of vector load size for B tensor!");
                }
                return false;
            }
        }
        else
        {
            if(kargs.K % (TilePartitioner::KPerBlock * kargs.k_batch) != 0 &&
               GemmPipeline::kPadK == false)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR("Can't support K that is not a multiple of k_batch * KPerBlock "
                                  "without padding!");
                }
                return false;
            }
            if(kargs.K % GemmPipeline::GetVectorSizeB() != 0)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR("K is not a multiple of vector load size for B tensor!");
                }
                return false;
            }
        }

        if constexpr(std::is_same_v<CLayout, tensor_layout::gemm::RowMajor>)
        {
            if(kargs.N % TilePartitioner::NPerBlock != 0 && GemmPipeline::kPadN == false)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR(
                        "Can't support N that is not a multiple of NPerBlock without padding!");
                }
                return false;
            }
            if(kargs.N % EpiloguePipeline::GetVectorSizeC() != 0)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR("N is not a multiple of vector load size for C tensor!");
                }
                return false;
            }
        }
        else
        {
            if(kargs.M % TilePartitioner::MPerBlock != 0 && GemmPipeline::kPadM == false)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR(
                        "Can't support M that is not a multiple of MPerBlock without padding!");
                }
                return false;
            }
            if(kargs.M % EpiloguePipeline::GetVectorSizeC() != 0)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR("M is not a multiple of vector load size for C tensor!");
                }
                return false;
            }
        }
        return true;
    }

    /**
     * @brief Runs single GEMM problem cooperatively by whole workgroup.
     *
     * @param a_ptr input A pointer
     * @param b_ptr input B pointer
     * @param aq_ptr input AQ pointer
     * @param bq_ptr input BQ pointer
     * @param c_ptr output C pointer
     * @param smem_ptr The start memory pointer of the shared memory block.
     * @param kargs GEMM kernel arguments
     * @param splitk_batch_offset splitk_batch_offset Utility structure used to calculate k batch.
     * @param block_idx_m The GEMM's output M dimension tile index processed by this workgroup.
     * @param block_idx_n The GEMM's output N dimension tile index processed by this workgroup.
     *
     */
    CK_TILE_DEVICE static void RunGemm(const ADataType* a_ptr,
                                       const BDataType* b_ptr,
                                       const AQDataType* aq_ptr,
                                       const BQDataType* bq_ptr,
                                       CDataType* c_ptr,
                                       void* smem_ptr,
                                       const QuantGemmKernelArgs& kargs,
                                       const SplitKBatchOffset& splitk_batch_offset,
                                       const index_t block_idx_m,
                                       const index_t block_idx_n)
    {
        // Create block windows using specialized methods
        const auto& a_block_window =
            MakeABlockWindow(a_ptr, kargs, splitk_batch_offset.splitted_k, block_idx_m);
        const auto& b_block_window =
            MakeBBlockWindow(b_ptr, kargs, splitk_batch_offset.splitted_k, block_idx_n);
        const auto& aq_block_window = MakeAQBlockWindow(aq_ptr, kargs, block_idx_m, block_idx_n);
        const auto& bq_block_window = MakeBQBlockWindow(bq_ptr, kargs, block_idx_m, block_idx_n);

        const index_t num_loop =
            amd_wave_read_first_lane(TilePartitioner::GetLoopNum(splitk_batch_offset.splitted_k));

        // Run GEMM cooperatively by whole workgroup.
        const auto& c_block_tile = [&]() {
            if constexpr(kQuantType == QuantType::AQuantGrouped)
            {
                index_t m = 0;
                if constexpr(PreshuffleQuant)
                {
                    m = kargs.M;
                }
                return GemmPipeline{}.template operator()(
                    a_block_window, b_block_window, aq_block_window, num_loop, smem_ptr, m);
            }
            else if constexpr(kQuantType == QuantType::BQuantGrouped)
            {
                index_t n = 0;
                if constexpr(PreshuffleQuant)
                {
                    n = kargs.N;
                }
                return GemmPipeline{}.template operator()(
                    a_block_window, b_block_window, bq_block_window, num_loop, smem_ptr, n);
            }
            else if constexpr(kQuantType == QuantType::ABQuantGrouped)
            {
                index_t m = 0;
                index_t n = 0;
                if constexpr(PreshuffleQuant)
                {
                    m = kargs.M;
                    n = kargs.N;
                }
                return GemmPipeline{}.template operator()(a_block_window,
                                                          b_block_window,
                                                          aq_block_window,
                                                          bq_block_window,
                                                          num_loop,
                                                          smem_ptr,
                                                          m,
                                                          n);
            }
            else if constexpr(kQuantType == QuantType::RowColQuant ||
                              kQuantType == QuantType::TensorQuant)
            {
                return GemmPipeline{}.template operator()(
                    a_block_window, b_block_window, num_loop, smem_ptr);
            }
        }();

        const index_t k_batch = amd_wave_read_first_lane(kargs.k_batch);

        // Run Epilogue Pipeline with k_batch dispatch
        if(k_batch == 1)
        {
            auto c_block_window = MakeCBlockWindow<memory_operation_enum::set>(
                c_ptr, kargs, block_idx_m, block_idx_n);

            if constexpr(kQuantType == QuantType::ABQuantGrouped ||
                         kQuantType == QuantType::AQuantGrouped ||
                         kQuantType == QuantType::BQuantGrouped)
            {
                EpiloguePipeline{}(c_block_window, c_block_tile, c_block_window, smem_ptr);
            }
            else if constexpr(kQuantType == QuantType::RowColQuant)
            {
                EpiloguePipeline{}(c_block_window,
                                   c_block_tile,
                                   c_block_window,
                                   smem_ptr,
                                   aq_block_window,
                                   bq_block_window);
            }
            else if constexpr(kQuantType == QuantType::TensorQuant)
            {
                const AccDataType aq_scale = type_convert<AccDataType>(*aq_ptr);
                const AccDataType bq_scale = type_convert<AccDataType>(*bq_ptr);
                EpiloguePipeline{}(
                    c_block_window, c_block_tile, c_block_window, smem_ptr, aq_scale, bq_scale);
            }
        }
        else
        {
            auto c_block_window = MakeCBlockWindow<memory_operation_enum::atomic_add>(
                c_ptr, kargs, block_idx_m, block_idx_n);

            if constexpr(kQuantType == QuantType::ABQuantGrouped ||
                         kQuantType == QuantType::AQuantGrouped ||
                         kQuantType == QuantType::BQuantGrouped)
            {
                EpiloguePipeline{}(c_block_window, c_block_tile, c_block_window, smem_ptr);
            }
            else if constexpr(kQuantType == QuantType::RowColQuant)
            {
                EpiloguePipeline{}(c_block_window,
                                   c_block_tile,
                                   c_block_window,
                                   smem_ptr,
                                   aq_block_window,
                                   bq_block_window);
            }
            else if constexpr(kQuantType == QuantType::TensorQuant)
            {
                const AccDataType aq_scale = type_convert<AccDataType>(*aq_ptr);
                const AccDataType bq_scale = type_convert<AccDataType>(*bq_ptr);
                EpiloguePipeline{}(
                    c_block_window, c_block_tile, c_block_window, smem_ptr, aq_scale, bq_scale);
            }
        }
    }

    CK_TILE_DEVICE void operator()(QuantGemmKernelArgs kargs) const
    {
        const auto blockId  = amd_wave_read_first_lane(blockIdx.x);
        const auto [iM, iN] = TilePartitioner{kargs.M, kargs.N}.GetOutputTileIndex(blockId);
        const index_t i_m   = amd_wave_read_first_lane(iM * TilePartitioner::MPerBlock);
        const index_t i_n   = amd_wave_read_first_lane(iN * TilePartitioner::NPerBlock);
        const SplitKBatchOffset splitk_batch_offset(kargs);

        // Apply splitk offset to input pointers
        const ADataType* a_ptr =
            static_cast<const ADataType*>(kargs.a_ptr) + splitk_batch_offset.a_k_split_offset;
        const BDataType* b_ptr =
            static_cast<const BDataType*>(kargs.b_ptr) + splitk_batch_offset.b_k_split_offset;
        const AQDataType* aq_ptr = static_cast<const AQDataType*>(kargs.aq_ptr);
        const BQDataType* bq_ptr = static_cast<const BQDataType*>(kargs.bq_ptr);
        CDataType* c_ptr         = static_cast<CDataType*>(kargs.c_ptr);

        // allocate LDS
        __shared__ char smem_ptr[GetSmemSize()];

        RunGemm(
            a_ptr, b_ptr, aq_ptr, bq_ptr, c_ptr, smem_ptr, kargs, splitk_batch_offset, i_m, i_n);
    }
};

} // namespace ck_tile
