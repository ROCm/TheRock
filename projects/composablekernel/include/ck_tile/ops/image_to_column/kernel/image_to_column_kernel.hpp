// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"

namespace ck_tile {

template <typename Problem_>
struct ImageToColumn
{
    static constexpr auto I0 = number<0>{};
    static constexpr auto I1 = number<1>{};
    static constexpr auto I2 = number<2>{};
    static constexpr auto I3 = number<3>{};
    static constexpr auto I4 = number<4>{};

    using Problem = remove_cvref_t<Problem_>;

    using InDataType  = remove_cvref_t<typename Problem::InDataType>;
    using OutDataType = remove_cvref_t<typename Problem::OutDataType>;

    static constexpr index_t NDimSpatial = Problem::NDimSpatial;

    static constexpr index_t AligmentIn  = Problem::AligmentIn;
    static constexpr index_t AligmentOut = Problem::AligmentOut;

    static_assert(NDimSpatial == 2, "Not supported.");

    static constexpr index_t kMPerBlock = Problem::BlockShape::kMPerBlock;
    static constexpr index_t kKPerBlock = Problem::BlockShape::kKPerBlock;
    static constexpr index_t kBlockSize = Problem::BlockShape::kBlockSize;

    struct Kargs
    {
        const void* p_in;
        void* p_out;

        const long_index_t G;
        const long_index_t N;
        const long_index_t C;

        const array<long_index_t, NDimSpatial> input_spatial_lengths;
        const array<long_index_t, NDimSpatial> filter_spatial_lengths;
        const array<long_index_t, NDimSpatial> output_spatial_lengths;
        const array<long_index_t, NDimSpatial + 3> image_g_n_c_wis_strides;
        const array<long_index_t, 3> gemm_g_m_k_strides;
        const array<long_index_t, NDimSpatial> conv_filter_strides;
        const array<long_index_t, NDimSpatial> conv_filter_dilations;
        const array<long_index_t, NDimSpatial> input_left_pads;
        const array<long_index_t, NDimSpatial> input_right_pads;
    };

    CK_TILE_HOST static constexpr Kargs
    MakeKargs(const void* p_in,
              void* p_out,
              const long_index_t G,
              const long_index_t N,
              const long_index_t C,
              const array<long_index_t, NDimSpatial> input_spatial_lengths,
              const array<long_index_t, NDimSpatial> filter_spatial_lengths,
              const array<long_index_t, NDimSpatial> output_spatial_lengths,
              const array<long_index_t, NDimSpatial + 3> image_g_n_c_wis_strides,
              const array<long_index_t, 3> gemm_g_m_k_strides,
              const array<long_index_t, NDimSpatial> conv_filter_strides,
              const array<long_index_t, NDimSpatial> conv_filter_dilations,
              const array<long_index_t, NDimSpatial> input_left_pads,
              const array<long_index_t, NDimSpatial> input_right_pads)
    {
        return Kargs{p_in,
                     p_out,
                     G,
                     N,
                     C,
                     input_spatial_lengths,
                     filter_spatial_lengths,
                     output_spatial_lengths,
                     image_g_n_c_wis_strides,
                     gemm_g_m_k_strides,
                     conv_filter_strides,
                     conv_filter_dilations,
                     input_left_pads,
                     input_right_pads};
    }

    CK_TILE_HOST static constexpr auto GridSize(index_t GemmM, index_t GemmK, index_t Batch)
    {
        return dim3(
            integer_divide_ceil(GemmM, kMPerBlock), integer_divide_ceil(GemmK, kKPerBlock), Batch);
    }

    CK_TILE_HOST static constexpr auto BlockSize() { return Problem::BlockShape::kBlockSize; }

    CK_TILE_DEVICE auto MakeImageMKDesc(const Kargs& kargs) const
    {
        static_assert(NDimSpatial == 2, "Not supported.");

        const auto in_n_hi_wi_c_desc = make_naive_tensor_descriptor(
            make_tuple(
                kargs.N, kargs.input_spatial_lengths[I0], kargs.input_spatial_lengths[I1], kargs.C),
            make_tuple(kargs.image_g_n_c_wis_strides[I1],
                       kargs.image_g_n_c_wis_strides[I3],
                       kargs.image_g_n_c_wis_strides[I4],
                       kargs.image_g_n_c_wis_strides[I2]),
            number<AligmentIn>{},
            I1);

        const auto in_n_hip_wip_c_desc = transform_tensor_descriptor(
            in_n_hi_wi_c_desc,
            make_tuple(make_pass_through_transform(kargs.N),
                       make_pad_transform(kargs.input_spatial_lengths[I0],
                                          kargs.input_left_pads[I0],
                                          kargs.input_right_pads[I0]),
                       make_pad_transform(kargs.input_spatial_lengths[I1],
                                          kargs.input_left_pads[I1],
                                          kargs.input_right_pads[I1]),
                       make_pass_through_transform(kargs.C)),
            make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}),
            make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}));

        const auto in_n_y_ho_x_wo_c_desc = transform_tensor_descriptor(
            in_n_hip_wip_c_desc,
            make_tuple(
                make_pass_through_transform(kargs.N),
                make_embed_transform(
                    make_tuple(kargs.filter_spatial_lengths[I0], kargs.output_spatial_lengths[I0]),
                    make_tuple(kargs.conv_filter_dilations[I0], kargs.conv_filter_strides[I0])),
                make_embed_transform(
                    make_tuple(kargs.filter_spatial_lengths[I1], kargs.output_spatial_lengths[I1]),
                    make_tuple(kargs.conv_filter_dilations[I1], kargs.conv_filter_strides[I1])),
                make_pass_through_transform(kargs.C)),
            make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}),
            make_tuple(sequence<0>{}, sequence<1, 2>{}, sequence<3, 4>{}, sequence<5>{}));

        return transform_tensor_descriptor(
            in_n_y_ho_x_wo_c_desc,
            make_tuple(
                make_merge_transform(make_tuple(
                    kargs.N, kargs.output_spatial_lengths[I0], kargs.output_spatial_lengths[I1])),
                make_merge_transform(make_tuple(
                    kargs.filter_spatial_lengths[I0], kargs.filter_spatial_lengths[I1], kargs.C))),
            make_tuple(sequence<0, 2, 4>{}, sequence<1, 3, 5>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));
    }

    CK_TILE_DEVICE auto CalculateMKDims(const Kargs& kargs) const
    {
        static_assert(NDimSpatial == 2, "Not supported.");
        const index_t M = kargs.N * static_cast<index_t>(kargs.output_spatial_lengths[I0] *
                                                         kargs.output_spatial_lengths[I1]);
        const index_t K = kargs.C * static_cast<index_t>(kargs.filter_spatial_lengths[I0] *
                                                         kargs.filter_spatial_lengths[I1]);
        return make_tuple(M, K);
    }

    CK_TILE_DEVICE static constexpr auto MakeBlockTileDistribution()
    {
        using P = typename Problem::BlockShape;
        // P: {kMWarpPerBlock * kKWarpPerBlock, kMThreadPerWarp * kKThreadPerWarp}
        // Y: {kMPerThread, kKPerThread}
        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<1>,
                tuple<sequence<P::kMWarpPerBlock, P::kMThreadPerWarp, P::kMPerThread>,
                      sequence<P::kKWarpPerBlock, P::kKThreadPerWarp, P::kKPerThread>>,
                tuple<sequence<1, 2>, sequence<1, 2>>,
                tuple<sequence<0, 0>, sequence<1, 1>>,
                sequence<1, 2>,
                sequence<2, 2>>{});
    }

    CK_TILE_DEVICE void ConvTensorRearrange(const Kargs& kargs) const
    {
        const auto [M, K] = CalculateMKDims(kargs);

        const index_t iM     = amd_wave_read_first_lane(blockIdx.x * kMPerBlock);
        const index_t iK     = amd_wave_read_first_lane(blockIdx.y * kKPerBlock);
        const index_t iBatch = amd_wave_read_first_lane(blockIdx.z);

        const auto in_offset  = iBatch * kargs.image_g_n_c_wis_strides[I0];
        const auto out_offset = iBatch * kargs.gemm_g_m_k_strides[I0];

        const auto image_m_k = make_tensor_view<address_space_enum::global>(
            static_cast<const InDataType*>(kargs.p_in) + in_offset, MakeImageMKDesc(kargs));
        const auto gemm_m_k = make_naive_tensor_view<address_space_enum::global>(
            static_cast<OutDataType*>(kargs.p_out) + out_offset,
            make_tuple(M, K),
            make_tuple(kargs.gemm_g_m_k_strides[I1], kargs.gemm_g_m_k_strides[I2]),
            number<AligmentOut>{},
            I1);

        const auto image_m_k_padded =
            pad_tensor_view(image_m_k,
                            make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}),
                            sequence<false, true>{});
        const auto gemm_m_k_padded =
            pad_tensor_view(gemm_m_k,
                            make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}),
                            sequence<false, true>{});

        constexpr auto dstr = MakeBlockTileDistribution();

        const auto image_tile =
            make_tile_window(image_m_k_padded,
                             make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}),
                             {iM, iK},
                             dstr);

        auto gemm_tile = make_tile_window(gemm_m_k_padded,
                                          make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}),
                                          {iM, iK},
                                          dstr);

        // load from Global
        const auto loaded_tile = load_tile(image_tile);
        // save to Global
        store_tile(gemm_tile, loaded_tile);
    }

    CK_TILE_DEVICE void operator()(Kargs& kargs) const { ConvTensorRearrange(kargs); }
};

} // namespace ck_tile
