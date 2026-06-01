// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ref/conv_common.hpp"
#include <array>
#include "ck_tile/host/device_memory.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include <hip/hip_runtime.h>

namespace ck_tile {

// Naive GPU reference kernel struct for backward weight grouped convolution
// Computes gradient with respect to weights
// Layout: Input=NDHWGC, Weight_grad=GKZYXC, Output_grad=NDHWGK (for 3D case)
//         Input=NHWGC,  Weight_grad=GKYXC,  Output_grad=NHWGK  (for 2D case)
//         Input=NWGC,   Weight_grad=GKXC,   Output_grad=NWGK   (for 1D case)
//
// One thread per weight element, uses grid-stride loop pattern

template <ck_tile::index_t NDimSpatial,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType>
struct naive_grouped_conv_bwd_weight_kernel
{
    static constexpr ck_tile::index_t kBlockSize = 256;

    __device__ void
    operator()(const InDataType* __restrict__ p_in,
               WeiDataType* __restrict__ p_wei_grad,
               const OutDataType* __restrict__ p_out_grad,
               // Tensor dimensions
               ck_tile::index_t G, // number of groups
               ck_tile::index_t N, // batch size
               ck_tile::index_t K, // output channels per group
               ck_tile::index_t C, // input channels per group
               // Input spatial dimensions
               const std::array<ck_tile::long_index_t, NDimSpatial>& in_spatial_lengths,
               // Weight spatial dimensions
               const std::array<ck_tile::long_index_t, NDimSpatial>& wei_spatial_lengths,
               // Output spatial dimensions
               const std::array<ck_tile::long_index_t, NDimSpatial>& out_spatial_lengths,
               // Convolution parameters
               const std::array<ck_tile::long_index_t, NDimSpatial>& conv_strides,
               const std::array<ck_tile::long_index_t, NDimSpatial>& conv_dilations,
               const std::array<ck_tile::long_index_t, NDimSpatial>& in_left_pads) const
    {
        const ck_tile::long_index_t tid         = get_block_id() * blockDim.x + get_thread_id();
        const ck_tile::long_index_t num_threads = blockDim.x * gridDim.x;

        // Calculate total weight elements
        ck_tile::long_index_t weight_length = G * K * C;
        for(ck_tile::index_t i = 0; i < NDimSpatial; ++i)
        {
            weight_length *= wei_spatial_lengths[i];
        }

        // Calculate strides for weight tensor (GKZYXC or GKYXC or GKXC)
        std::array<ck_tile::long_index_t, NDimSpatial + 3> wei_strides;
        ck_tile::long_index_t stride = 1;
        wei_strides[NDimSpatial + 2] = stride; // C stride
        stride *= C;
        for(ck_tile::index_t i = NDimSpatial - 1; i >= 0; --i)
        {
            wei_strides[i + 2] = stride;
            stride *= wei_spatial_lengths[i];
        }
        wei_strides[1] = stride; // K stride
        stride *= K;
        wei_strides[0] = stride; // G stride

        // Calculate strides for input tensor (NDHWGC or NHWGC or NWGC)
        std::array<ck_tile::long_index_t, NDimSpatial + 3> in_strides;
        stride                      = 1;
        in_strides[NDimSpatial + 2] = stride; // C stride
        stride *= C;
        in_strides[NDimSpatial + 1] = stride; // G stride
        stride *= G;
        for(ck_tile::index_t i = NDimSpatial - 1; i >= 0; --i)
        {
            in_strides[i + 1] = stride;
            stride *= in_spatial_lengths[i];
        }
        in_strides[0] = stride; // N stride

        // Calculate strides for output tensor (NDHWGK or NHWGK or NWGK)
        std::array<ck_tile::long_index_t, NDimSpatial + 3> out_strides;
        stride                       = 1;
        out_strides[NDimSpatial + 2] = stride; // K stride
        stride *= K;
        out_strides[NDimSpatial + 1] = stride; // G stride
        stride *= G;
        for(ck_tile::index_t i = NDimSpatial - 1; i >= 0; --i)
        {
            out_strides[i + 1] = stride;
            stride *= out_spatial_lengths[i];
        }
        out_strides[0] = stride; // N stride

        // Grid-stride loop over all weight elements
        for(ck_tile::long_index_t ii = tid; ii < weight_length; ii += num_threads)
        {
            // Decode linear index to multi-dimensional indices
            ck_tile::long_index_t tmp = ii;

            // Extract G (group)
            ck_tile::index_t g = tmp / wei_strides[0];
            tmp -= g * wei_strides[0];

            // Extract K (output channel)
            ck_tile::index_t k = tmp / wei_strides[1];
            tmp -= k * wei_strides[1];

            // Extract spatial dimensions (come before C in GKZYXC layout)
            ck_tile::index_t wei_spatial_idx[6];
            for(ck_tile::index_t i = 0; i < NDimSpatial; ++i)
            {
                wei_spatial_idx[i] = tmp / wei_strides[i + 2];
                tmp -= wei_spatial_idx[i] * wei_strides[i + 2];
            }

            // Extract C (input channel) - comes last
            ck_tile::index_t c = tmp;

            // Accumulate in float
            float v_acc = 0.0f;

            // Loop over batch
            for(ck_tile::index_t n = 0; n < N; ++n)
            {
                // Loop over output spatial dimensions
                if constexpr(NDimSpatial == 1)
                {
                    for(ck_tile::index_t wo = 0; wo < out_spatial_lengths[0]; ++wo)
                    {
                        // Calculate input spatial coordinate
                        ck_tile::long_index_t wi =
                            static_cast<ck_tile::long_index_t>(wo * conv_strides[0]) +
                            static_cast<ck_tile::long_index_t>(wei_spatial_idx[0] *
                                                               conv_dilations[0]) -
                            static_cast<ck_tile::long_index_t>(in_left_pads[0]);

                        // Bounds check
                        if(wi >= 0 && wi < in_spatial_lengths[0])
                        {
                            std::array<ck_tile::index_t, 1> in_spatial = {static_cast<index_t>(wi)};
                            std::array<ck_tile::index_t, 1> out_spatial = {
                                static_cast<index_t>(wo)};
                            ck_tile::long_index_t in_idx =
                                detail::calculate_input_index<1>(n, g, c, in_spatial, in_strides);
                            ck_tile::long_index_t out_idx = detail::calculate_output_index<1>(
                                n, g, k, out_spatial, out_strides);

                            v_acc += type_convert<float>(p_out_grad[out_idx]) *
                                     type_convert<float>(p_in[in_idx]);
                        }
                    }
                }
                else if constexpr(NDimSpatial == 2)
                {
                    for(ck_tile::index_t ho = 0; ho < out_spatial_lengths[0]; ++ho)
                    {
                        ck_tile::long_index_t hi =
                            static_cast<ck_tile::long_index_t>(ho * conv_strides[0]) +
                            static_cast<ck_tile::long_index_t>(wei_spatial_idx[0] *
                                                               conv_dilations[0]) -
                            static_cast<ck_tile::long_index_t>(in_left_pads[0]);

                        for(ck_tile::index_t wo = 0; wo < out_spatial_lengths[1]; ++wo)
                        {
                            ck_tile::long_index_t wi =
                                static_cast<ck_tile::long_index_t>(wo * conv_strides[1]) +
                                static_cast<ck_tile::long_index_t>(wei_spatial_idx[1] *
                                                                   conv_dilations[1]) -
                                static_cast<ck_tile::long_index_t>(in_left_pads[1]);

                            // Bounds check
                            if(hi >= 0 && hi < in_spatial_lengths[0] && wi >= 0 &&
                               wi < in_spatial_lengths[1])
                            {
                                std::array<ck_tile::index_t, 2> in_spatial = {
                                    static_cast<index_t>(hi), static_cast<index_t>(wi)};
                                std::array<ck_tile::index_t, 2> out_spatial = {
                                    static_cast<index_t>(ho), static_cast<index_t>(wo)};
                                ck_tile::long_index_t in_idx = detail::calculate_input_index<2>(
                                    n, g, c, in_spatial, in_strides);
                                ck_tile::long_index_t out_idx = detail::calculate_output_index<2>(
                                    n, g, k, out_spatial, out_strides);

                                v_acc += type_convert<float>(p_out_grad[out_idx]) *
                                         type_convert<float>(p_in[in_idx]);
                            }
                        }
                    }
                }
                else if constexpr(NDimSpatial == 3)
                {
                    for(ck_tile::index_t do_ = 0; do_ < out_spatial_lengths[0]; ++do_)
                    {
                        ck_tile::long_index_t di =
                            static_cast<ck_tile::long_index_t>(do_ * conv_strides[0]) +
                            static_cast<ck_tile::long_index_t>(wei_spatial_idx[0] *
                                                               conv_dilations[0]) -
                            static_cast<ck_tile::long_index_t>(in_left_pads[0]);

                        for(ck_tile::index_t ho = 0; ho < out_spatial_lengths[1]; ++ho)
                        {
                            ck_tile::long_index_t hi =
                                static_cast<ck_tile::long_index_t>(ho * conv_strides[1]) +
                                static_cast<ck_tile::long_index_t>(wei_spatial_idx[1] *
                                                                   conv_dilations[1]) -
                                static_cast<ck_tile::long_index_t>(in_left_pads[1]);

                            for(ck_tile::index_t wo = 0; wo < out_spatial_lengths[2]; ++wo)
                            {
                                ck_tile::long_index_t wi =
                                    static_cast<ck_tile::long_index_t>(wo * conv_strides[2]) +
                                    static_cast<ck_tile::long_index_t>(wei_spatial_idx[2] *
                                                                       conv_dilations[2]) -
                                    static_cast<ck_tile::long_index_t>(in_left_pads[2]);

                                // Bounds check
                                if(di >= 0 && di < in_spatial_lengths[0] && hi >= 0 &&
                                   hi < in_spatial_lengths[1] && wi >= 0 &&
                                   wi < in_spatial_lengths[2])
                                {
                                    std::array<ck_tile::index_t, 3> in_spatial = {
                                        static_cast<index_t>(di),
                                        static_cast<index_t>(hi),
                                        static_cast<index_t>(wi)};
                                    std::array<ck_tile::index_t, 3> out_spatial = {
                                        static_cast<index_t>(do_),
                                        static_cast<index_t>(ho),
                                        static_cast<index_t>(wo)};
                                    ck_tile::long_index_t in_idx = detail::calculate_input_index<3>(
                                        n, g, c, in_spatial, in_strides);
                                    ck_tile::long_index_t out_idx =
                                        detail::calculate_output_index<3>(
                                            n, g, k, out_spatial, out_strides);

                                    v_acc += type_convert<float>(p_out_grad[out_idx]) *
                                             type_convert<float>(p_in[in_idx]);
                                }
                            }
                        }
                    }
                }
            }

            // Convert accumulator to output type and write
            p_wei_grad[ii] = type_convert<WeiDataType>(v_acc);
        }
    }
};

// Host-side launcher for naive grouped convolution backward weight
template <ck_tile::index_t NDimSpatial,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType>
CK_TILE_HOST float
naive_grouped_conv_bwd_weight(const InDataType* p_in_dev,
                              WeiDataType* p_wei_grad_dev,
                              const OutDataType* p_out_grad_dev,
                              ck_tile::index_t G,
                              ck_tile::index_t N,
                              ck_tile::index_t K,
                              ck_tile::index_t C,
                              std::vector<ck_tile::long_index_t> in_spatial_lengths,
                              std::vector<ck_tile::long_index_t> wei_spatial_lengths,
                              std::vector<ck_tile::long_index_t> out_spatial_lengths,
                              std::vector<ck_tile::long_index_t> conv_strides,
                              std::vector<ck_tile::long_index_t> conv_dilations,
                              std::vector<ck_tile::long_index_t> in_left_pads,
                              ck_tile::stream_config stream_config = {})
{
    // Convert vectors to arrays
    auto in_spatial_arr     = to_array_with_default<NDimSpatial>(in_spatial_lengths);
    auto wei_spatial_arr    = to_array_with_default<NDimSpatial>(wei_spatial_lengths);
    auto out_spatial_arr    = to_array_with_default<NDimSpatial>(out_spatial_lengths);
    auto conv_strides_arr   = to_array_with_default<NDimSpatial>(conv_strides);
    auto conv_dilations_arr = to_array_with_default<NDimSpatial>(conv_dilations);
    auto in_left_pads_arr   = to_array_with_default<NDimSpatial>(in_left_pads, 0);

    // Calculate grid size
    ck_tile::long_index_t weight_length = G * K * C;
    for(ck_tile::index_t i = 0; i < NDimSpatial; ++i)
    {
        weight_length *= wei_spatial_lengths[i];
    }

    using KernelType =
        naive_grouped_conv_bwd_weight_kernel<NDimSpatial, InDataType, WeiDataType, OutDataType>;

    constexpr ck_tile::index_t block_size = KernelType::kBlockSize;
    const ck_tile::index_t grid_size      = (weight_length + block_size - 1) / block_size;

    // Launch kernel
    float elapsed_ms = launch_kernel(stream_config,
                                     make_kernel(KernelType{},
                                                 dim3(grid_size),
                                                 dim3(block_size),
                                                 0, // dynamic shared memory size
                                                 p_in_dev,
                                                 p_wei_grad_dev,
                                                 p_out_grad_dev,
                                                 G,
                                                 N,
                                                 K,
                                                 C,
                                                 in_spatial_arr,
                                                 wei_spatial_arr,
                                                 out_spatial_arr,
                                                 conv_strides_arr,
                                                 conv_dilations_arr,
                                                 in_left_pads_arr));

    return elapsed_ms;
}

} // namespace ck_tile
