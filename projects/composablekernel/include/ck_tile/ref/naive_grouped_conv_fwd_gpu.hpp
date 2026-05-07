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

// Naive GPU reference kernel struct for forward grouped convolution
// Layout: Input=NDHWGC, Weight=GKZYXC, Output=NDHWGK (for 3D case)
//         Input=NHWGC,  Weight=GKYXC,  Output=NHWGK  (for 2D case)
//         Input=NWGC,   Weight=GKXC,   Output=NWGK   (for 1D case)
//
// One thread per output element, uses grid-stride loop pattern

template <ck_tile::index_t NDimSpatial,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType>
struct naive_grouped_conv_fwd_kernel
{
    static constexpr ck_tile::index_t kBlockSize = 256;

    __device__ void
    operator()(const InDataType* __restrict__ p_in,
               const WeiDataType* __restrict__ p_wei,
               OutDataType* __restrict__ p_out,
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

        // Calculate total output elements
        ck_tile::long_index_t output_length = G * N * K;
        for(ck_tile::index_t i = 0; i < NDimSpatial; ++i)
        {
            output_length *= out_spatial_lengths[i];
        }

        // Calculate strides for output tensor (NDHWGK or NHWGK or NWGK)
        std::array<ck_tile::long_index_t, NDimSpatial + 3> out_strides; // N, spatial dims, G, K
        ck_tile::long_index_t stride = 1;
        out_strides[NDimSpatial + 2] = stride; // K stride
        stride *= K;
        out_strides[NDimSpatial + 1] = stride; // G stride
        stride *= G;
        for(ck_tile::index_t i = NDimSpatial - 1; i >= 0; --i) // Spatial strides (reversed)
        {
            out_strides[i + 1] = stride;
            stride *= out_spatial_lengths[i];
        }
        out_strides[0] = stride; // N stride

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

        // Calculate strides for weight tensor (GKZYXC or GKYXC or GKXC)
        std::array<ck_tile::long_index_t, NDimSpatial + 3> wei_strides;
        stride                       = 1;
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

        // Grid-stride loop over all output elements
        for(ck_tile::long_index_t ii = tid; ii < output_length; ii += num_threads)
        {
            // Decode linear index to multi-dimensional indices
            ck_tile::long_index_t tmp = ii;

            // Extract N (batch)
            ck_tile::index_t n = tmp / out_strides[0];
            tmp -= n * out_strides[0];

            // Extract spatial dimensions (D, H, W)
            ck_tile::index_t out_spatial_idx[6]; // Max 6 spatial dimensions
            for(ck_tile::index_t i = 0; i < NDimSpatial; ++i)
            {
                out_spatial_idx[i] = tmp / out_strides[i + 1];
                tmp -= out_spatial_idx[i] * out_strides[i + 1];
            }

            // Extract G (group)
            ck_tile::index_t g = tmp / out_strides[NDimSpatial + 1];
            tmp -= g * out_strides[NDimSpatial + 1];

            // Extract K (output channel)
            ck_tile::index_t k = tmp;

            // Accumulate in float
            float v_acc = 0.0f;

            // Loop over input channels
            for(ck_tile::index_t c = 0; c < C; ++c)
            {
                // Loop over filter spatial dimensions
                if constexpr(NDimSpatial == 1)
                {
                    for(ck_tile::index_t x = 0; x < wei_spatial_lengths[0]; ++x)
                    {
                        // Calculate input spatial coordinate
                        ck_tile::long_index_t wi =
                            static_cast<ck_tile::long_index_t>(out_spatial_idx[0] *
                                                               conv_strides[0]) +
                            static_cast<ck_tile::long_index_t>(x * conv_dilations[0]) -
                            static_cast<ck_tile::long_index_t>(in_left_pads[0]);

                        // Bounds check
                        if(wi >= 0 && wi < in_spatial_lengths[0])
                        {
                            std::array<ck_tile::index_t, 1> in_spatial = {static_cast<index_t>(wi)};
                            std::array<ck_tile::index_t, 1> wei_spatial = {x};
                            ck_tile::long_index_t in_idx =
                                detail::calculate_input_index<1>(n, g, c, in_spatial, in_strides);
                            ck_tile::long_index_t wei_idx = detail::calculate_weight_index<1>(
                                g, k, c, wei_spatial, wei_strides);

                            v_acc += type_convert<float>(p_in[in_idx]) *
                                     type_convert<float>(p_wei[wei_idx]);
                        }
                    }
                }
                else if constexpr(NDimSpatial == 2)
                {
                    for(ck_tile::index_t y = 0; y < wei_spatial_lengths[0]; ++y)
                    {
                        ck_tile::long_index_t hi =
                            static_cast<ck_tile::long_index_t>(out_spatial_idx[0] *
                                                               conv_strides[0]) +
                            static_cast<ck_tile::long_index_t>(y * conv_dilations[0]) -
                            static_cast<ck_tile::long_index_t>(in_left_pads[0]);

                        for(ck_tile::index_t x = 0; x < wei_spatial_lengths[1]; ++x)
                        {
                            ck_tile::long_index_t wi =
                                static_cast<ck_tile::long_index_t>(out_spatial_idx[1] *
                                                                   conv_strides[1]) +
                                static_cast<ck_tile::long_index_t>(x * conv_dilations[1]) -
                                static_cast<ck_tile::long_index_t>(in_left_pads[1]);

                            // Bounds check
                            if(hi >= 0 && hi < in_spatial_lengths[0] && wi >= 0 &&
                               wi < in_spatial_lengths[1])
                            {
                                std::array<ck_tile::index_t, 2> in_spatial = {
                                    static_cast<index_t>(hi), static_cast<index_t>(wi)};
                                std::array<ck_tile::index_t, 2> wei_spatial = {y, x};
                                ck_tile::long_index_t in_idx = detail::calculate_input_index<2>(
                                    n, g, c, in_spatial, in_strides);
                                ck_tile::long_index_t wei_idx = detail::calculate_weight_index<2>(
                                    g, k, c, wei_spatial, wei_strides);

                                v_acc += type_convert<float>(p_in[in_idx]) *
                                         type_convert<float>(p_wei[wei_idx]);
                            }
                        }
                    }
                }
                else if constexpr(NDimSpatial == 3)
                {
                    for(ck_tile::index_t z = 0; z < wei_spatial_lengths[0]; ++z)
                    {
                        ck_tile::long_index_t di =
                            static_cast<ck_tile::long_index_t>(out_spatial_idx[0] *
                                                               conv_strides[0]) +
                            static_cast<ck_tile::long_index_t>(z * conv_dilations[0]) -
                            static_cast<ck_tile::long_index_t>(in_left_pads[0]);

                        for(ck_tile::index_t y = 0; y < wei_spatial_lengths[1]; ++y)
                        {
                            ck_tile::long_index_t hi =
                                static_cast<ck_tile::long_index_t>(out_spatial_idx[1] *
                                                                   conv_strides[1]) +
                                static_cast<ck_tile::long_index_t>(y * conv_dilations[1]) -
                                static_cast<ck_tile::long_index_t>(in_left_pads[1]);

                            for(ck_tile::index_t x = 0; x < wei_spatial_lengths[2]; ++x)
                            {
                                ck_tile::long_index_t wi =
                                    static_cast<ck_tile::long_index_t>(out_spatial_idx[2] *
                                                                       conv_strides[2]) +
                                    static_cast<ck_tile::long_index_t>(x * conv_dilations[2]) -
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
                                    std::array<ck_tile::index_t, 3> wei_spatial = {z, y, x};
                                    ck_tile::long_index_t in_idx = detail::calculate_input_index<3>(
                                        n, g, c, in_spatial, in_strides);
                                    ck_tile::long_index_t wei_idx =
                                        detail::calculate_weight_index<3>(
                                            g, k, c, wei_spatial, wei_strides);

                                    v_acc += type_convert<float>(p_in[in_idx]) *
                                             type_convert<float>(p_wei[wei_idx]);
                                }
                            }
                        }
                    }
                }
            }

            // Convert accumulator to output type and write
            p_out[ii] = type_convert<OutDataType>(v_acc);
        }
    }
};

// Host-side launcher for naive grouped convolution forward
template <ck_tile::index_t NDimSpatial,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType>
CK_TILE_HOST float naive_grouped_conv_fwd(const InDataType* p_in_dev,
                                          const WeiDataType* p_wei_dev,
                                          OutDataType* p_out_dev,
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
    // Convert vectors to arrays (std::array can be passed by value to kernel)
    auto in_spatial_arr     = to_array_with_default<NDimSpatial>(in_spatial_lengths);
    auto wei_spatial_arr    = to_array_with_default<NDimSpatial>(wei_spatial_lengths);
    auto out_spatial_arr    = to_array_with_default<NDimSpatial>(out_spatial_lengths);
    auto conv_strides_arr   = to_array_with_default<NDimSpatial>(conv_strides);
    auto conv_dilations_arr = to_array_with_default<NDimSpatial>(conv_dilations);
    auto in_left_pads_arr   = to_array_with_default<NDimSpatial>(in_left_pads, 0);

    // Calculate grid size
    ck_tile::long_index_t output_length = G * N * K;
    for(ck_tile::index_t i = 0; i < NDimSpatial; ++i)
    {
        output_length *= out_spatial_lengths[i];
    }

    using KernelType =
        naive_grouped_conv_fwd_kernel<NDimSpatial, InDataType, WeiDataType, OutDataType>;

    constexpr ck_tile::index_t block_size = KernelType::kBlockSize;
    const ck_tile::index_t grid_size      = (output_length + block_size - 1) / block_size;

    // Launch kernel
    float elapsed_ms = launch_kernel(stream_config,
                                     make_kernel(KernelType{},
                                                 dim3(grid_size),
                                                 dim3(block_size),
                                                 0, // dynamic shared memory size
                                                 p_in_dev,
                                                 p_wei_dev,
                                                 p_out_dev,
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
