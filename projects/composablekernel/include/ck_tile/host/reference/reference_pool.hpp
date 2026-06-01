// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"
#include "ck_tile/ops/pooling/kernel/pool_kernel.hpp"
#include <thread>
#include <cmath>

namespace ck_tile {

template <typename InDataType,
          typename ComputeDataType,
          typename OutDataType,
          typename IndexDataType,
          typename ReduceOp,
          typename TensorShape,
          typename WindowShape,
          bool OutputIndex = false>
CK_TILE_HOST void reference_pool2d(const HostTensor<InDataType>& input,
                                   HostTensor<OutDataType>& output,
                                   HostTensor<IndexDataType>& output_index,
                                   PoolKernelArgs<TensorShape, WindowShape> kargs,
                                   ReduceOp reduce_op)
{
    const ck_tile::index_t N = kargs.input_shape.at(ck_tile::number<0>{});
    const ck_tile::index_t H = kargs.input_shape.at(ck_tile::number<1>{});
    const ck_tile::index_t W = kargs.input_shape.at(ck_tile::number<2>{});
    const ck_tile::index_t C = kargs.input_shape.at(ck_tile::number<3>{});

    const ck_tile::index_t Ho = kargs.output_shape.at(ck_tile::number<1>{});
    const ck_tile::index_t Wo = kargs.output_shape.at(ck_tile::number<2>{});

    const ck_tile::index_t Y = kargs.window_lengths.at(ck_tile::number<0>{});
    const ck_tile::index_t X = kargs.window_lengths.at(ck_tile::number<1>{});

    const ck_tile::index_t Sy = kargs.window_strides.at(ck_tile::number<0>{});
    const ck_tile::index_t Sx = kargs.window_strides.at(ck_tile::number<1>{});

    const ck_tile::index_t Dy = kargs.window_dilations.at(ck_tile::number<0>{});
    const ck_tile::index_t Dx = kargs.window_dilations.at(ck_tile::number<1>{});

    const ck_tile::index_t LeftPy = kargs.input_left_pads.at(ck_tile::number<0>{});
    const ck_tile::index_t LeftPx = kargs.input_left_pads.at(ck_tile::number<1>{});
    // Right padding is handled implicitly by bounds checking

    auto f = [&](auto n, auto ho, auto wo, auto c) {
        ComputeDataType v_acc = reduce_op.template GetIdentityValue<ComputeDataType>();

        IndexDataType current_index = 0; // Declare outside if constexpr for efficiency

        for(ck_tile::index_t y = 0; y < Y; ++y)
        {
            // Calculate input height index with stride, dilation, and padding
            ck_tile::index_t hi = ho * Sy + y * Dy - LeftPy;

            for(ck_tile::index_t x = 0; x < X; ++x)
            {
                // Calculate input width index with stride, dilation, and padding
                ck_tile::index_t wi = wo * Sx + x * Dx - LeftPx;

                if(hi >= 0 && hi < H && wi >= 0 && wi < W)
                {
                    const ComputeDataType v_in = type_convert<ComputeDataType>(input(n, hi, wi, c));

                    if constexpr(OutputIndex)
                    {
                        IndexDataType flat_index = input.GetOffsetFromMultiIndex(n, hi, wi, c);
                        bool changed             = false;
                        v_acc                    = reduce_op(v_acc, v_in, changed);
                        if(changed)
                        {
                            current_index = flat_index;
                        }
                    }
                    else
                    {
                        v_acc = reduce_op(v_acc, v_in);
                    }
                }
                // For positions outside bounds, we implicitly use identity value
            }
        }

        output(n, ho, wo, c) = ck_tile::type_convert<OutDataType>(v_acc);

        if constexpr(OutputIndex)
        {
            output_index(n, ho, wo, c) = current_index;
        }
    };

    // Parallelize over all output dimensions
    make_ParallelTensorFunctor(f, N, Ho, Wo, C)(std::thread::hardware_concurrency());
}

template <typename InDataType,
          typename ComputeDataType,
          typename OutDataType,
          typename IndexDataType,
          typename ReduceOp,
          typename TensorShape,
          typename WindowShape,
          bool OutputIndex = false>
CK_TILE_HOST void reference_pool3d(const HostTensor<InDataType>& input,
                                   HostTensor<OutDataType>& output,
                                   HostTensor<IndexDataType>& output_index,
                                   PoolKernelArgs<TensorShape, WindowShape> kargs,
                                   ReduceOp reduce_op)
{
    const ck_tile::index_t N = kargs.input_shape.at(ck_tile::number<0>{});
    const ck_tile::index_t D = kargs.input_shape.at(ck_tile::number<1>{});
    const ck_tile::index_t H = kargs.input_shape.at(ck_tile::number<2>{});
    const ck_tile::index_t W = kargs.input_shape.at(ck_tile::number<3>{});
    const ck_tile::index_t C = kargs.input_shape.at(ck_tile::number<4>{});

    const ck_tile::index_t Do = kargs.output_shape.at(ck_tile::number<1>{});
    const ck_tile::index_t Ho = kargs.output_shape.at(ck_tile::number<2>{});
    const ck_tile::index_t Wo = kargs.output_shape.at(ck_tile::number<3>{});

    const ck_tile::index_t Z = kargs.window_lengths.at(ck_tile::number<0>{});
    const ck_tile::index_t Y = kargs.window_lengths.at(ck_tile::number<1>{});
    const ck_tile::index_t X = kargs.window_lengths.at(ck_tile::number<2>{});

    const ck_tile::index_t Sz = kargs.window_strides.at(ck_tile::number<0>{});
    const ck_tile::index_t Sy = kargs.window_strides.at(ck_tile::number<1>{});
    const ck_tile::index_t Sx = kargs.window_strides.at(ck_tile::number<2>{});

    const ck_tile::index_t Dz = kargs.window_dilations.at(ck_tile::number<0>{});
    const ck_tile::index_t Dy = kargs.window_dilations.at(ck_tile::number<1>{});
    const ck_tile::index_t Dx = kargs.window_dilations.at(ck_tile::number<2>{});

    const ck_tile::index_t LeftPz = kargs.input_left_pads.at(ck_tile::number<0>{});
    const ck_tile::index_t LeftPy = kargs.input_left_pads.at(ck_tile::number<1>{});
    const ck_tile::index_t LeftPx = kargs.input_left_pads.at(ck_tile::number<2>{});
    // Right padding is handled implicitly by bounds checking

    auto f = [&](auto n, auto do_, auto ho, auto wo, auto c) {
        ComputeDataType v_acc = reduce_op.template GetIdentityValue<ComputeDataType>();

        IndexDataType current_index = 0; // Declare outside if constexpr for efficiency

        for(ck_tile::index_t z = 0; z < Z; ++z)
        {
            // Calculate input depth index with stride, dilation, and padding
            ck_tile::index_t di = do_ * Sz + z * Dz - LeftPz;

            for(ck_tile::index_t y = 0; y < Y; ++y)
            {
                // Calculate input height index with stride, dilation, and padding
                ck_tile::index_t hi = ho * Sy + y * Dy - LeftPy;

                for(ck_tile::index_t x = 0; x < X; ++x)
                {
                    // Calculate input width index with stride, dilation, and padding
                    ck_tile::index_t wi = wo * Sx + x * Dx - LeftPx;

                    if(di >= 0 && di < D && hi >= 0 && hi < H && wi >= 0 && wi < W)
                    {
                        const ComputeDataType v_in =
                            type_convert<ComputeDataType>(input(n, di, hi, wi, c));

                        if constexpr(OutputIndex)
                        {
                            IndexDataType flat_index =
                                input.GetOffsetFromMultiIndex(n, di, hi, wi, c);
                            bool changed = false;
                            v_acc        = reduce_op(v_acc, v_in, changed);
                            if(changed)
                            {
                                current_index = flat_index;
                            }
                        }
                        else
                        {
                            v_acc = reduce_op(v_acc, v_in);
                        }
                    }
                    // For positions outside bounds, we implicitly use identity value
                }
            }
        }

        output(n, do_, ho, wo, c) = ck_tile::type_convert<OutDataType>(v_acc);

        if constexpr(OutputIndex)
        {

            output_index(n, do_, ho, wo, c) = current_index;
        }
    };

    // Parallelize over all output dimensions
    make_ParallelTensorFunctor(f, N, Do, Ho, Wo, C)(std::thread::hardware_concurrency());
}
} // namespace ck_tile
