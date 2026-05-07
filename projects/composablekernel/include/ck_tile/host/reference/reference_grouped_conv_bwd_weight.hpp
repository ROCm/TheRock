// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdlib>
#include <thread>

#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"

namespace ck_tile {

template <ck_tile::index_t NDimSpatial,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType>
CK_TILE_HOST void
reference_grouped_conv_bwd_weight(const HostTensor<InDataType>& input,
                                  HostTensor<WeiDataType>& weight,
                                  const HostTensor<OutDataType>& output,
                                  std::vector<ck_tile::long_index_t> conv_strides,
                                  std::vector<ck_tile::long_index_t> conv_dilations,
                                  std::vector<ck_tile::long_index_t> in_left_pads,
                                  std::vector<ck_tile::long_index_t>)
{
    if(!(input.get_num_of_dimension() == NDimSpatial + 3 &&
         weight.get_num_of_dimension() == NDimSpatial + 3 &&
         output.get_num_of_dimension() == NDimSpatial + 3))
    {
        throw std::runtime_error("wrong! inconsistent dimension");
    }

    if constexpr(NDimSpatial == 1)
    {
        auto func = [&](auto g, auto k, auto c, auto x) {
            float v_acc = 0;

            for(std::size_t n = 0; n < output.get_lengths()[1]; ++n)
            {
                for(std::size_t wo = 0; wo < output.get_lengths()[3]; ++wo)
                {
                    auto wi = static_cast<ck_tile::long_index_t>(wo * conv_strides[0]) +
                              static_cast<ck_tile::long_index_t>(x * conv_dilations[0]) -
                              static_cast<ck_tile::long_index_t>(in_left_pads[0]);

                    if(wi >= 0 && ck_tile::type_convert<std::size_t>(wi) < input.get_lengths()[3])
                    {
                        InDataType v_in   = input(g, n, c, wi);
                        OutDataType v_out = output(g, n, k, wo);
                        v_acc += ck_tile::type_convert<float>(v_out) *
                                 ck_tile::type_convert<float>(v_in);
                    }
                }
            }
            OutDataType v_acc_converted = ck_tile::type_convert<WeiDataType>(v_acc);
            weight(g, k, c, x)          = v_acc_converted;
        };

        make_ParallelTensorFunctor(func,
                                   weight.get_lengths()[0],
                                   weight.get_lengths()[1],
                                   weight.get_lengths()[2],
                                   weight.get_lengths()[3])(std::thread::hardware_concurrency());
    }
    else if constexpr(NDimSpatial == 2)
    {
        auto func = [&](auto g, auto k, auto c, auto y, auto x) {
            float v_acc = 0;

            for(std::size_t n = 0; n < output.get_lengths()[1]; ++n)
            {
                for(std::size_t ho = 0; ho < output.get_lengths()[3]; ++ho)
                {
                    auto hi = static_cast<ck_tile::long_index_t>(ho * conv_strides[0]) +
                              static_cast<ck_tile::long_index_t>(y * conv_dilations[0]) -
                              static_cast<ck_tile::long_index_t>(in_left_pads[0]);

                    for(std::size_t wo = 0; wo < output.get_lengths()[4]; ++wo)
                    {
                        auto wi = static_cast<ck_tile::long_index_t>(wo * conv_strides[1]) +
                                  static_cast<ck_tile::long_index_t>(x * conv_dilations[1]) -
                                  static_cast<ck_tile::long_index_t>(in_left_pads[1]);

                        if(hi >= 0 &&
                           ck_tile::type_convert<std::size_t>(hi) < input.get_lengths()[3] &&
                           wi >= 0 &&
                           ck_tile::type_convert<std::size_t>(wi) < input.get_lengths()[4])
                        {
                            InDataType v_in   = input(g, n, c, hi, wi);
                            OutDataType v_out = output(g, n, k, ho, wo);

                            v_acc += ck_tile::type_convert<float>(v_out) *
                                     ck_tile::type_convert<float>(v_in);
                        }
                    }
                }
            }
            WeiDataType v_acc_converted = ck_tile::type_convert<WeiDataType>(v_acc);
            weight(g, k, c, y, x)       = v_acc_converted;
        };

        make_ParallelTensorFunctor(func,
                                   weight.get_lengths()[0],
                                   weight.get_lengths()[1],
                                   weight.get_lengths()[2],
                                   weight.get_lengths()[3],
                                   weight.get_lengths()[4])(std::thread::hardware_concurrency());
    }
    else if constexpr(NDimSpatial == 3)
    {
        auto func = [&](auto g, auto k, auto c, auto z, auto y, auto x) {
            float v_acc = 0;

            for(std::size_t n = 0; n < output.get_lengths()[1]; ++n)
            {
                for(std::size_t do_ = 0; do_ < output.get_lengths()[3]; ++do_)
                {
                    auto di = static_cast<ck_tile::long_index_t>(do_ * conv_strides[0]) +
                              static_cast<ck_tile::long_index_t>(z * conv_dilations[0]) -
                              static_cast<ck_tile::long_index_t>(in_left_pads[0]);
                    for(std::size_t ho = 0; ho < output.get_lengths()[4]; ++ho)
                    {
                        auto hi = static_cast<ck_tile::long_index_t>(ho * conv_strides[1]) +
                                  static_cast<ck_tile::long_index_t>(y * conv_dilations[1]) -
                                  static_cast<ck_tile::long_index_t>(in_left_pads[1]);
                        for(std::size_t wo = 0; wo < output.get_lengths()[5]; ++wo)
                        {
                            auto wi = static_cast<ck_tile::long_index_t>(wo * conv_strides[2]) +
                                      static_cast<ck_tile::long_index_t>(x * conv_dilations[2]) -
                                      static_cast<ck_tile::long_index_t>(in_left_pads[2]);
                            if(di >= 0 &&
                               ck_tile::type_convert<std::size_t>(di) < input.get_lengths()[3] &&
                               hi >= 0 &&
                               ck_tile::type_convert<std::size_t>(hi) < input.get_lengths()[4] &&
                               wi >= 0 &&
                               ck_tile::type_convert<std::size_t>(wi) < input.get_lengths()[5])
                            {
                                InDataType v_in   = input(g, n, c, di, hi, wi);
                                OutDataType v_out = output(g, n, k, do_, ho, wo);

                                v_acc += ck_tile::type_convert<float>(v_out) *
                                         ck_tile::type_convert<float>(v_in);
                            }
                        }
                    }
                }
            }
            WeiDataType v_acc_converted = ck_tile::type_convert<WeiDataType>(v_acc);
            weight(g, k, c, z, y, x)    = v_acc_converted;
        };

        make_ParallelTensorFunctor(func,
                                   weight.get_lengths()[0],
                                   weight.get_lengths()[1],
                                   weight.get_lengths()[2],
                                   weight.get_lengths()[3],
                                   weight.get_lengths()[4],
                                   weight.get_lengths()[5])(std::thread::hardware_concurrency());
    }
    else
    {
        throw std::runtime_error(
            "Ref_conv_bwd_weight: number of dimensions must be between 1 and 3.");
    }
}
} // namespace ck_tile
