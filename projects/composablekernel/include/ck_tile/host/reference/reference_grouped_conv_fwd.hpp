// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdlib>
#include <thread>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/elementwise.hpp"
#include "ck_tile/host/host_tensor.hpp"

namespace ck_tile {

template <ck_tile::index_t NDimSpatial,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename Elfunc = ck_tile::element_wise::PassThrough,
          typename Tuple  = ck_tile::tuple<>>
CK_TILE_HOST void reference_grouped_conv_fwd(const HostTensor<InDataType>& input,
                                             const HostTensor<WeiDataType>& weight,
                                             HostTensor<OutDataType>& output,
                                             std::vector<ck_tile::long_index_t> conv_strides,
                                             std::vector<ck_tile::long_index_t> conv_dilations,
                                             std::vector<ck_tile::long_index_t> in_left_pads,
                                             std::vector<ck_tile::long_index_t>,
                                             Elfunc elfunc = Elfunc{},
                                             Tuple ds      = {})
{
    if(!(input.get_num_of_dimension() == NDimSpatial + 3 &&
         weight.get_num_of_dimension() == NDimSpatial + 3 &&
         output.get_num_of_dimension() == NDimSpatial + 3))
    {
        throw std::runtime_error("wrong! inconsistent dimension");
    }

    if constexpr(NDimSpatial == 1)
    {
        auto func = [&](auto g, auto n, auto k, auto wo) {
            float v_acc = 0;

            for(std::size_t c = 0; c < weight.get_lengths()[2]; ++c)
            {
                for(std::size_t x = 0; x < weight.get_lengths()[3]; ++x)
                {
                    auto wi = static_cast<ck_tile::long_index_t>(wo * conv_strides[0]) +
                              static_cast<ck_tile::long_index_t>(x * conv_dilations[0]) -
                              static_cast<ck_tile::long_index_t>(in_left_pads[0]);

                    if(wi >= 0 && ck_tile::type_convert<std::size_t>(wi) < input.get_lengths()[3])
                    {
                        InDataType v_in   = input(g, n, c, wi);
                        WeiDataType v_wei = weight(g, k, c, x);
                        v_acc += ck_tile::type_convert<float>(v_in) *
                                 ck_tile::type_convert<float>(v_wei);
                    }
                }
            }
            if constexpr(Tuple::size() > 0)
                elfunc(v_acc, v_acc, ds.at(ck_tile::number<0>{})(g, n, k, wo));
            else
                elfunc(v_acc, v_acc);
            OutDataType v_acc_out = ck_tile::type_convert<OutDataType>(v_acc);
            output(g, n, k, wo)   = v_acc_out;
        };

        make_ParallelTensorFunctor(func,
                                   output.get_lengths()[0],
                                   output.get_lengths()[1],
                                   output.get_lengths()[2],
                                   output.get_lengths()[3])(std::thread::hardware_concurrency());
    }
    else if constexpr(NDimSpatial == 2)
    {
        auto func = [&](auto g, auto n, auto k, auto ho, auto wo) {
            float v_acc = 0;

            for(std::size_t c = 0; c < weight.get_lengths()[2]; ++c)
            {
                for(std::size_t y = 0; y < weight.get_lengths()[3]; ++y)
                {
                    auto hi = static_cast<ck_tile::long_index_t>(ho * conv_strides[0]) +
                              static_cast<ck_tile::long_index_t>(y * conv_dilations[0]) -
                              static_cast<ck_tile::long_index_t>(in_left_pads[0]);

                    for(std::size_t x = 0; x < weight.get_lengths()[4]; ++x)
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
                            WeiDataType v_wei = weight(g, k, c, y, x);

                            v_acc += ck_tile::type_convert<float>(v_in) *
                                     ck_tile::type_convert<float>(v_wei);
                        }
                    }
                }
            }
            if constexpr(Tuple::size() > 0)
                elfunc(v_acc, v_acc, ds.at(ck_tile::number<0>{})(g, n, k, ho, wo));
            else
                elfunc(v_acc, v_acc);
            OutDataType v_acc_out   = ck_tile::type_convert<OutDataType>(v_acc);
            output(g, n, k, ho, wo) = v_acc_out;
        };

        make_ParallelTensorFunctor(func,
                                   output.get_lengths()[0],
                                   output.get_lengths()[1],
                                   output.get_lengths()[2],
                                   output.get_lengths()[3],
                                   output.get_lengths()[4])(std::thread::hardware_concurrency());
    }
    else if constexpr(NDimSpatial == 3)
    {
        auto func = [&](auto g, auto n, auto k, auto d_o, auto ho, auto wo) {
            float v_acc = 0;

            for(std::size_t c = 0; c < weight.get_lengths()[2]; ++c)
            {
                for(std::size_t z = 0; z < weight.get_lengths()[3]; ++z)
                {
                    auto di = static_cast<ck_tile::long_index_t>(d_o * conv_strides[0]) +
                              static_cast<ck_tile::long_index_t>(z * conv_dilations[0]) -
                              static_cast<ck_tile::long_index_t>(in_left_pads[0]);
                    for(std::size_t y = 0; y < weight.get_lengths()[4]; ++y)
                    {
                        auto hi = static_cast<ck_tile::long_index_t>(ho * conv_strides[1]) +
                                  static_cast<ck_tile::long_index_t>(y * conv_dilations[1]) -
                                  static_cast<ck_tile::long_index_t>(in_left_pads[1]);
                        for(std::size_t x = 0; x < weight.get_lengths()[5]; ++x)
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
                                WeiDataType v_wei = weight(g, k, c, z, y, x);

                                v_acc += ck_tile::type_convert<float>(v_in) *
                                         ck_tile::type_convert<float>(v_wei);
                            }
                        }
                    }
                }
            }
            if constexpr(Tuple::size() > 0)
                elfunc(v_acc, v_acc, ds.at(ck_tile::number<0>{})(g, n, k, d_o, ho, wo));
            else
                elfunc(v_acc, v_acc);
            OutDataType v_acc_out        = ck_tile::type_convert<OutDataType>(v_acc);
            output(g, n, k, d_o, ho, wo) = v_acc_out;
        };

        make_ParallelTensorFunctor(func,
                                   output.get_lengths()[0],
                                   output.get_lengths()[1],
                                   output.get_lengths()[2],
                                   output.get_lengths()[3],
                                   output.get_lengths()[4],
                                   output.get_lengths()[5])(std::thread::hardware_concurrency());
    }
    else
    {
        throw std::runtime_error("Ref_Conv_fwd: number of dimensions must be between 1 and 3.");
    }
}
} // namespace ck_tile
