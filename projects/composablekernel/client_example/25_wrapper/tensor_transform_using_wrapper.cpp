// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>

#include "ck/ck.hpp"

#include "ck/utility/number.hpp"
#include "ck/utility/tuple.hpp"
#include "ck/utility/sequence.hpp"

#include "ck/wrapper/layout.hpp"

using DataType = int;

template <typename Layout>
void Print1d(const Layout& layout)
{
    std::cout << "Print1d" << std::endl;
    for(ck::index_t w = 0; w < ck::wrapper::size(layout); w++)
    {
        std::cout << layout(ck::make_tuple(w)) << " ";
    }
    std::cout << std::endl;
}

template <typename Layout>
void Print2d(const Layout& layout)
{
    std::cout << "Print2d" << std::endl;
    for(ck::index_t h = 0; h < ck::wrapper::size<0>(layout); h++)
    {
        for(ck::index_t w = 0; w < ck::wrapper::size<1>(layout); w++)
        {
            std::cout << layout(ck::make_tuple(h, w)) << " ";
        }
        std::cout << std::endl;
    }
}

// Print in (x,y),z pattern
template <typename Layout>
void Print3dCustom(const Layout& layout)
{
    std::cout << "Print3dCustom" << std::endl;
    for(ck::index_t d = 0; d < ck::wrapper::size<0>(ck::wrapper::get<0>(layout)); d++)
    {
        for(ck::index_t h = 0; h < ck::wrapper::size<1>(ck::wrapper::get<0>(layout)); h++)
        {
            for(ck::index_t w = 0; w < ck::wrapper::size<1>(layout); w++)
            {
                std::cout << layout(ck::make_tuple(ck::make_tuple(d, h), w)) << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

int main()
{
    // Layout traverse in row-major
    std::cout << "Note: Layout traverse in column-major" << std::endl;
    // Basic descriptor 0, 1, 2, ... 30, 31 (compile-time descriptor)
    // (dims:4,8 strides:1,4)
    const auto shape_4x8       = ck::make_tuple(ck::Number<4>{}, ck::Number<8>{});
    const auto layout_4x8_s1x4 = ck::wrapper::make_layout(shape_4x8);
    std::cout << "dims:4,8 strides:1,4" << std::endl;
    Print2d(layout_4x8_s1x4);
    using Cord1x1Type                = ck::Tuple<ck::Number<1>, ck::Number<1>>;
    constexpr ck::index_t offset_1x1 = layout_4x8_s1x4.template operator()<Cord1x1Type>();
    std::cout << "Constexpr calculated [1, 1] offset:" << offset_1x1 << std::endl;

    // Basic descriptor 0, 1, 8, 9, 16, 17, ... 30, 31 (runtime descriptor)
    // dims:4,(2,4) strides:2,(1,8)
    const auto shape_4x2x4         = ck::make_tuple(4, ck::make_tuple(2, 4));
    const auto strides_s2x1x8      = ck::make_tuple(2, ck::make_tuple(1, 8));
    const auto layout_4x2x4_s2x1x8 = ck::wrapper::make_layout(shape_4x2x4, strides_s2x1x8);

    std::cout << "dims:4,(2,4) strides:2,(1,8)" << std::endl;
    Print2d(layout_4x2x4_s2x1x8);

    // Basic descriptor 0, 1, 8, 9, 16, 17, ... 30, 31 (compile-time descriptor)
    // dims:(2,2),(2,4) strides:((1,4),(2,8)
    const auto shape_2x2x2x4    = ck::make_tuple(ck::make_tuple(ck::Number<2>{}, ck::Number<2>{}),
                                              ck::make_tuple(ck::Number<2>{}, ck::Number<4>{}));
    const auto strides_s1x4x2x8 = ck::make_tuple(ck::make_tuple(ck::Number<1>{}, ck::Number<4>{}),
                                                 ck::make_tuple(ck::Number<2>{}, ck::Number<8>{}));
    static const auto layout_2x2x2x4_s1x4x2x8 =
        ck::wrapper::make_layout(shape_2x2x2x4, strides_s1x4x2x8);

    std::cout << "dims:(2,2),(2,4) strides:(1,4),(2,8)" << std::endl;
    Print2d(layout_2x2x2x4_s1x4x2x8);
    Print3dCustom(layout_2x2x2x4_s1x4x2x8);

    // Basic descriptor 0, 1, 8, 9, 16, 17, ... 30, 31 (compile-time descriptor)
    // dims:((2,2),2),4 strides:((1,4),2),8
    // Transform to 2d
    const auto shape_2x2x2x4_nested = ck::make_tuple(
        ck::make_tuple(ck::make_tuple(ck::Number<2>{}, ck::Number<2>{}), ck::Number<2>{}),
        ck::Number<4>{});
    const auto strides_s1x4x2x8_nested = ck::make_tuple(
        ck::make_tuple(ck::make_tuple(ck::Number<1>{}, ck::Number<4>{}), ck::Number<2>{}),
        ck::Number<8>{});
    static const auto layout_2x2x2x4_s1x4x2x8_nested =
        ck::wrapper::make_layout(shape_2x2x2x4_nested, strides_s1x4x2x8_nested);

    std::cout << "dims:((2,2),2),4 strides:((1,4),2),8" << std::endl;
    Print1d(layout_2x2x2x4_s1x4x2x8_nested);
    Print2d(layout_2x2x2x4_s1x4x2x8_nested);
    Print3dCustom(layout_2x2x2x4_s1x4x2x8_nested);

    return 0;
}
