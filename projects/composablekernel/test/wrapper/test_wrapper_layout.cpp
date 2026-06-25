// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdlib>
#include <iostream>
#include <initializer_list>
#include <vector>
#include <gtest/gtest.h>

#include "ck/utility/common_header.hpp"

#include "ck/wrapper/layout.hpp"

#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/multi_index_transform_helper.hpp"

class TestWrapperLayout : public ::testing::Test
{
    protected:
    static constexpr auto I0 = ck::Number<0>{};
    static constexpr auto I1 = ck::Number<1>{};

    template <typename Desc,
              typename Desc1d,
              typename LayoutRuntime,
              typename LayoutCompiletime,
              typename Idxs>
    void Run(Desc& desc,
             Desc1d& desc_1d,
             LayoutRuntime& layout_runtime,
             LayoutCompiletime& layout_compiletime,
             const std::vector<Idxs>& idxs)
    {
        // 1d check
        EXPECT_EQ(desc_1d.GetLength(I0), ck::wrapper::size(layout_runtime));
        // Check layout compiletime and runtime result consistency
        EXPECT_EQ(ck::wrapper::size(layout_runtime), ck::wrapper::size(layout_compiletime));

        for(ck::index_t i = 0; i < desc_1d.GetLength(I0); i++)
        {
            const ck::index_t layout_runtime_offset_1d     = layout_runtime(ck::make_tuple(i));
            const ck::index_t layout_compiletime_offset_1d = layout_compiletime(ck::make_tuple(i));
            const ck::index_t desc_offset_1d = desc_1d.CalculateOffset(ck::make_tuple(i));
            EXPECT_EQ(layout_runtime_offset_1d, desc_offset_1d);
            EXPECT_EQ(layout_compiletime_offset_1d, layout_runtime_offset_1d);
        }
        // size(layout)-d check, don't check if access is hierarchical
        if constexpr(!IsNestedTuple(Idxs{}))
        {
            ck::static_for<0, Idxs::Size(), 1>{}([&](auto d) {
                EXPECT_EQ(desc.GetLength(ck::Number<d>{}), ck::wrapper::size<d>(layout_runtime));
                EXPECT_EQ(ck::wrapper::size<d>(layout_runtime),
                          ck::wrapper::size<d>(layout_compiletime));
            });
        }
        for(const auto idx : idxs)
        {
            const ck::index_t layout_runtime_offset     = layout_runtime(idx);
            const ck::index_t layout_compiletime_offset = layout_compiletime(idx);
            const ck::index_t desc_offset =
                desc.CalculateOffset(UnrollNestedTuple(idx)); // Unroll if nested
            EXPECT_EQ(layout_runtime_offset, desc_offset);
            EXPECT_EQ(layout_runtime_offset, layout_compiletime_offset);
        }
    }
};

TEST_F(TestWrapperLayout, 2d)
{
    // dims:(4, 3) strides:(1, 4)
    constexpr ck::index_t d1 = 4;
    constexpr ck::index_t d0 = 3;
    constexpr ck::index_t s1 = 1;
    constexpr ck::index_t s0 = 4;
    const auto desc =
        ck::make_naive_tensor_descriptor(ck::make_tuple(ck::Number<d1>{}, ck::Number<d0>{}),
                                         ck::make_tuple(ck::Number<s1>{}, ck::Number<s0>{}));
    // Reverse due to column major
    const auto desc_1d = transform_tensor_descriptor(
        desc,
        ck::make_tuple(ck::make_merge_transform(ck::make_tuple(d0, d1))),
        ck::make_tuple(ck::Sequence<1, 0>{}),
        ck::make_tuple(ck::Sequence<0>{}));
    const auto layout_runtime = ck::wrapper::make_layout(ck::make_tuple(d1, d0));
    const auto layout_compiletime =
        ck::wrapper::make_layout(ck::make_tuple(ck::Number<d1>{}, ck::Number<d0>{}),
                                 ck::make_tuple(ck::Number<s1>{}, ck::Number<s0>{}));
    std::vector<ck::Tuple<ck::index_t, ck::index_t>> idxs;

    for(ck::index_t h = 0; h < d1; h++)
    {
        for(ck::index_t w = 0; w < d0; w++)
        {
            idxs.emplace_back(h, w);
        }
    }

    this->Run(desc, desc_1d, layout_runtime, layout_compiletime, idxs);
}

TEST_F(TestWrapperLayout, 3d_nested)
{
    // dims:((2, 3), 4, 3) strides:((2, 4), 12, 48)
    constexpr ck::index_t d3 = 2;
    constexpr ck::index_t d2 = 3;
    constexpr ck::index_t d1 = 4;
    constexpr ck::index_t d0 = 3;
    constexpr ck::index_t s3 = 2;
    constexpr ck::index_t s2 = 4;
    constexpr ck::index_t s1 = 12;
    constexpr ck::index_t s0 = 48;
    const auto desc          = ck::make_naive_tensor_descriptor(
        ck::make_tuple(ck::Number<d3>{}, ck::Number<d2>{}, ck::Number<d1>{}, ck::Number<d0>{}),
        ck::make_tuple(ck::Number<s3>{}, ck::Number<s2>{}, ck::Number<s1>{}, ck::Number<s0>{}));
    // Reverse due to column major
    const auto desc_1d = transform_tensor_descriptor(
        desc,
        ck::make_tuple(ck::make_merge_transform(ck::make_tuple(d0, d1, d2, d3))),
        ck::make_tuple(ck::Sequence<3, 2, 1, 0>{}),
        ck::make_tuple(ck::Sequence<0>{}));
    const auto desc_3d = transform_tensor_descriptor(
        desc,
        ck::make_tuple(ck::make_merge_transform(ck::make_tuple(d2, d3)),
                       ck::make_pass_through_transform(d1),
                       ck::make_pass_through_transform(d2)),
        ck::make_tuple(ck::Sequence<1, 0>{}, ck::Sequence<2>{}, ck::Sequence<3>{}),
        ck::make_tuple(ck::Sequence<0>{}, ck::Sequence<1>{}, ck::Sequence<2>{}));
    const auto layout_runtime =
        ck::wrapper::make_layout(ck::make_tuple(ck::make_tuple(d3, d2), d1, d0),
                                 ck::make_tuple(ck::make_tuple(s3, s2), s1, s0));
    const auto layout_compiletime = ck::wrapper::make_layout(
        ck::make_tuple(
            ck::make_tuple(ck::Number<d3>{}, ck::Number<d2>{}), ck::Number<d1>{}, ck::Number<d0>{}),
        ck::make_tuple(ck::make_tuple(ck::Number<s3>{}, ck::Number<s2>{}),
                       ck::Number<s1>{},
                       ck::Number<s0>{}));
    std::vector<ck::Tuple<ck::index_t, ck::index_t, ck::index_t>> idxs_3d;

    for(ck::index_t d = 0; d < d2 * d3; d++)
    {
        for(ck::index_t h = 0; h < d1; h++)
        {
            for(ck::index_t w = 0; w < d0; w++)
            {
                idxs_3d.emplace_back(d, h, w);
            }
        }
    }
    this->Run(desc_3d, desc_1d, layout_runtime, layout_compiletime, idxs_3d);

    // Check also 4d iteration
    std::vector<ck::Tuple<ck::Tuple<ck::index_t, ck::index_t>, ck::index_t, ck::index_t>> idxs_4d;

    for(ck::index_t e = 0; e < d3; e++)
    {
        for(ck::index_t d = 0; d < d2; d++)
        {
            for(ck::index_t h = 0; h < d1; h++)
            {
                for(ck::index_t w = 0; w < d0; w++)
                {
                    idxs_4d.emplace_back(ck::make_tuple(e, d), h, w);
                }
            }
        }
    }
    this->Run(desc, desc_1d, layout_runtime, layout_compiletime, idxs_4d);
}

TEST_F(TestWrapperLayout, 2d_nested)
{
    // dims:((2, 3), (4, 3)) strides:((2, 4), (48, 12))
    constexpr ck::index_t d3 = 2;
    constexpr ck::index_t d2 = 3;
    constexpr ck::index_t d1 = 4;
    constexpr ck::index_t d0 = 3;
    constexpr ck::index_t s3 = 2;
    constexpr ck::index_t s2 = 4;
    constexpr ck::index_t s1 = 48;
    constexpr ck::index_t s0 = 12;
    const auto desc          = ck::make_naive_tensor_descriptor(
        ck::make_tuple(ck::Number<d3>{}, ck::Number<d2>{}, ck::Number<d1>{}, ck::Number<d0>{}),
        ck::make_tuple(ck::Number<s3>{}, ck::Number<s2>{}, ck::Number<s1>{}, ck::Number<s0>{}));
    // Reverse due to column major
    const auto desc_1d = transform_tensor_descriptor(
        desc,
        ck::make_tuple(ck::make_merge_transform(ck::make_tuple(d0, d1, d2, d3))),
        ck::make_tuple(ck::Sequence<3, 2, 1, 0>{}),
        ck::make_tuple(ck::Sequence<0>{}));
    const auto desc_2d = transform_tensor_descriptor(
        desc,
        ck::make_tuple(ck::make_merge_transform(ck::make_tuple(d2, d3)),
                       ck::make_merge_transform(ck::make_tuple(d0, d1))),
        ck::make_tuple(ck::Sequence<1, 0>{}, ck::Sequence<3, 2>{}),
        ck::make_tuple(ck::Sequence<0>{}, ck::Sequence<1>{}));
    const auto layout_runtime =
        ck::wrapper::make_layout(ck::make_tuple(ck::make_tuple(d3, d2), ck::make_tuple(d1, d0)),
                                 ck::make_tuple(ck::make_tuple(s3, s2), ck::make_tuple(s1, s0)));
    const auto layout_compiletime = ck::wrapper::make_layout(
        ck::make_tuple(ck::make_tuple(ck::Number<d3>{}, ck::Number<d2>{}),
                       ck::make_tuple(ck::Number<d1>{}, ck::Number<d0>{})),
        ck::make_tuple(ck::make_tuple(ck::Number<s3>{}, ck::Number<s2>{}),
                       ck::make_tuple(ck::Number<s1>{}, ck::Number<s0>{})));
    std::vector<ck::Tuple<ck::index_t, ck::index_t>> idxs_2d;

    for(ck::index_t h = 0; h < d2 * d3; h++)
    {
        for(ck::index_t w = 0; w < d0 * d1; w++)
        {
            idxs_2d.emplace_back(h, w);
        }
    }
    this->Run(desc_2d, desc_1d, layout_runtime, layout_compiletime, idxs_2d);
    // Check also 4d iteration
    std::vector<ck::Tuple<ck::Tuple<ck::index_t, ck::index_t>, ck::Tuple<ck::index_t, ck::index_t>>>
        idxs_4d;

    for(ck::index_t e = 0; e < d3; e++)
    {
        for(ck::index_t d = 0; d < d2; d++)
        {
            for(ck::index_t h = 0; h < d1; h++)
            {
                for(ck::index_t w = 0; w < d0; w++)
                {
                    idxs_4d.emplace_back(ck::make_tuple(e, d), ck::make_tuple(h, w));
                }
            }
        }
    }
    this->Run(desc, desc_1d, layout_runtime, layout_compiletime, idxs_4d);
}

TEST_F(TestWrapperLayout, 3d_double_nested)
{
    // dims:(((2, 2), 3), (4, 3)) strides:(((2, 4), 8), (96, 24))
    constexpr ck::index_t d4 = 2;
    constexpr ck::index_t d3 = 2;
    constexpr ck::index_t d2 = 3;
    constexpr ck::index_t d1 = 4;
    constexpr ck::index_t d0 = 3;
    constexpr ck::index_t s4 = 2;
    constexpr ck::index_t s3 = 4;
    constexpr ck::index_t s2 = 8;
    constexpr ck::index_t s1 = 96;
    constexpr ck::index_t s0 = 24;
    const auto desc          = ck::make_naive_tensor_descriptor(ck::make_tuple(ck::Number<d4>{},
                                                                      ck::Number<d3>{},
                                                                      ck::Number<d2>{},
                                                                      ck::Number<d1>{},
                                                                      ck::Number<d0>{}),
                                                       ck::make_tuple(ck::Number<s4>{},
                                                                      ck::Number<s3>{},
                                                                      ck::Number<s2>{},
                                                                      ck::Number<s1>{},
                                                                      ck::Number<s0>{}));
    // Reverse due to column major
    const auto desc_1d = transform_tensor_descriptor(
        desc,
        ck::make_tuple(ck::make_merge_transform(ck::make_tuple(d0, d1, d2, d3, d4))),
        ck::make_tuple(ck::Sequence<4, 3, 2, 1, 0>{}),
        ck::make_tuple(ck::Sequence<0>{}));
    const auto desc_3d = transform_tensor_descriptor(
        desc,
        ck::make_tuple(ck::make_merge_transform(ck::make_tuple(d3, d4)),
                       ck::make_pass_through_transform(d2),
                       ck::make_merge_transform(ck::make_tuple(d0, d1))),
        ck::make_tuple(ck::Sequence<1, 0>{}, ck::Sequence<2>{}, ck::Sequence<4, 3>{}),
        ck::make_tuple(ck::Sequence<0>{}, ck::Sequence<1>{}, ck::Sequence<2>{}));
    const auto desc_2d = transform_tensor_descriptor(
        desc_3d,
        ck::make_tuple(ck::make_merge_transform(ck::make_tuple(d2, d3 * d4)),
                       ck::make_pass_through_transform(d1 * d0)),
        ck::make_tuple(ck::Sequence<1, 0>{}, ck::Sequence<2>{}),
        ck::make_tuple(ck::Sequence<0>{}, ck::Sequence<1>{}));
    const auto layout_runtime = ck::wrapper::make_layout(
        ck::make_tuple(ck::make_tuple(ck::make_tuple(d4, d3), d2), ck::make_tuple(d1, d0)),
        ck::make_tuple(ck::make_tuple(ck::make_tuple(d4, s3), s2), ck::make_tuple(s1, s0)));
    const auto layout_compiletime = ck::wrapper::make_layout(
        ck::make_tuple(
            ck::make_tuple(ck::make_tuple(ck::Number<d4>{}, ck::Number<d3>{}), ck::Number<d2>{}),
            ck::make_tuple(ck::Number<d1>{}, ck::Number<d0>{})),
        ck::make_tuple(
            ck::make_tuple(ck::make_tuple(ck::Number<d4>{}, ck::Number<s3>{}), ck::Number<s2>{}),
            ck::make_tuple(ck::Number<s1>{}, ck::Number<s0>{})));
    std::vector<ck::Tuple<ck::index_t, ck::index_t>> idxs_2d;

    for(ck::index_t h = 0; h < d2 * d3 * d4; h++)
    {
        for(ck::index_t w = 0; w < d0 * d1; w++)
        {
            idxs_2d.emplace_back(h, w);
        }
    }
    this->Run(desc_2d, desc_1d, layout_runtime, layout_compiletime, idxs_2d);
    // Check also 3d iteration
    std::vector<ck::Tuple<ck::Tuple<ck::index_t, ck::index_t>, ck::index_t>> idxs_3d;

    for(ck::index_t d = 0; d < d3 * d4; d++)
    {
        for(ck::index_t h = 0; h < d2; h++)
        {
            for(ck::index_t w = 0; w < d1 * d0; w++)
            {
                idxs_3d.emplace_back(ck::make_tuple(d, h), w);
            }
        }
    }
    this->Run(desc_3d, desc_1d, layout_runtime, layout_compiletime, idxs_3d);
    // Check also 5d iteration
    std::vector<ck::Tuple<ck::Tuple<ck::Tuple<ck::index_t, ck::index_t>, ck::index_t>,
                          ck::Tuple<ck::index_t, ck::index_t>>>
        idxs_5d;

    for(ck::index_t f = 0; f < d4; f++)
    {
        for(ck::index_t e = 0; e < d3; e++)
        {
            for(ck::index_t d = 0; d < d2; d++)
            {
                for(ck::index_t h = 0; h < d1; h++)
                {
                    for(ck::index_t w = 0; w < d0; w++)
                    {
                        idxs_5d.emplace_back(ck::make_tuple(ck::make_tuple(f, e), d),
                                             ck::make_tuple(h, w));
                    }
                }
            }
        }
    }
    this->Run(desc, desc_1d, layout_runtime, layout_compiletime, idxs_5d);
}

TEST(TestLayoutHelpers, SizeAndGet)
{
    // dims:(((2, 2), 3), (4, 3))
    constexpr ck::index_t d4  = 2;
    constexpr ck::index_t d3  = 2;
    constexpr ck::index_t d2  = 3;
    constexpr ck::index_t d1  = 4;
    constexpr ck::index_t d0  = 3;
    const auto layout_runtime = ck::wrapper::make_layout(
        ck::make_tuple(ck::make_tuple(ck::make_tuple(d4, d3), d2), ck::make_tuple(d1, d0)));
    const auto layout_compiletime = ck::wrapper::make_layout(ck::make_tuple(
        ck::make_tuple(ck::make_tuple(ck::Number<d4>{}, ck::Number<d3>{}), ck::Number<d2>{}),
        ck::make_tuple(ck::Number<d1>{}, ck::Number<d0>{})));

    // Size of layout
    EXPECT_EQ(ck::wrapper::size(layout_runtime), d4 * d3 * d2 * d1 * d0);
    EXPECT_EQ(ck::wrapper::size(layout_compiletime), d4 * d3 * d2 * d1 * d0);

    // Size of dims
    EXPECT_EQ(ck::wrapper::size<0>(layout_runtime), d4 * d3 * d2);
    EXPECT_EQ(ck::wrapper::size<0>(layout_compiletime), d4 * d3 * d2);
    EXPECT_EQ(ck::wrapper::size<1>(layout_runtime), d1 * d0);
    EXPECT_EQ(ck::wrapper::size<1>(layout_compiletime), d1 * d0);

    // Access through new layout (using get with layout object)
    EXPECT_EQ(ck::wrapper::size<0>(ck::wrapper::get<0>(layout_runtime)), d4 * d3);
    EXPECT_EQ(ck::wrapper::size<0>(ck::wrapper::get<0>(layout_compiletime)), d4 * d3);
    EXPECT_EQ(ck::wrapper::size<1>(ck::wrapper::get<0>(layout_runtime)), d2);
    EXPECT_EQ(ck::wrapper::size<1>(ck::wrapper::get<0>(layout_compiletime)), d2);

    EXPECT_EQ(ck::wrapper::size<0>(ck::wrapper::get<0>(ck::wrapper::get<0>(layout_runtime))), d4);
    EXPECT_EQ(ck::wrapper::size<0>(ck::wrapper::get<0>(ck::wrapper::get<0>(layout_compiletime))),
              d4);
    EXPECT_EQ(ck::wrapper::size<1>(ck::wrapper::get<0>(ck::wrapper::get<0>(layout_runtime))), d3);
    EXPECT_EQ(ck::wrapper::size<1>(ck::wrapper::get<0>(ck::wrapper::get<0>(layout_compiletime))),
              d3);

    EXPECT_EQ(ck::wrapper::size<1>(ck::wrapper::get<0>(layout_runtime)), d2);
    EXPECT_EQ(ck::wrapper::size<1>(ck::wrapper::get<0>(layout_compiletime)), d2);

    EXPECT_EQ(ck::wrapper::size<0>(ck::wrapper::get<1>(layout_runtime)), d1);
    EXPECT_EQ(ck::wrapper::size<0>(ck::wrapper::get<1>(layout_compiletime)), d1);
    EXPECT_EQ(ck::wrapper::size<1>(ck::wrapper::get<1>(layout_runtime)), d0);
    EXPECT_EQ(ck::wrapper::size<1>(ck::wrapper::get<1>(layout_compiletime)), d0);
}

TEST(TestLayoutHelpers, DepthAndRank)
{
    // dims:(((2, 2), 3), (4, 3))
    constexpr ck::index_t d4  = 2;
    constexpr ck::index_t d3  = 2;
    constexpr ck::index_t d2  = 3;
    constexpr ck::index_t d1  = 4;
    constexpr ck::index_t d0  = 3;
    const auto layout_runtime = ck::wrapper::make_layout(
        ck::make_tuple(ck::make_tuple(ck::make_tuple(d4, d3), d2), ck::make_tuple(d1, d0)));
    const auto layout_compiletime = ck::wrapper::make_layout(ck::make_tuple(
        ck::make_tuple(ck::make_tuple(ck::Number<d4>{}, ck::Number<d3>{}), ck::Number<d2>{}),
        ck::make_tuple(ck::Number<d1>{}, ck::Number<d0>{})));

    EXPECT_EQ(ck::wrapper::depth(layout_runtime), 3);
    EXPECT_EQ(ck::wrapper::depth(layout_compiletime), 3);
    EXPECT_EQ(ck::wrapper::depth(ck::make_tuple(ck::make_tuple(d4, d3), d2)), 2);
    // Check for integer
    EXPECT_EQ(ck::wrapper::depth(d0), 0);

    EXPECT_EQ(ck::wrapper::rank(layout_runtime), 2);
    EXPECT_EQ(ck::wrapper::rank(layout_compiletime), 2);
    EXPECT_EQ(ck::wrapper::rank(ck::make_tuple(ck::make_tuple(d4, d3), d2)), 2);
    // Check for integer
    EXPECT_EQ(ck::wrapper::rank(d0), 1);
}

TEST(TestLayoutHelpers, ShapeAndStrides)
{
    // dims:(((2, 2), 3), (4, 3))
    constexpr ck::index_t d4     = 2;
    constexpr ck::index_t d3     = 2;
    constexpr ck::index_t d2     = 3;
    constexpr ck::index_t d1     = 4;
    constexpr ck::index_t d0     = 3;
    constexpr ck::index_t s4     = 2;
    constexpr ck::index_t s3     = 4;
    constexpr ck::index_t s2     = 8;
    constexpr ck::index_t s1     = 96;
    constexpr ck::index_t s0     = 24;
    const auto shape_compiletime = ck::make_tuple(
        ck::make_tuple(ck::make_tuple(ck::Number<d4>{}, ck::Number<d3>{}), ck::Number<d2>{}),
        ck::make_tuple(ck::Number<d1>{}, ck::Number<d0>{}));
    const auto strides_compiletime = ck::make_tuple(
        ck::make_tuple(ck::make_tuple(ck::Number<s4>{}, ck::Number<s3>{}), ck::Number<s2>{}),
        ck::make_tuple(ck::Number<s1>{}, ck::Number<s0>{}));
    const auto shape_runtime =
        ck::make_tuple(ck::make_tuple(ck::make_tuple(d4, d3), d2), ck::make_tuple(d1, d0));
    const auto strides_runtime =
        ck::make_tuple(ck::make_tuple(ck::make_tuple(s4, s3), s2), ck::make_tuple(s1, s0));
    const auto layout_runtime = ck::wrapper::make_layout(shape_runtime, strides_runtime);
    const auto layout_compiletime =
        ck::wrapper::make_layout(shape_compiletime, strides_compiletime);

    constexpr bool check_compiletime_shape =
        std::is_same_v<decltype(shape_compiletime),
                       std::remove_reference_t<decltype(shape(layout_compiletime))>>;
    constexpr bool check_runtime_shape =
        std::is_same_v<decltype(shape_runtime),
                       std::remove_reference_t<decltype(shape(layout_runtime))>>;
    EXPECT_TRUE(check_compiletime_shape);
    EXPECT_TRUE(check_runtime_shape);
}

TEST(TestLayoutHelpers, Hierarchical)
{
    // dims:(((2, 2), 3), (4, 3))
    constexpr ck::index_t d4 = 2;
    constexpr ck::index_t d3 = 2;
    constexpr ck::index_t d2 = 3;
    constexpr ck::index_t d1 = 4;
    constexpr ck::index_t d0 = 3;
    const auto runtime_shape =
        ck::make_tuple(ck::make_tuple(ck::make_tuple(d4, d3), d2), ck::make_tuple(d1, d0));
    const auto layout_runtime     = ck::wrapper::make_layout(runtime_shape);
    const auto layout_compiletime = ck::wrapper::make_layout(ck::make_tuple(
        ck::make_tuple(ck::make_tuple(ck::Number<d4>{}, ck::Number<d3>{}), ck::Number<d2>{}),
        ck::make_tuple(ck::Number<d1>{}, ck::Number<d0>{})));

    EXPECT_EQ((ck::wrapper::rank<0, 0>(runtime_shape)), 2);
    EXPECT_EQ((ck::wrapper::rank<0, 0>(layout_runtime)), 2);
    EXPECT_EQ((ck::wrapper::rank<0, 0>(layout_compiletime)), 2);

    EXPECT_EQ((ck::wrapper::depth<0, 0>(runtime_shape)), 1);
    EXPECT_EQ((ck::wrapper::depth<0, 0>(layout_runtime)), 1);
    EXPECT_EQ((ck::wrapper::depth<0, 0>(layout_compiletime)), 1);

    EXPECT_EQ((ck::wrapper::size<0, 0>(runtime_shape)), d4 * d3);
    EXPECT_EQ((ck::wrapper::size<0, 0>(layout_runtime)), d4 * d3);
    EXPECT_EQ((ck::wrapper::size<0, 0>(layout_compiletime)), d4 * d3);

    EXPECT_EQ((ck::wrapper::get<0, 0, 0>(runtime_shape)), d4);
}
