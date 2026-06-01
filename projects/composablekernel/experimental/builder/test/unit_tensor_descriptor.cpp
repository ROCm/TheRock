// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/builder/testing/tensor_descriptor.hpp"
#include "testing_utils.hpp"
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <array>
#include <sstream>
#include <vector>

namespace ckb = ck_tile::builder;
namespace ckt = ck_tile::builder::test;

using ck_tile::test::StringEqWithDiff;
using ::testing::ElementsAreArray;
using ::testing::Eq;
using ::testing::Throws;

TEST(TensorDescriptor, Basic)
{
    constexpr auto dt     = ckb::DataType::FP16;
    constexpr size_t rank = 3;
    ckt::Extent lengths   = {123, 456, 789};
    ckt::Extent strides   = {456 * 789, 789, 1};

    ckt::TensorDescriptor<dt, rank> descriptor(lengths, strides);

    EXPECT_THAT(descriptor.get_lengths(), ElementsAreArray(lengths));
    EXPECT_THAT(descriptor.get_strides(), ElementsAreArray(strides));
}

TEST(TensorDescriptor, ComputeSize)
{
    constexpr auto dt     = ckb::DataType::FP32;
    constexpr size_t rank = 3;
    ckt::Extent lengths   = {305, 130, 924};
    ckt::Extent strides   = {1001 * 1000, 1, 1000};

    ckt::TensorDescriptor<dt, rank> descriptor(lengths, strides);

    // Compute the location of the last item in memory,
    // then add one to get the minimum size.
    size_t expected_size  = 1;
    size_t expected_numel = 1;
    for(size_t i = 0; i < lengths.size(); ++i)
    {
        expected_size += (lengths[i] - 1) * strides[i];
        expected_numel *= lengths[i];
    }

    EXPECT_THAT(descriptor.get_element_size(), Eq(expected_numel));
    EXPECT_THAT(descriptor.get_element_space_size(), Eq(expected_size));
    EXPECT_THAT(descriptor.get_element_space_size_in_bytes(),
                Eq(expected_size * ckt::data_type_sizeof(dt)));
}

TEST(TensorDescriptor, PackedRightLayout)
{
    const ckt::Extent lengths = {5125, 623, 1177, 1534};
    const auto strides        = ckt::PackedRightLayout{}(lengths);

    EXPECT_THAT(strides, ElementsAreArray({623 * 1177 * 1534, 1177 * 1534, 1534, 1}));
}

TEST(TensorDescriptor, PackedLeftLayout)
{
    const ckt::Extent lengths = {4, 15, 925, 662, 1462};
    const auto strides        = ckt::PackedLeftLayout{}(lengths);

    EXPECT_THAT(strides, ElementsAreArray({1, 4, 4 * 15, 4 * 15 * 925, 4 * 15 * 925 * 662}));
}

TEST(TensorDescriptor, MakeDescriptor)
{
    {
        const ckt::Extent lengths = {10, 11, 12, 13, 14};

        // Note: automatic inference of RANK.
        const auto desc =
            ckt::make_descriptor<ckb::DataType::I32>(lengths, ckt::PackedRightLayout{});

        EXPECT_THAT(desc.get_lengths(), ElementsAreArray(lengths));
        EXPECT_THAT(desc.get_strides(),
                    ElementsAreArray({11 * 12 * 13 * 14, 12 * 13 * 14, 13 * 14, 14, 1}));
    }

    {
        const ckt::Extent lengths = {4, 3, 2};
        const ckt::Extent strides = {60, 1, 7};

        // Note: automatic inference of RANK.
        const auto desc = ckt::make_descriptor<ckb::DataType::FP8>(lengths, strides);

        EXPECT_THAT(desc.get_lengths(), ElementsAreArray(lengths));
        EXPECT_THAT(desc.get_strides(), ElementsAreArray(strides));
    }
}

TEST(TensorDescriptor, GetSpaceDescriptor)
{
    {
        const auto desc  = ckt::make_descriptor<ckb::DataType::FP32>(ckt::Extent{4, 4, 4},
                                                                    ckt::PackedLeftLayout{});
        const auto space = desc.get_space_descriptor();

        const auto expected = 4 * 4 * 4;

        EXPECT_THAT(decltype(space)::data_type, Eq(ckb::DataType::FP32));
        EXPECT_THAT(decltype(space)::rank, Eq(1));

        EXPECT_THAT(decltype(space)::data_type, Eq(ckb::DataType::FP32));
        EXPECT_THAT(decltype(space)::rank, Eq(1));
        EXPECT_THAT(space.get_lengths(), ElementsAreArray({expected}));
        EXPECT_THAT(space.get_strides(), ElementsAreArray({1}));
        EXPECT_THAT(space.get_element_size(), Eq(expected));
        EXPECT_THAT(space.get_element_space_size(), Eq(expected));
    }

    {
        const ckt::Extent lengths = {6, 3, 4};
        const ckt::Extent strides = {102, 1, 2002};
        const auto desc           = ckt::make_descriptor<ckb::DataType::FP32>(lengths, strides);
        const auto space          = desc.get_space_descriptor();

        // Compute the location of the last item in memory,
        // then add one to get the minimum size.
        size_t expected_size = 1;
        for(size_t i = 0; i < lengths.size(); ++i)
        {
            expected_size += (lengths[i] - 1) * strides[i];
        }

        EXPECT_THAT(decltype(space)::data_type, Eq(ckb::DataType::FP32));
        EXPECT_THAT(decltype(space)::rank, Eq(1));
        EXPECT_THAT(space.get_lengths(), ElementsAreArray({expected_size}));
        EXPECT_THAT(space.get_strides(), ElementsAreArray({1}));
        EXPECT_THAT(space.get_element_size(), Eq(expected_size));
        EXPECT_THAT(space.get_element_space_size(), Eq(expected_size));
    }
}

TEST(TensorDescriptor, EmptyExtent)
{
    // A rank-0 tensor points to a single element
    const auto desc = ckt::make_descriptor<ckb::DataType::FP16>(ckt::Extent{}, ckt::Extent{});
    EXPECT_THAT(decltype(desc)::rank, Eq(0));
    EXPECT_THAT(desc.get_lengths().size(), Eq(0));
    EXPECT_THAT(desc.get_strides().size(), Eq(0));
    EXPECT_THAT(desc.get_element_size(), Eq(1));
    EXPECT_THAT(desc.get_element_space_size(), Eq(1));
    EXPECT_THAT(desc.get_element_space_size_in_bytes(), Eq(2));

    // We expect a rank-1 tensor with the one dimension being 1.
    const auto space = desc.get_space_descriptor();

    const auto expected = 1;

    EXPECT_THAT(decltype(space)::rank, Eq(1));
    EXPECT_THAT(space.get_lengths(), ElementsAreArray({expected}));
    EXPECT_THAT(space.get_strides(), ElementsAreArray({1}));
    EXPECT_THAT(space.get_element_size(), Eq(expected));
    EXPECT_THAT(space.get_element_space_size(), Eq(expected));
    EXPECT_THAT(space.get_element_space_size_in_bytes(), Eq(2));
}

TEST(TensorDescriptor, ExtentFromVector)
{
    EXPECT_THAT(ckt::Extent<4>::from_vector(std::vector<size_t>{1, 2, 3, 4}),
                ElementsAreArray({1, 2, 3, 4}));

    EXPECT_THAT([] { return ckt::Extent<5>::from_vector(std::vector<size_t>{1, 2}); },
                Throws<std::runtime_error>());
}

TEST(TensorDescriptor, IsPacked)
{
    constexpr auto dt = ckb::DataType::I32; // Irrelevant for this test
    EXPECT_TRUE(
        ckt::make_descriptor<dt>(ckt::Extent{101, 43, 25, 662, 654}, ckt::PackedLeftLayout{})
            .is_packed());
    EXPECT_TRUE(
        ckt::make_descriptor<dt>(ckt::Extent{5334, 235, 1563, 256, 23}, ckt::PackedRightLayout{})
            .is_packed());
    EXPECT_TRUE(ckt::make_descriptor<dt>(ckt::Extent{}, ckt::Extent{}).is_packed());
    EXPECT_TRUE(
        ckt::make_descriptor<dt>(ckt::Extent{461, 345, 5, 93}, ckt::Extent{160425, 5, 1, 1725})
            .is_packed());
    EXPECT_FALSE(
        ckt::make_descriptor<dt>(ckt::Extent{10, 11, 12}, ckt::Extent{1, 100, 1100}).is_packed());
    EXPECT_FALSE(
        ckt::make_descriptor<dt>(ckt::Extent{30, 20, 10}, ckt::Extent{1, 1, 1}).is_packed());
}

TEST(TensorDescriptor, PrintExtent)
{
    {
        const ckt::Extent extent{6233, 55, 1235, 52, 203};
        std::stringstream ss;
        ss << extent;
        EXPECT_THAT(ss.str(), StringEqWithDiff("[6233, 55, 1235, 52, 203]"));
    }

    {
        const ckt::Extent extent{};
        std::stringstream ss;
        ss << extent;
        EXPECT_THAT(ss.str(), StringEqWithDiff("[]"));
    }
}
