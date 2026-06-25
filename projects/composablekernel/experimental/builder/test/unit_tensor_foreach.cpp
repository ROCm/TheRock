// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/builder/testing/tensor_descriptor.hpp"
#include "ck_tile/builder/testing/tensor_buffer.hpp"
#include "ck_tile/builder/testing/tensor_foreach.hpp"
#include "testing_utils.hpp"
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <algorithm>
#include <functional>

namespace ckb = ck_tile::builder;
namespace ckt = ck_tile::builder::test;

using ::testing::Each;
using ::testing::Eq;

TEST(TensorForeach, NdIter)
{
    {
        ckt::NdIter iter(ckt::Extent{523, 345, 123, 601});

        EXPECT_THAT(iter.numel(), Eq(13'338'296'505ULL));
        EXPECT_THAT(iter(0), Eq(ckt::Extent{0, 0, 0, 0}));
        EXPECT_THAT(iter(1), Eq(ckt::Extent{0, 0, 0, 1}));
        EXPECT_THAT(iter(601), Eq(ckt::Extent{0, 0, 1, 0}));
        EXPECT_THAT(iter(601 * 123), Eq(ckt::Extent{0, 1, 0, 0}));
        EXPECT_THAT(iter(601 * 123 * 10), Eq(ckt::Extent{0, 10, 0, 0}));
        EXPECT_THAT(iter(((34 * 345 + 63) * 123 + 70) * 601 + 5), Eq(ckt::Extent{34, 63, 70, 5}));
    }

    {
        ckt::NdIter iter(ckt::Extent{});

        EXPECT_THAT(iter.numel(), Eq(1));
        EXPECT_THAT(iter(0), Eq(ckt::Extent{}));
    }
}

TEST(TensorForeach, CalculateOffset)
{
    EXPECT_THAT(ckt::calculate_offset(ckt::Extent{1, 2, 3}, ckt::Extent{100, 10, 1}), Eq(123));
    EXPECT_THAT(ckt::calculate_offset(ckt::Extent{523, 266, 263}, ckt::Extent{1, 545, 10532}),
                Eq(2915409));
    EXPECT_THAT(ckt::calculate_offset(ckt::Extent{}, ckt::Extent{}), Eq(0));
    // Note: >4 GB overflow test
    EXPECT_THAT(ckt::calculate_offset(ckt::Extent{8, 2, 5, 7, 0, 4, 1, 3, 6, 9},
                                      ckt::Extent{1'000,
                                                  1'000'000,
                                                  10'000'000,
                                                  1'000'000'000,
                                                  1,
                                                  10'000,
                                                  100,
                                                  10,
                                                  100'000'000,
                                                  100'000}),
                Eq(size_t{7'652'948'130}));
}

TEST(TensorForeach, VisitsCorrectCount)
{
    // tensor_foreach should visit every index exactly once.
    // This test checks that the count is at least correct.

    const ckt::Extent shape = {10, 20, 30};

    auto d_count = ckt::alloc_buffer(sizeof(uint64_t));
    ckt::check_hip(hipMemset(d_count.get(), 0, sizeof(uint64_t)));

    ckt::tensor_foreach(shape, [count = d_count.get()]([[maybe_unused]] const auto& index) {
        atomicAdd(reinterpret_cast<uint64_t*>(count), 1);
    });

    uint64_t actual;
    ckt::check_hip(hipMemcpy(&actual, d_count.get(), sizeof(uint64_t), hipMemcpyDeviceToHost));

    const auto expected = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());

    EXPECT_THAT(actual, Eq(expected));
}

TEST(TensorForeach, VisitsEveryIndex)
{
    const ckt::Extent shape = {5, 6, 7, 8, 9, 10, 11};
    const auto total = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());

    // We know this is correct due to testing in unit_tensor_descriptor.cpp
    const auto stride = ckt::PackedRightLayout{}(shape);

    auto d_output = ckt::alloc_buffer(sizeof(uint32_t) * total);
    ckt::check_hip(hipMemset(d_output.get(), 0, sizeof(uint32_t) * total));

    ckt::tensor_foreach(shape, [output = d_output.get(), stride](const auto& index) {
        // We know this is correct due to the CalculateOffset test.
        auto offset = ckt::calculate_offset(index, stride);

        // Use atomic add so that we can check that every index is visited exactly once.
        atomicAdd(&reinterpret_cast<uint32_t*>(output)[offset], 1);
    });

    std::vector<uint32_t> actual(total);
    ckt::check_hip(
        hipMemcpy(actual.data(), d_output.get(), sizeof(uint32_t) * total, hipMemcpyDeviceToHost));

    EXPECT_THAT(actual, Each(Eq(1)));
}

TEST(TensorForeach, FillTensorBuffer)
{
    auto desc =
        ckt::make_descriptor<ckb::DataType::I32>(ckt::Extent{31, 54, 13}, ckt::PackedRightLayout{});

    auto buffer = ckt::alloc_tensor_buffer(desc);

    ckt::fill_tensor_buffer(desc, buffer.get(), [](size_t i) { return static_cast<uint32_t>(i); });

    std::vector<uint32_t> h_buffer(desc.get_element_space_size());
    ckt::check_hip(hipMemcpy(
        h_buffer.data(), buffer.get(), h_buffer.size() * sizeof(uint32_t), hipMemcpyDeviceToHost));

    for(size_t i = 0; i < h_buffer.size(); ++i)
    {
        EXPECT_THAT(h_buffer[i], Eq(static_cast<uint32_t>(i)));
    }
}

TEST(TensorForeach, FillTensor)
{
    // FillTensor with non-packed indices should not write out-of-bounds.
    const ckt::Extent shape = {4, 23, 35};
    const ckt::Extent pad   = {12, 53, 100};
    auto desc = ckt::make_descriptor<ckb::DataType::I32>(shape, ckt::PackedRightLayout{}(pad));
    const auto strides = desc.get_strides();

    auto size   = desc.get_element_space_size();
    auto buffer = ckt::alloc_tensor_buffer(desc);

    ckt::fill_tensor_buffer(desc, buffer.get(), []([[maybe_unused]] size_t i) { return 123; });

    ckt::fill_tensor(desc, buffer.get(), []([[maybe_unused]] const auto& index) { return 1; });

    auto d_error = ckt::alloc_buffer(sizeof(uint32_t) * size);
    ckt::check_hip(hipMemset(d_error.get(), 0, sizeof(uint32_t)));

    ckt::tensor_foreach(
        // Iterate over the entire padding so that we can check out-of-bounds elements
        pad,
        [shape, pad, strides, size, error = d_error.get(), tensor = buffer.get()](
            const auto& index) {
            const auto offset = ckt::calculate_offset(index, strides);
            const auto value  = reinterpret_cast<const uint32_t*>(tensor)[offset];

            // Note: The space of the descriptor will not actually be (12, 53, 100) but
            // more like (4, 53, 100), as the outer stride is irrelevant. So we have to
            // perform an extra bounds check here.
            if(offset < size)
            {
                // Check if the coordinate is within the shape bounds.
                bool in_bounds = true;
                for(size_t i = 0; i < shape.size(); ++i)
                {
                    if(index[i] >= shape[i])
                    {
                        in_bounds = false;
                    }
                }

                // In-bounds elements are 1, out-of-bounds is 123.
                if(in_bounds && value != 1)
                {
                    atomicAdd(reinterpret_cast<uint32_t*>(error), 1);
                }
                else if(!in_bounds && value != 123)
                {
                    atomicAdd(reinterpret_cast<uint32_t*>(error), 1);
                }
            }
        });

    uint32_t error_count = 0;
    ckt::check_hip(hipMemcpy(&error_count, d_error.get(), sizeof(uint32_t), hipMemcpyDeviceToHost));

    EXPECT_THAT(error_count, Eq(0));
}

TEST(TensorForeach, ClearTensorZeros)
{
    const ckt::Extent shape = {5, 4, 5, 4, 5, 4, 5, 6};
    const ckt::Extent pad   = {6, 6, 6, 6, 6, 6, 6, 6};

    const auto desc =
        ckt::make_descriptor<ckb::DataType::I32>(shape, ckt::PackedRightLayout{}(pad));

    auto buffer = ckt::alloc_tensor_buffer(desc);
    ckt::clear_tensor_buffer(desc, buffer.get());

    // Check that all values are zeroed.
    auto d_count = ckt::alloc_buffer(sizeof(uint64_t));
    ckt::check_hip(hipMemset(d_count.get(), 0, sizeof(uint64_t)));

    {
        const auto size    = desc.get_element_space_size();
        const auto strides = desc.get_strides();
        auto* count        = d_count.get();
        const auto* tensor = reinterpret_cast<const uint32_t*>(buffer.get());
        // Note: iterate over the entire pad, so that we can check out-of-bounds elements.
        ckt::tensor_foreach(pad,
                            [count, tensor, strides, size]([[maybe_unused]] const auto& index) {
                                const auto offset = ckt::calculate_offset(index, strides);

                                // Note: The space of the descriptor will not actually be (6, 6,
                                // ...) but more like (5, 6, ...), as the outer stride is
                                // irrelevant. So we have to perform an extra bounds check here.
                                if(offset < size && tensor[offset] != 0)
                                {
                                    atomicAdd(reinterpret_cast<uint64_t*>(count), 1);
                                }
                            });
    }

    uint64_t actual;
    ckt::check_hip(hipMemcpy(&actual, d_count.get(), sizeof(uint64_t), hipMemcpyDeviceToHost));

    EXPECT_THAT(actual, Eq(0));
}
