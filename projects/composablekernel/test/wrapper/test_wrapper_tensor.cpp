// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdlib>
#include <iostream>
#include <initializer_list>
#include <vector>
#include <gtest/gtest.h>

#include "ck/library/utility/device_memory.hpp"

#include "ck/host_utility/kernel_launch.hpp"

#include "ck/utility/common_header.hpp"

#include "ck/wrapper/layout.hpp"
#include "ck/wrapper/tensor.hpp"

using ::ck::DeviceMem;
using ::ck::launch_and_time_kernel;

// Compare data in tensor with offset from layout.
// Data and offset should match if physical memory has been initialized with
// sequentially increasing values from 0.
template <typename TensorType>
__host__ __device__ bool TestTensorCheck3d(TensorType& tensor)
{
    const auto& layout = ck::wrapper::layout(tensor);
    for(ck::index_t d = 0; d < ck::wrapper::size<0>(ck::wrapper::get<0>(layout)); d++)
    {
        for(ck::index_t h = 0; h < ck::wrapper::size<1>(ck::wrapper::get<0>(layout)); h++)
        {
            for(ck::index_t w = 0; w < ck::wrapper::size<1>(layout); w++)
            {
                const auto idx = ck::make_tuple(ck::make_tuple(d, h), w);
                if(tensor(idx) != layout(idx))
                {
                    return false;
                }
            }
        }
    }
    return true;
}

template <typename TensorType>
__host__ __device__ bool TestTensorCheck1d(TensorType& tensor, ck::index_t start_offset = 0)
{
    const auto& layout = ck::wrapper::layout(tensor);
    for(ck::index_t w = 0; w < ck::wrapper::size<0>(layout); w++)
    {
        if(tensor(w) - start_offset != layout(ck::make_tuple(w)))
        {
            return false;
        }
    }
    return true;
}

template <ck::index_t nelems, typename TensorType>
__host__ __device__ bool StaticTestTensorCheck1d(TensorType& tensor)
{
    const auto& layout = ck::wrapper::layout(tensor);
    bool success       = true;
    ck::static_for<0, nelems, 1>{}([&](auto w) {
        if(tensor(ck::Number<w.value>{}) != layout(ck::make_tuple(w.value)))
        {
            success = false;
        }
    });
    return success;
}

template <typename TensorType>
__host__ __device__ void InitTensor(TensorType& tensor)
{
    for(ck::index_t i = 0; i < ck::wrapper::size(ck::wrapper::layout(tensor)); i++)
    {
        tensor(i) = i;
    }
}

template <ck::index_t nelems, typename TensorType>
__host__ __device__ void StaticInitTensor(TensorType& tensor)
{

    ck::static_for<0, nelems, 1>{}([&](auto i) { tensor(ck::Number<i.value>{}) = i.value; });
}

// Tests
TEST(TestTensor, ReadWriteHostMemory)
{
    constexpr ck::index_t nelems = 8;

    std::array<ck::index_t, nelems> data;
    const auto layout = ck::wrapper::make_layout(ck::make_tuple(ck::make_tuple(2, 2), 2));
    auto tensor = ck::wrapper::make_tensor<ck::wrapper::MemoryTypeEnum::Generic>(&data[0], layout);
    InitTensor(tensor);

    EXPECT_TRUE(TestTensorCheck1d(tensor));
    EXPECT_TRUE(TestTensorCheck3d(tensor));
}

__global__ void TestTensorReadWriteDevice(void* data, void* success)
{
    constexpr ck::index_t nelems = 8;
    __shared__ ck::index_t p_shared[nelems];

    ck::index_t* casted_data_ptr = static_cast<ck::index_t*>(data);
    bool* casted_success_ptr     = static_cast<bool*>(success);

    const auto layout = ck::wrapper::make_layout(ck::make_tuple(ck::make_tuple(2, 2), 2));
    constexpr auto vgpr_layout =
        ck::wrapper::make_layout(make_tuple(ck::Number<nelems>{}), make_tuple(ck::Number<1>{}));

    auto tensor_global =
        ck::wrapper::make_tensor<ck::wrapper::MemoryTypeEnum::Global>(casted_data_ptr, layout);
    auto tensor_lds = ck::wrapper::make_tensor<ck::wrapper::MemoryTypeEnum::Lds>(p_shared, layout);
    auto tensor_vgpr =
        ck::wrapper::make_register_tensor<ck::wrapper::MemoryTypeEnum::Vgpr, ck::index_t>(
            vgpr_layout);

    InitTensor(tensor_global);
    InitTensor(tensor_lds);
    StaticInitTensor<nelems>(tensor_vgpr);

    *casted_success_ptr = TestTensorCheck1d(tensor_global);
    *casted_success_ptr &= TestTensorCheck3d(tensor_global);

    *casted_success_ptr &= TestTensorCheck1d(tensor_lds);
    *casted_success_ptr &= TestTensorCheck3d(tensor_lds);

    *casted_success_ptr &= StaticTestTensorCheck1d<nelems>(tensor_vgpr);
}

TEST(TestTensor, ReadWriteGlobalLdsRegistersMemory)
{
    constexpr ck::index_t nelems = 8;
    std::array<ck::index_t, nelems> host_data;

    DeviceMem data_buf(nelems * sizeof(ck::index_t));
    data_buf.ToDevice(&host_data[0]);
    DeviceMem success_buf(sizeof(bool));

    launch_and_time_kernel(StreamConfig{},
                           TestTensorReadWriteDevice,
                           dim3(1),
                           dim3(1),
                           0,
                           data_buf.GetDeviceBuffer(),
                           success_buf.GetDeviceBuffer());

    bool success;
    success_buf.FromDevice(&success);
    EXPECT_TRUE(success);
}

TEST(TestTensor, Slicing)
{
    constexpr ck::index_t nelems = 8;

    std::array<ck::index_t, nelems> data;
    const auto shape   = ck::make_tuple(ck::make_tuple(2, 2), 2);
    const auto strides = ck::make_tuple(ck::make_tuple(1, 2), 4);
    const auto layout  = ck::wrapper::make_layout(shape, strides);
    auto tensor = ck::wrapper::make_tensor<ck::wrapper::MemoryTypeEnum::Generic>(&data[0], layout);
    InitTensor(tensor);

    auto tensor2x2x2 =
        tensor(ck::make_tuple(ck::wrapper::slice(2), ck::wrapper::slice(2)), ck::wrapper::slice(2));
    EXPECT_EQ(tensor2x2x2(0), layout(ck::make_tuple(ck::make_tuple(0, 0), 0)));
    EXPECT_EQ(ck::wrapper::rank(tensor2x2x2), 2);
    EXPECT_EQ(ck::wrapper::depth(tensor2x2x2), 2);
    EXPECT_EQ(ck::wrapper::size(tensor2x2x2), 8);
    EXPECT_TRUE(TestTensorCheck1d(tensor2x2x2));

    auto tensor2x2 = tensor(ck::make_tuple(1, ck::wrapper::slice(2)), ck::wrapper::slice(2));
    EXPECT_EQ(tensor2x2(0), layout(ck::make_tuple(ck::make_tuple(1, 0), 0)));
    EXPECT_EQ(ck::wrapper::rank(tensor2x2), 2);
    EXPECT_EQ(ck::wrapper::depth(tensor2x2), 2);
    EXPECT_EQ(ck::wrapper::size(tensor2x2), 4);
    EXPECT_TRUE(TestTensorCheck1d(tensor2x2));

    auto tensor1x1 = tensor(ck::make_tuple(1, ck::wrapper::slice(1, 2)), ck::wrapper::slice(1, 2));
    EXPECT_EQ(tensor1x1(0), layout(ck::make_tuple(ck::make_tuple(1, 1), 1)));
    EXPECT_EQ(rank(tensor1x1), 2);
    EXPECT_EQ(depth(tensor1x1), 2);
    EXPECT_EQ(size(tensor1x1), 1);
    EXPECT_TRUE(TestTensorCheck1d(tensor1x1));

    auto tensor2 = tensor(ck::make_tuple(1, 1), ck::wrapper::slice(0, 2));
    EXPECT_EQ(tensor2(0), layout(ck::make_tuple(ck::make_tuple(1, 1), 0)));
    EXPECT_EQ(ck::wrapper::rank(tensor2), 1);
    EXPECT_EQ(ck::wrapper::depth(tensor2), 1);
    EXPECT_EQ(ck::wrapper::size(tensor2), 2);
    EXPECT_TRUE(TestTensorCheck1d(tensor2));

    auto tensor2_v2 = tensor(2, ck::wrapper::slice(0, 2));
    EXPECT_EQ(tensor2_v2(0), layout(ck::make_tuple(2, 0)));
    EXPECT_EQ(ck::wrapper::rank(tensor2_v2), 1);
    EXPECT_EQ(ck::wrapper::depth(tensor2_v2), 1);
    EXPECT_EQ(ck::wrapper::size(tensor2_v2), 2);
    EXPECT_TRUE(TestTensorCheck1d(tensor2_v2));

    // negative indexing
    auto tensor1x2 = tensor(ck::make_tuple(1, ck::wrapper::slice(0, -2)), ck::wrapper::slice());
    EXPECT_EQ(tensor1x2(0), layout(ck::make_tuple(ck::make_tuple(1, 0), 0)));
    EXPECT_EQ(rank(tensor1x2), 2);
    EXPECT_EQ(depth(tensor1x2), 2);
    EXPECT_EQ(size(tensor1x2), 2);
    EXPECT_TRUE(TestTensorCheck1d(tensor1x2));
}
