// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <memory>

#include "HipdnnEnginePluginHandle.hpp"

class TestMiopenHipdnnEnginePluginHandle : public ::testing::Test
{
protected:
    void SetUp() override
    {
        _handle = std::make_unique<HipdnnEnginePluginHandle>();
    }

    std::unique_ptr<HipdnnEnginePluginHandle> _handle;
};

TEST_F(TestMiopenHipdnnEnginePluginHandle, DefaultConstruction)
{
    EXPECT_EQ(_handle->miopenHandle, nullptr);
    EXPECT_EQ(_handle->miopenContainer, nullptr);
    EXPECT_EQ(_handle->getStream(), nullptr);
}

TEST_F(TestMiopenHipdnnEnginePluginHandle, StoreDetachedBuffer)
{
    flatbuffers::FlatBufferBuilder builder;
    auto createdString = builder.CreateString("test");
    builder.Finish(createdString);
    auto buffer = std::make_unique<flatbuffers::DetachedBuffer>(builder.Release());
    const void* ptr = reinterpret_cast<const void*>(0x12354);

    _handle->storeEngineDetailsDetachedBuffer(ptr, std::move(buffer));

    // Buffer should be stored, verify by trying to remove it
    _handle->removeEngineDetailsDetachedBuffer(ptr);
}

TEST_F(TestMiopenHipdnnEnginePluginHandle, RemoveDetachedBuffer)
{
    flatbuffers::FlatBufferBuilder builder;
    auto createdString = builder.CreateString("test");
    builder.Finish(createdString);
    auto buffer = std::make_unique<flatbuffers::DetachedBuffer>(builder.Release());
    const void* ptr = reinterpret_cast<const void*>(0x12354);

    _handle->storeEngineDetailsDetachedBuffer(ptr, std::move(buffer));
    _handle->removeEngineDetailsDetachedBuffer(ptr);

    // Should not crash when removing non-existent buffer
    _handle->removeEngineDetailsDetachedBuffer(ptr);
}

TEST_F(TestMiopenHipdnnEnginePluginHandle, RemoveNonExistentBuffer)
{
    const void* fakePtr = reinterpret_cast<const void*>(0x12345678);

    // Should not crash when removing non-existent buffer
    EXPECT_NO_THROW(_handle->removeEngineDetailsDetachedBuffer(fakePtr));
}

TEST_F(TestMiopenHipdnnEnginePluginHandle, MultipleBuffers)
{
    flatbuffers::FlatBufferBuilder builder1;
    auto createdString1 = builder1.CreateString("test");
    builder1.Finish(createdString1);
    auto buffer1 = std::make_unique<flatbuffers::DetachedBuffer>(builder1.Release());
    const void* ptr1 = reinterpret_cast<const void*>(0x12354);

    flatbuffers::FlatBufferBuilder builder2;
    auto createdString2 = builder2.CreateString("test2");
    builder2.Finish(createdString2);
    auto buffer2 = std::make_unique<flatbuffers::DetachedBuffer>(builder2.Release());
    const void* ptr2 = reinterpret_cast<const void*>(0x54311);

    _handle->storeEngineDetailsDetachedBuffer(ptr1, std::move(buffer1));
    _handle->storeEngineDetailsDetachedBuffer(ptr2, std::move(buffer2));

    _handle->removeEngineDetailsDetachedBuffer(ptr1);
    _handle->removeEngineDetailsDetachedBuffer(ptr2);
}

TEST_F(TestMiopenHipdnnEnginePluginHandle, ThrowsWithNoMiopenHandle)
{
    EXPECT_THROW(_handle->setStream(nullptr), hipdnn_plugin_sdk::HipdnnPluginException);
}
