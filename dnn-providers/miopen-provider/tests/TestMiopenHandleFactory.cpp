// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <miopen/miopen.h>

#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "HipdnnEnginePluginHandle.hpp"
#include "MiopenHandleFactory.hpp"

using namespace miopen_legacy_plugin;
using namespace hipdnn_plugin_sdk;

TEST(TestMiopenHandleFactory, ThrowsOnNullHandle)
{
    EXPECT_THROW(MiopenHandleFactory::createMiopenHandle(nullptr), HipdnnPluginException);
}

TEST(TestMiopenHandleFactory, ThrowsOnDestroyNullHandle)
{
    hipdnnEnginePluginHandle_t handle = nullptr;
    EXPECT_THROW(MiopenHandleFactory::destroyMiopenHandle(handle), HipdnnPluginException);
}

TEST(TestGpuMiopenHandleFactory, CreatesAndDestroysHandle)
{
    SKIP_IF_NO_DEVICES();

    hipdnnEnginePluginHandle_t handle = nullptr;
    EXPECT_NO_THROW(MiopenHandleFactory::createMiopenHandle(&handle));
    ASSERT_NE(handle, nullptr);
    ASSERT_NE(handle->miopenHandle, nullptr);

    // Clean up
    miopenDestroy(handle->miopenHandle);
    delete handle;
}
