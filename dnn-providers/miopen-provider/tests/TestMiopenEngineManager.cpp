// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <memory>
#include <set>

#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_test_sdk/utilities/MockEngineConfig.hpp>
#include <hipdnn_test_sdk/utilities/MockGraph.hpp>

#include "EngineManager.hpp"
#include "HipdnnEnginePluginExecutionContext.hpp"
#include "HipdnnEnginePluginHandle.hpp"
#include "mocks/MockEngine.hpp"
#include "mocks/MockHipdnnEnginePluginExecutionContext.hpp"

using namespace miopen_legacy_plugin;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_plugin_sdk;
using ::testing::Return;

TEST(TestMiopenEngineManager, ReturnsApplicableEngineIds)
{
    std::set<std::unique_ptr<IEngine>> engines;

    auto mockEngine1 = std::make_unique<MockEngine>();
    EXPECT_CALL(*mockEngine1, id()).WillRepeatedly(Return(1));
    EXPECT_CALL(*mockEngine1, isApplicable(::testing::_, ::testing::_))
        .WillRepeatedly(Return(true));

    auto mockEngine2 = std::make_unique<MockEngine>();
    EXPECT_CALL(*mockEngine2, id()).WillRepeatedly(Return(2));
    EXPECT_CALL(*mockEngine2, isApplicable(::testing::_, ::testing::_))
        .WillRepeatedly(Return(false));

    EngineManager manager;
    manager.addEngine(std::move(mockEngine1));
    manager.addEngine(std::move(mockEngine2));

    MockGraph mockGraph;
    HipdnnEnginePluginHandle dummyHandle;
    auto applicable = manager.getApplicableEngineIds(dummyHandle, mockGraph);

    EXPECT_EQ(applicable.size(), 1);
    EXPECT_EQ(applicable[0], 1);
}

TEST(TestMiopenEngineManager, ReturnsMultipleApplicableEngineIds)
{
    std::set<std::unique_ptr<IEngine>> engines;

    auto mockEngine1 = std::make_unique<MockEngine>();
    EXPECT_CALL(*mockEngine1, id()).WillRepeatedly(Return(1));
    EXPECT_CALL(*mockEngine1, isApplicable(::testing::_, ::testing::_))
        .WillRepeatedly(Return(true));

    auto mockEngine2 = std::make_unique<MockEngine>();
    EXPECT_CALL(*mockEngine2, id()).WillRepeatedly(Return(2));
    EXPECT_CALL(*mockEngine2, isApplicable(::testing::_, ::testing::_))
        .WillRepeatedly(Return(true));

    EngineManager manager;
    manager.addEngine(std::move(mockEngine1));
    manager.addEngine(std::move(mockEngine2));

    MockGraph mockGraph;
    HipdnnEnginePluginHandle dummyHandle;
    auto applicable = manager.getApplicableEngineIds(dummyHandle, mockGraph);

    EXPECT_EQ(applicable.size(), 2);
    EXPECT_TRUE(std::find(applicable.begin(), applicable.end(), 1) != applicable.end());
    EXPECT_TRUE(std::find(applicable.begin(), applicable.end(), 2) != applicable.end());
}

TEST(TestMiopenEngineManager, ReturnsNoApplicableEngineIds)
{
    std::set<std::unique_ptr<IEngine>> engines;

    auto mockEngine1 = std::make_unique<MockEngine>();
    EXPECT_CALL(*mockEngine1, id()).WillRepeatedly(Return(1));
    EXPECT_CALL(*mockEngine1, isApplicable(::testing::_, ::testing::_))
        .WillRepeatedly(Return(false));

    auto mockEngine2 = std::make_unique<MockEngine>();
    EXPECT_CALL(*mockEngine2, id()).WillRepeatedly(Return(2));
    EXPECT_CALL(*mockEngine2, isApplicable(::testing::_, ::testing::_))
        .WillRepeatedly(Return(false));

    EngineManager manager;
    manager.addEngine(std::move(mockEngine1));
    manager.addEngine(std::move(mockEngine2));

    MockGraph mockGraph;
    HipdnnEnginePluginHandle dummyHandle;
    auto applicable = manager.getApplicableEngineIds(dummyHandle, mockGraph);

    EXPECT_TRUE(applicable.empty());
}

TEST(TestMiopenEngineManager, ReturnsEngineDetails)
{
    EngineManager manager;

    hipdnnPluginConstData_t engineDetails;
    engineDetails.ptr = reinterpret_cast<const void*>(0x12345678);
    engineDetails.size = 200;
    auto mockEngine = std::make_unique<MockEngine>();
    EXPECT_CALL(*mockEngine, id()).WillRepeatedly(Return(1));
    EXPECT_CALL(*mockEngine, getDetails(::testing::_, ::testing::_))
        .WillOnce([&engineDetails](HipdnnEnginePluginHandle& handle, hipdnnPluginConstData_t& out) {
            (void)handle;
            out.ptr = engineDetails.ptr;
            out.size = engineDetails.size;
        });

    manager.addEngine(std::move(mockEngine));

    MockGraph mockGraph;
    HipdnnEnginePluginHandle dummyHandle = {};
    hipdnnPluginConstData_t details;
    manager.getEngineDetails(dummyHandle, mockGraph, 1, details);

    EXPECT_EQ(details.ptr, engineDetails.ptr);
    EXPECT_EQ(details.size, engineDetails.size);
}

TEST(TestMiopenEngineManager, ThrowsOnInvalidEngineId)
{
    EngineManager manager;

    MockGraph mockGraph;
    hipdnnPluginConstData_t engineDetails;

    HipdnnEnginePluginHandle dummyHandle = {};
    EXPECT_THROW(manager.getEngineDetails(dummyHandle, mockGraph, 999, engineDetails),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestMiopenEngineManager, GetWorkspaceSizeReturnsCorrectValue)
{
    EngineManager manager;

    auto mockEngine = std::make_unique<MockEngine>();
    EXPECT_CALL(*mockEngine, id()).WillRepeatedly(Return(42));
    HipdnnEnginePluginHandle dummyHandle = {};
    MockGraph mockGraph;
    EXPECT_CALL(*mockEngine, getWorkspaceSize(::testing::_, ::testing::_)).WillOnce(Return(4096));

    manager.addEngine(std::move(mockEngine));

    size_t workspaceSize = manager.getWorkspaceSize(dummyHandle, 42, mockGraph);
    EXPECT_EQ(workspaceSize, 4096);
}

TEST(TestMiopenEngineManager, GetWorkspaceSizeThrowsOnInvalidEngineId)
{
    EngineManager manager;
    HipdnnEnginePluginHandle dummyHandle = {};
    MockGraph mockGraph;

    EXPECT_THROW(manager.getWorkspaceSize(dummyHandle, 999, mockGraph),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestMiopenEngineManager, InitializeExecutionContextCallsEngine)
{
    auto mockEngine = std::make_unique<MockEngine>();
    EXPECT_CALL(*mockEngine, id()).WillRepeatedly(Return(7));
    EXPECT_CALL(*mockEngine, initializeExecutionContext(::testing::_, ::testing::_, ::testing::_))
        .Times(1);

    EngineManager manager;
    manager.addEngine(std::move(mockEngine));
    HipdnnEnginePluginHandle dummyHandle = {};
    MockGraph mockGraph;
    MockEngineConfig mockEngineConfig;
    ON_CALL(mockEngineConfig, engineId()).WillByDefault(Return(7));
    EXPECT_CALL(mockEngineConfig, engineId()).Times(testing::AnyNumber()); // Uninteresting call
    MockHipdnnEnginePluginExecutionContext execCtx;

    manager.initializeExecutionContext(dummyHandle, mockGraph, mockEngineConfig, execCtx);
}

TEST(TestMiopenEngineManager, InitializeExecutionContextThrowsOnInvalidEngineId)
{
    MockHipdnnEnginePluginExecutionContext execCtx;
    EngineManager manager;
    HipdnnEnginePluginHandle dummyHandle = {};
    MockGraph mockGraph;
    MockEngineConfig mockEngineConfig;

    EXPECT_CALL(mockEngineConfig, engineId()).Times(testing::AnyNumber()); // Uninteresting call
    EXPECT_THROW(
        manager.initializeExecutionContext(dummyHandle, mockGraph, mockEngineConfig, execCtx),
        hipdnn_plugin_sdk::HipdnnPluginException);
}
