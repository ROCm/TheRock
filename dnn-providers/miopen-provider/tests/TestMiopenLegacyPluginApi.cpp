// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipdnn_plugin_sdk/PluginApi.h"
#include <MiopenLegacyPlugin.hpp>
#include <array>
#include <gtest/gtest.h>
#include <iostream>

namespace
{
void testLoggingCallback(hipdnnSeverity_t severity, const char* msg)
{
    (void)severity;
    // std::cout << msg << "\n"; // uncomment to see formatted log messages during tests.
    // It does not use the true callback yet since the plugin is not yet loaded.
    (void)msg;
}

class TestMiopenLegacyPluginApi : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ASSERT_EQ(hipdnnPluginSetLoggingCallbackImpl(testLoggingCallback),
                  HIPDNN_PLUGIN_STATUS_SUCCESS);
    }
};
} // namespace

TEST_F(TestMiopenLegacyPluginApi, GetNameSuccess)
{
    const char* name = nullptr;
    EXPECT_EQ(hipdnnPluginGetNameImpl(&name), HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_STREQ(name, "miopen_legacy_plugin");
}

TEST_F(TestMiopenLegacyPluginApi, GetNameNullptr)
{
    EXPECT_EQ(hipdnnPluginGetNameImpl(nullptr), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST_F(TestMiopenLegacyPluginApi, GetVersionSuccess)
{
    const char* version = nullptr;
    EXPECT_EQ(hipdnnPluginGetVersionImpl(&version), HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_STREQ(version, "1.0.0");
}

TEST_F(TestMiopenLegacyPluginApi, GetVersionNullptr)
{
    EXPECT_EQ(hipdnnPluginGetVersionImpl(nullptr), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST_F(TestMiopenLegacyPluginApi, GetTypeSuccess)
{
    hipdnnPluginType_t type;
    EXPECT_EQ(hipdnnPluginGetTypeImpl(&type), HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(type, HIPDNN_PLUGIN_TYPE_ENGINE);
}

TEST_F(TestMiopenLegacyPluginApi, GetTypeNullptr)
{
    EXPECT_EQ(hipdnnPluginGetTypeImpl(nullptr), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST_F(TestMiopenLegacyPluginApi, GetLastErrorStringSuccess)
{
    const char* errorStr = nullptr;
    hipdnnPluginGetLastErrorStringImpl(&errorStr);
    ASSERT_NE(errorStr, nullptr);
    EXPECT_GE(strlen(errorStr), 0u);
}

TEST_F(TestMiopenLegacyPluginApi, GetLastErrorStringNullptr)
{
    EXPECT_NO_THROW(hipdnnPluginGetLastErrorStringImpl(nullptr));
}
