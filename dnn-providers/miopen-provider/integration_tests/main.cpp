/*
Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
*/

#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/utilities/HipErrorHandler.hpp>

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    hipdnn_frontend::initializeFrontendLogging();

    // Register HipErrorHandler to check and clear HIP errors after each test
    testing::TestEventListeners& listeners = testing::UnitTest::GetInstance()->listeners();
    listeners.Append(new hipdnn_test_sdk::utilities::HipErrorHandler);

    auto result = RUN_ALL_TESTS();
    spdlog::shutdown();
    return result;
}
