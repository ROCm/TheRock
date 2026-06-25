// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdlib>          // Standard C library (exit codes, malloc)
#include <iostream>         // C++ I/O streams (cout, cerr)
#include <initializer_list> // C++ initializer list support (unused here)
#include <vector>           // C++ vector container - stores test cases
#include <string>           // String operations
#include <gtest/gtest.h>    // Google Test framework - provides TEST_P, INSTANTIATE_TEST_SUITE_P

#include "ck/utility/common_header.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/host_utility/device_prop.hpp"

#include "profiler/profile_grouped_conv_bwd_weight_impl.hpp" // The actual GPU profiler that does convolution work
#include "../common/csv_test_loader.hpp"                     // Shared CSV test case loader

using namespace ck::tensor_layout::convolution;

// Load CSV data for 2D tests
static std::vector<ck::utils::conv::ConvParam> Get2DTestCases()
{
    static std::vector<ck::utils::conv::ConvParam> test_cases;
    if(test_cases.empty())
    {
        std::string test_data_dir = ck::test::GetTestDataPath();
        if(test_data_dir.empty())
        {
            std::cerr << "FATAL: test_data directory not found" << std::endl;
            return test_cases;
        }

        std::vector<std::string> csv_paths = {test_data_dir + "/conv_test_set_2d_dataset.csv"};
        bool loaded = ck::test::load_and_populate_test_cases(csv_paths, test_cases, "2D");
        if(!loaded)
        {
            std::cerr << "FATAL: Failed to load 2D test cases from " << csv_paths[0] << std::endl;
        }
    }
    return test_cases;
}

// Load CSV data for 3D tests
static std::vector<ck::utils::conv::ConvParam> Get3DTestCases()
{
    static std::vector<ck::utils::conv::ConvParam> test_cases;
    if(test_cases.empty())
    {
        std::string test_data_dir = ck::test::GetTestDataPath();
        if(test_data_dir.empty())
        {
            std::cerr << "FATAL: test_data directory not found" << std::endl;
            return test_cases;
        }

        std::vector<std::string> csv_paths = {test_data_dir + "/conv_test_set_3d_dataset.csv"};
        bool loaded = ck::test::load_and_populate_test_cases(csv_paths, test_cases, "3D");
        if(!loaded)
        {
            std::cerr << "FATAL: Failed to load 3D test cases from " << csv_paths[0] << std::endl;
        }
    }
    return test_cases;
}

// Helper template to run a single backward weight convolution test
template <ck::index_t NDimSpatial,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType>
bool RunConvBwdWeightTest(const ck::utils::conv::ConvParam& param, ck::index_t split_k)
{
    return ck::profiler::profile_grouped_conv_bwd_weight_impl<NDimSpatial,
                                                              InLayout,
                                                              WeiLayout,
                                                              OutLayout,
                                                              InDataType,
                                                              WeiDataType,
                                                              OutDataType>(
        2,                       // do_verification
        1,                       // init_method
        false,                   // do_log
        false,                   // time_kernel
        param,                   // ConvParam
        std::to_string(split_k), // Split-K value as string
        -1);                     // instance_index
}

// 2D Tests - NHWGK layout - Float - SplitK=1
class TestGroupedConvndBwdWeight2dNHWGKFloatSplitK1
    : public ::testing::TestWithParam<ck::utils::conv::ConvParam>
{
};
TEST_P(TestGroupedConvndBwdWeight2dNHWGKFloatSplitK1, ConvTest)
{
    EXPECT_TRUE((RunConvBwdWeightTest<2, NHWGC, GKYXC, NHWGK, float, float, float>(GetParam(), 1)));
}
INSTANTIATE_TEST_SUITE_P(Dataset,
                         TestGroupedConvndBwdWeight2dNHWGKFloatSplitK1,
                         ::testing::ValuesIn(Get2DTestCases()));

// 2D Tests - NHWGK layout - Float - SplitK=2
class TestGroupedConvndBwdWeight2dNHWGKFloatSplitK2
    : public ::testing::TestWithParam<ck::utils::conv::ConvParam>
{
};
TEST_P(TestGroupedConvndBwdWeight2dNHWGKFloatSplitK2, ConvTest)
{
    EXPECT_TRUE((RunConvBwdWeightTest<2, NHWGC, GKYXC, NHWGK, float, float, float>(GetParam(), 2)));
}
INSTANTIATE_TEST_SUITE_P(Dataset,
                         TestGroupedConvndBwdWeight2dNHWGKFloatSplitK2,
                         ::testing::ValuesIn(Get2DTestCases()));

// 2D Tests - NHWGK layout - Half - SplitK=1
class TestGroupedConvndBwdWeight2dNHWGKHalfSplitK1
    : public ::testing::TestWithParam<ck::utils::conv::ConvParam>
{
};
TEST_P(TestGroupedConvndBwdWeight2dNHWGKHalfSplitK1, ConvTest)
{
    EXPECT_TRUE((RunConvBwdWeightTest<2, NHWGC, GKYXC, NHWGK, ck::half_t, ck::half_t, ck::half_t>(
        GetParam(), 1)));
}
INSTANTIATE_TEST_SUITE_P(Dataset,
                         TestGroupedConvndBwdWeight2dNHWGKHalfSplitK1,
                         ::testing::ValuesIn(Get2DTestCases()));

// 2D Tests - NHWGK layout - Half - SplitK=2
class TestGroupedConvndBwdWeight2dNHWGKHalfSplitK2
    : public ::testing::TestWithParam<ck::utils::conv::ConvParam>
{
};
TEST_P(TestGroupedConvndBwdWeight2dNHWGKHalfSplitK2, ConvTest)
{
    EXPECT_TRUE((RunConvBwdWeightTest<2, NHWGC, GKYXC, NHWGK, ck::half_t, ck::half_t, ck::half_t>(
        GetParam(), 2)));
}
INSTANTIATE_TEST_SUITE_P(Dataset,
                         TestGroupedConvndBwdWeight2dNHWGKHalfSplitK2,
                         ::testing::ValuesIn(Get2DTestCases()));

// 2D Tests - NHWGK layout - BFloat16 - SplitK=1
class TestGroupedConvndBwdWeight2dNHWGKBFloat16SplitK1
    : public ::testing::TestWithParam<ck::utils::conv::ConvParam>
{
};
TEST_P(TestGroupedConvndBwdWeight2dNHWGKBFloat16SplitK1, ConvTest)
{
    EXPECT_TRUE((RunConvBwdWeightTest<2, NHWGC, GKYXC, NHWGK, ck::bhalf_t, float, ck::bhalf_t>(
        GetParam(), 1)));
}
INSTANTIATE_TEST_SUITE_P(Dataset,
                         TestGroupedConvndBwdWeight2dNHWGKBFloat16SplitK1,
                         ::testing::ValuesIn(Get2DTestCases()));

// 2D Tests - NHWGK layout - BFloat16 - SplitK=2
class TestGroupedConvndBwdWeight2dNHWGKBFloat16SplitK2
    : public ::testing::TestWithParam<ck::utils::conv::ConvParam>
{
};
TEST_P(TestGroupedConvndBwdWeight2dNHWGKBFloat16SplitK2, ConvTest)
{
    EXPECT_TRUE((RunConvBwdWeightTest<2, NHWGC, GKYXC, NHWGK, ck::bhalf_t, float, ck::bhalf_t>(
        GetParam(), 2)));
}
INSTANTIATE_TEST_SUITE_P(Dataset,
                         TestGroupedConvndBwdWeight2dNHWGKBFloat16SplitK2,
                         ::testing::ValuesIn(Get2DTestCases()));

// 3D Tests - NDHWGK layout - Float - SplitK=1
class TestGroupedConvndBwdWeight3dNDHWGKFloatSplitK1
    : public ::testing::TestWithParam<ck::utils::conv::ConvParam>
{
};
TEST_P(TestGroupedConvndBwdWeight3dNDHWGKFloatSplitK1, ConvTest)
{
    EXPECT_TRUE(
        (RunConvBwdWeightTest<3, NDHWGC, GKZYXC, NDHWGK, float, float, float>(GetParam(), 1)));
}
INSTANTIATE_TEST_SUITE_P(Dataset,
                         TestGroupedConvndBwdWeight3dNDHWGKFloatSplitK1,
                         ::testing::ValuesIn(Get3DTestCases()));

// 3D Tests - NDHWGK layout - Float - SplitK=2
class TestGroupedConvndBwdWeight3dNDHWGKFloatSplitK2
    : public ::testing::TestWithParam<ck::utils::conv::ConvParam>
{
};
TEST_P(TestGroupedConvndBwdWeight3dNDHWGKFloatSplitK2, ConvTest)
{
    EXPECT_TRUE(
        (RunConvBwdWeightTest<3, NDHWGC, GKZYXC, NDHWGK, float, float, float>(GetParam(), 2)));
}
INSTANTIATE_TEST_SUITE_P(Dataset,
                         TestGroupedConvndBwdWeight3dNDHWGKFloatSplitK2,
                         ::testing::ValuesIn(Get3DTestCases()));

// 3D Tests - NDHWGK layout - Half - SplitK=1
class TestGroupedConvndBwdWeight3dNDHWGKHalfSplitK1
    : public ::testing::TestWithParam<ck::utils::conv::ConvParam>
{
};
TEST_P(TestGroupedConvndBwdWeight3dNDHWGKHalfSplitK1, ConvTest)
{
    EXPECT_TRUE(
        (RunConvBwdWeightTest<3, NDHWGC, GKZYXC, NDHWGK, ck::half_t, ck::half_t, ck::half_t>(
            GetParam(), 1)));
}
INSTANTIATE_TEST_SUITE_P(Dataset,
                         TestGroupedConvndBwdWeight3dNDHWGKHalfSplitK1,
                         ::testing::ValuesIn(Get3DTestCases()));

// 3D Tests - NDHWGK layout - Half - SplitK=2
class TestGroupedConvndBwdWeight3dNDHWGKHalfSplitK2
    : public ::testing::TestWithParam<ck::utils::conv::ConvParam>
{
};
TEST_P(TestGroupedConvndBwdWeight3dNDHWGKHalfSplitK2, ConvTest)
{
    EXPECT_TRUE(
        (RunConvBwdWeightTest<3, NDHWGC, GKZYXC, NDHWGK, ck::half_t, ck::half_t, ck::half_t>(
            GetParam(), 2)));
}
INSTANTIATE_TEST_SUITE_P(Dataset,
                         TestGroupedConvndBwdWeight3dNDHWGKHalfSplitK2,
                         ::testing::ValuesIn(Get3DTestCases()));

// 3D Tests - NDHWGK layout - BFloat16 - SplitK=1
class TestGroupedConvndBwdWeight3dNDHWGKBFloat16SplitK1
    : public ::testing::TestWithParam<ck::utils::conv::ConvParam>
{
};
TEST_P(TestGroupedConvndBwdWeight3dNDHWGKBFloat16SplitK1, ConvTest)
{
    EXPECT_TRUE((RunConvBwdWeightTest<3, NDHWGC, GKZYXC, NDHWGK, ck::bhalf_t, float, ck::bhalf_t>(
        GetParam(), 1)));
}
INSTANTIATE_TEST_SUITE_P(Dataset,
                         TestGroupedConvndBwdWeight3dNDHWGKBFloat16SplitK1,
                         ::testing::ValuesIn(Get3DTestCases()));

// 3D Tests - NDHWGK layout - BFloat16 - SplitK=2
class TestGroupedConvndBwdWeight3dNDHWGKBFloat16SplitK2
    : public ::testing::TestWithParam<ck::utils::conv::ConvParam>
{
};
TEST_P(TestGroupedConvndBwdWeight3dNDHWGKBFloat16SplitK2, ConvTest)
{
    EXPECT_TRUE((RunConvBwdWeightTest<3, NDHWGC, GKZYXC, NDHWGK, ck::bhalf_t, float, ck::bhalf_t>(
        GetParam(), 2)));
}
INSTANTIATE_TEST_SUITE_P(Dataset,
                         TestGroupedConvndBwdWeight3dNDHWGKBFloat16SplitK2,
                         ::testing::ValuesIn(Get3DTestCases()));
