// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdlib>          // Standard C library (exit codes, malloc)
#include <iostream>         // C++ I/O streams (cout, cerr)
#include <initializer_list> // C++ initializer list support (unused here)
#include <vector>           // C++ vector container - stores test cases
#include <string>           // String operations
#include <gtest/gtest.h>    // Google Test framework - provides TEST_P, INSTANTIATE_TEST_SUITE_P

#include "profiler/profile_grouped_conv_fwd_impl.hpp" // The actual GPU profiler that does convolution work
#include "../common/csv_test_loader.hpp"              // Shared CSV test case loader

using namespace ck::tensor_layout::convolution; // Import tensor layout names (NHWGC, GKYXC, etc.)

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

// Helper template to run a single convolution test
template <ck::index_t NDimSpatial,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout,
          typename DataType>
bool RunConvTest(const ck::utils::conv::ConvParam& param)
{
    using IndexType = ck::long_index_t;
    return ck::profiler::profile_grouped_conv_fwd_impl<NDimSpatial,
                                                       InLayout,
                                                       WeiLayout,
                                                       OutLayout,
                                                       DataType,
                                                       DataType,
                                                       DataType,
                                                       DataType,
                                                       DataType,
                                                       IndexType>(2,     // do_verification
                                                                  1,     // init_method
                                                                  false, // do_log
                                                                  false, // time_kernel
                                                                  param);
}

// 2D Tests - Float
class TestGroupedConvndFwd2dFloat : public ::testing::TestWithParam<ck::utils::conv::ConvParam>
{
};
TEST_P(TestGroupedConvndFwd2dFloat, ConvTest)
{
    EXPECT_TRUE((RunConvTest<2, NHWGC, GKYXC, NHWGK, float>(GetParam())));
}
INSTANTIATE_TEST_SUITE_P(Dataset,
                         TestGroupedConvndFwd2dFloat,
                         ::testing::ValuesIn(Get2DTestCases()));

// 2D Tests - Half
class TestGroupedConvndFwd2dHalf : public ::testing::TestWithParam<ck::utils::conv::ConvParam>
{
};
TEST_P(TestGroupedConvndFwd2dHalf, ConvTest)
{
    EXPECT_TRUE((RunConvTest<2, NHWGC, GKYXC, NHWGK, ck::half_t>(GetParam())));
}
INSTANTIATE_TEST_SUITE_P(Dataset,
                         TestGroupedConvndFwd2dHalf,
                         ::testing::ValuesIn(Get2DTestCases()));

// 2D Tests - BFloat16
class TestGroupedConvndFwd2dBFloat16 : public ::testing::TestWithParam<ck::utils::conv::ConvParam>
{
};
TEST_P(TestGroupedConvndFwd2dBFloat16, ConvTest)
{
    EXPECT_TRUE((RunConvTest<2, NHWGC, GKYXC, NHWGK, ck::bhalf_t>(GetParam())));
}
INSTANTIATE_TEST_SUITE_P(Dataset,
                         TestGroupedConvndFwd2dBFloat16,
                         ::testing::ValuesIn(Get2DTestCases()));

// 2D Tests - Int8
class TestGroupedConvndFwd2dInt8 : public ::testing::TestWithParam<ck::utils::conv::ConvParam>
{
};
TEST_P(TestGroupedConvndFwd2dInt8, ConvTest)
{
    EXPECT_TRUE((RunConvTest<2, NHWGC, GKYXC, NHWGK, int8_t>(GetParam())));
}
INSTANTIATE_TEST_SUITE_P(Dataset,
                         TestGroupedConvndFwd2dInt8,
                         ::testing::ValuesIn(Get2DTestCases()));

// 3D Tests - Float
class TestGroupedConvndFwd3dFloat : public ::testing::TestWithParam<ck::utils::conv::ConvParam>
{
};
TEST_P(TestGroupedConvndFwd3dFloat, ConvTest)
{
    EXPECT_TRUE((RunConvTest<3, NDHWGC, GKZYXC, NDHWGK, float>(GetParam())));
}
INSTANTIATE_TEST_SUITE_P(Dataset,
                         TestGroupedConvndFwd3dFloat,
                         ::testing::ValuesIn(Get3DTestCases()));

// 3D Tests - Half
class TestGroupedConvndFwd3dHalf : public ::testing::TestWithParam<ck::utils::conv::ConvParam>
{
};
TEST_P(TestGroupedConvndFwd3dHalf, ConvTest)
{
    EXPECT_TRUE((RunConvTest<3, NDHWGC, GKZYXC, NDHWGK, ck::half_t>(GetParam())));
}
INSTANTIATE_TEST_SUITE_P(Dataset,
                         TestGroupedConvndFwd3dHalf,
                         ::testing::ValuesIn(Get3DTestCases()));

// 3D Tests - BFloat16
class TestGroupedConvndFwd3dBFloat16 : public ::testing::TestWithParam<ck::utils::conv::ConvParam>
{
};
TEST_P(TestGroupedConvndFwd3dBFloat16, ConvTest)
{
    EXPECT_TRUE((RunConvTest<3, NDHWGC, GKZYXC, NDHWGK, ck::bhalf_t>(GetParam())));
}
INSTANTIATE_TEST_SUITE_P(Dataset,
                         TestGroupedConvndFwd3dBFloat16,
                         ::testing::ValuesIn(Get3DTestCases()));
