// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdlib>          // Standard C library (exit codes, malloc)
#include <iostream>         // C++ I/O streams (cout, cerr)
#include <initializer_list> // C++ initializer list support (unused here)
#include <vector>           // C++ vector container - stores test cases
#include <string>           // String operations
#include <gtest/gtest.h>    // Google Test framework - provides TEST_P, INSTANTIATE_TEST_SUITE_P

#include "profiler/profile_grouped_conv_bwd_data_impl.hpp" // The actual GPU profiler that does convolution work
#include "../common/csv_test_loader.hpp"                   // Shared CSV test case loader

using namespace ck::tensor_layout::convolution; // Import tensor layout names (GNHWK, GKYXC, etc.)

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

// Helper template to run a single backward data convolution test with split_k
template <ck::index_t NDimSpatial,
          typename OutLayout,
          typename WeiLayout,
          typename InLayout,
          typename DataType>
bool RunConvBwdDataTest(const ck::utils::conv::ConvParam& param, ck::index_t split_k)
{
    return ck::profiler::profile_grouped_conv_bwd_data_impl<NDimSpatial,
                                                            OutLayout,
                                                            WeiLayout,
                                                            InLayout,
                                                            DataType,
                                                            DataType,
                                                            DataType>(2,       // do_verification
                                                                      1,       // init_method
                                                                      false,   // do_log
                                                                      false,   // time_kernel
                                                                      param,   // ConvParam
                                                                      split_k, // Split-K value
                                                                      -1);     // instance_index
}

// 2D Tests - GNHWK layout - Float - SplitK=1
class TestGroupedConvndBwdData2dGNHWKFloatSplitK1
    : public ::testing::TestWithParam<ck::utils::conv::ConvParam>
{
};
TEST_P(TestGroupedConvndBwdData2dGNHWKFloatSplitK1, ConvTest)
{
    EXPECT_TRUE((RunConvBwdDataTest<2, GNHWK, GKYXC, GNHWC, float>(GetParam(), 1)));
}
INSTANTIATE_TEST_SUITE_P(Dataset,
                         TestGroupedConvndBwdData2dGNHWKFloatSplitK1,
                         ::testing::ValuesIn(Get2DTestCases()));

// 2D Tests - GNHWK layout - Float - SplitK=2
class TestGroupedConvndBwdData2dGNHWKFloatSplitK2
    : public ::testing::TestWithParam<ck::utils::conv::ConvParam>
{
};
TEST_P(TestGroupedConvndBwdData2dGNHWKFloatSplitK2, ConvTest)
{
    EXPECT_TRUE((RunConvBwdDataTest<2, GNHWK, GKYXC, GNHWC, float>(GetParam(), 2)));
}
INSTANTIATE_TEST_SUITE_P(Dataset,
                         TestGroupedConvndBwdData2dGNHWKFloatSplitK2,
                         ::testing::ValuesIn(Get2DTestCases()));

// 2D Tests - GNHWK layout - Half - SplitK=1
class TestGroupedConvndBwdData2dGNHWKHalfSplitK1
    : public ::testing::TestWithParam<ck::utils::conv::ConvParam>
{
};
TEST_P(TestGroupedConvndBwdData2dGNHWKHalfSplitK1, ConvTest)
{
    EXPECT_TRUE((RunConvBwdDataTest<2, GNHWK, GKYXC, GNHWC, ck::half_t>(GetParam(), 1)));
}
INSTANTIATE_TEST_SUITE_P(Dataset,
                         TestGroupedConvndBwdData2dGNHWKHalfSplitK1,
                         ::testing::ValuesIn(Get2DTestCases()));

// 2D Tests - GNHWK layout - Half - SplitK=2
class TestGroupedConvndBwdData2dGNHWKHalfSplitK2
    : public ::testing::TestWithParam<ck::utils::conv::ConvParam>
{
};
TEST_P(TestGroupedConvndBwdData2dGNHWKHalfSplitK2, ConvTest)
{
    EXPECT_TRUE((RunConvBwdDataTest<2, GNHWK, GKYXC, GNHWC, ck::half_t>(GetParam(), 2)));
}
INSTANTIATE_TEST_SUITE_P(Dataset,
                         TestGroupedConvndBwdData2dGNHWKHalfSplitK2,
                         ::testing::ValuesIn(Get2DTestCases()));

// 2D Tests - GNHWK layout - BFloat16 - SplitK=1
class TestGroupedConvndBwdData2dGNHWKBFloat16SplitK1
    : public ::testing::TestWithParam<ck::utils::conv::ConvParam>
{
};
TEST_P(TestGroupedConvndBwdData2dGNHWKBFloat16SplitK1, ConvTest)
{
    EXPECT_TRUE((RunConvBwdDataTest<2, GNHWK, GKYXC, GNHWC, ck::bhalf_t>(GetParam(), 1)));
}
INSTANTIATE_TEST_SUITE_P(Dataset,
                         TestGroupedConvndBwdData2dGNHWKBFloat16SplitK1,
                         ::testing::ValuesIn(Get2DTestCases()));

// 2D Tests - GNHWK layout - BFloat16 - SplitK=2
class TestGroupedConvndBwdData2dGNHWKBFloat16SplitK2
    : public ::testing::TestWithParam<ck::utils::conv::ConvParam>
{
};
TEST_P(TestGroupedConvndBwdData2dGNHWKBFloat16SplitK2, ConvTest)
{
    EXPECT_TRUE((RunConvBwdDataTest<2, GNHWK, GKYXC, GNHWC, ck::bhalf_t>(GetParam(), 2)));
}
INSTANTIATE_TEST_SUITE_P(Dataset,
                         TestGroupedConvndBwdData2dGNHWKBFloat16SplitK2,
                         ::testing::ValuesIn(Get2DTestCases()));

// 2D Tests - NHWGK layout - Float - SplitK=1
class TestGroupedConvndBwdData2dNHWGKFloatSplitK1
    : public ::testing::TestWithParam<ck::utils::conv::ConvParam>
{
};
TEST_P(TestGroupedConvndBwdData2dNHWGKFloatSplitK1, ConvTest)
{
    EXPECT_TRUE((RunConvBwdDataTest<2, NHWGK, GKYXC, NHWGC, float>(GetParam(), 1)));
}
INSTANTIATE_TEST_SUITE_P(Dataset,
                         TestGroupedConvndBwdData2dNHWGKFloatSplitK1,
                         ::testing::ValuesIn(Get2DTestCases()));

// 2D Tests - NHWGK layout - Float - SplitK=2
class TestGroupedConvndBwdData2dNHWGKFloatSplitK2
    : public ::testing::TestWithParam<ck::utils::conv::ConvParam>
{
};
TEST_P(TestGroupedConvndBwdData2dNHWGKFloatSplitK2, ConvTest)
{
    EXPECT_TRUE((RunConvBwdDataTest<2, NHWGK, GKYXC, NHWGC, float>(GetParam(), 2)));
}
INSTANTIATE_TEST_SUITE_P(Dataset,
                         TestGroupedConvndBwdData2dNHWGKFloatSplitK2,
                         ::testing::ValuesIn(Get2DTestCases()));

// 2D Tests - NHWGK layout - Half - SplitK=1
class TestGroupedConvndBwdData2dNHWGKHalfSplitK1
    : public ::testing::TestWithParam<ck::utils::conv::ConvParam>
{
};
TEST_P(TestGroupedConvndBwdData2dNHWGKHalfSplitK1, ConvTest)
{
    EXPECT_TRUE((RunConvBwdDataTest<2, NHWGK, GKYXC, NHWGC, ck::half_t>(GetParam(), 1)));
}
INSTANTIATE_TEST_SUITE_P(Dataset,
                         TestGroupedConvndBwdData2dNHWGKHalfSplitK1,
                         ::testing::ValuesIn(Get2DTestCases()));

// 2D Tests - NHWGK layout - Half - SplitK=2
class TestGroupedConvndBwdData2dNHWGKHalfSplitK2
    : public ::testing::TestWithParam<ck::utils::conv::ConvParam>
{
};
TEST_P(TestGroupedConvndBwdData2dNHWGKHalfSplitK2, ConvTest)
{
    EXPECT_TRUE((RunConvBwdDataTest<2, NHWGK, GKYXC, NHWGC, ck::half_t>(GetParam(), 2)));
}
INSTANTIATE_TEST_SUITE_P(Dataset,
                         TestGroupedConvndBwdData2dNHWGKHalfSplitK2,
                         ::testing::ValuesIn(Get2DTestCases()));

// 2D Tests - NHWGK layout - BFloat16 - SplitK=1
class TestGroupedConvndBwdData2dNHWGKBFloat16SplitK1
    : public ::testing::TestWithParam<ck::utils::conv::ConvParam>
{
};
TEST_P(TestGroupedConvndBwdData2dNHWGKBFloat16SplitK1, ConvTest)
{
    EXPECT_TRUE((RunConvBwdDataTest<2, NHWGK, GKYXC, NHWGC, ck::bhalf_t>(GetParam(), 1)));
}
INSTANTIATE_TEST_SUITE_P(Dataset,
                         TestGroupedConvndBwdData2dNHWGKBFloat16SplitK1,
                         ::testing::ValuesIn(Get2DTestCases()));

// 2D Tests - NHWGK layout - BFloat16 - SplitK=2
class TestGroupedConvndBwdData2dNHWGKBFloat16SplitK2
    : public ::testing::TestWithParam<ck::utils::conv::ConvParam>
{
};
TEST_P(TestGroupedConvndBwdData2dNHWGKBFloat16SplitK2, ConvTest)
{
    EXPECT_TRUE((RunConvBwdDataTest<2, NHWGK, GKYXC, NHWGC, ck::bhalf_t>(GetParam(), 2)));
}
INSTANTIATE_TEST_SUITE_P(Dataset,
                         TestGroupedConvndBwdData2dNHWGKBFloat16SplitK2,
                         ::testing::ValuesIn(Get2DTestCases()));

// 3D Tests - NDHWGK layout - Float - SplitK=1
class TestGroupedConvndBwdData3dNDHWGKFloatSplitK1
    : public ::testing::TestWithParam<ck::utils::conv::ConvParam>
{
};
TEST_P(TestGroupedConvndBwdData3dNDHWGKFloatSplitK1, ConvTest)
{
    EXPECT_TRUE((RunConvBwdDataTest<3, NDHWGK, GKZYXC, NDHWGC, float>(GetParam(), 1)));
}
INSTANTIATE_TEST_SUITE_P(Dataset,
                         TestGroupedConvndBwdData3dNDHWGKFloatSplitK1,
                         ::testing::ValuesIn(Get3DTestCases()));

// 3D Tests - NDHWGK layout - Float - SplitK=2
class TestGroupedConvndBwdData3dNDHWGKFloatSplitK2
    : public ::testing::TestWithParam<ck::utils::conv::ConvParam>
{
};
TEST_P(TestGroupedConvndBwdData3dNDHWGKFloatSplitK2, ConvTest)
{
    EXPECT_TRUE((RunConvBwdDataTest<3, NDHWGK, GKZYXC, NDHWGC, float>(GetParam(), 2)));
}
INSTANTIATE_TEST_SUITE_P(Dataset,
                         TestGroupedConvndBwdData3dNDHWGKFloatSplitK2,
                         ::testing::ValuesIn(Get3DTestCases()));

// 3D Tests - NDHWGK layout - Half - SplitK=1
class TestGroupedConvndBwdData3dNDHWGKHalfSplitK1
    : public ::testing::TestWithParam<ck::utils::conv::ConvParam>
{
};
TEST_P(TestGroupedConvndBwdData3dNDHWGKHalfSplitK1, ConvTest)
{
    EXPECT_TRUE((RunConvBwdDataTest<3, NDHWGK, GKZYXC, NDHWGC, ck::half_t>(GetParam(), 1)));
}
INSTANTIATE_TEST_SUITE_P(Dataset,
                         TestGroupedConvndBwdData3dNDHWGKHalfSplitK1,
                         ::testing::ValuesIn(Get3DTestCases()));

// 3D Tests - NDHWGK layout - Half - SplitK=2
class TestGroupedConvndBwdData3dNDHWGKHalfSplitK2
    : public ::testing::TestWithParam<ck::utils::conv::ConvParam>
{
};
TEST_P(TestGroupedConvndBwdData3dNDHWGKHalfSplitK2, ConvTest)
{
    EXPECT_TRUE((RunConvBwdDataTest<3, NDHWGK, GKZYXC, NDHWGC, ck::half_t>(GetParam(), 2)));
}
INSTANTIATE_TEST_SUITE_P(Dataset,
                         TestGroupedConvndBwdData3dNDHWGKHalfSplitK2,
                         ::testing::ValuesIn(Get3DTestCases()));

// 3D Tests - NDHWGK layout - BFloat16 - SplitK=1
class TestGroupedConvndBwdData3dNDHWGKBFloat16SplitK1
    : public ::testing::TestWithParam<ck::utils::conv::ConvParam>
{
};
TEST_P(TestGroupedConvndBwdData3dNDHWGKBFloat16SplitK1, ConvTest)
{
    EXPECT_TRUE((RunConvBwdDataTest<3, NDHWGK, GKZYXC, NDHWGC, ck::bhalf_t>(GetParam(), 1)));
}
INSTANTIATE_TEST_SUITE_P(Dataset,
                         TestGroupedConvndBwdData3dNDHWGKBFloat16SplitK1,
                         ::testing::ValuesIn(Get3DTestCases()));

// 3D Tests - NDHWGK layout - BFloat16 - SplitK=2
class TestGroupedConvndBwdData3dNDHWGKBFloat16SplitK2
    : public ::testing::TestWithParam<ck::utils::conv::ConvParam>
{
};
TEST_P(TestGroupedConvndBwdData3dNDHWGKBFloat16SplitK2, ConvTest)
{
    EXPECT_TRUE((RunConvBwdDataTest<3, NDHWGK, GKZYXC, NDHWGC, ck::bhalf_t>(GetParam(), 2)));
}
INSTANTIATE_TEST_SUITE_P(Dataset,
                         TestGroupedConvndBwdData3dNDHWGKBFloat16SplitK2,
                         ::testing::ValuesIn(Get3DTestCases()));
