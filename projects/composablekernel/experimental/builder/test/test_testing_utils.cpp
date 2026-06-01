// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <ck/library/tensor_operation_instance/gpu/grouped_convolution_forward.hpp>

#include "testing_utils.hpp"

using ck_tile::test::HipError;
using ck_tile::test::HipSuccess;
using ck_tile::test::InstanceMatcher;
using ck_tile::test::InstanceSet;
using ck_tile::test::StringEqWithDiff;

TEST(InstanceSet, FromFactory)
{
    using DeviceOp = ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<
        2,                                                   // NDimSpatial
        ck::tensor_operation::device::instance::NHWGC,       // InLayout
        ck::tensor_operation::device::instance::GKYXC,       // WeiLayout
        ck::tensor_operation::device::instance::Empty_Tuple, // DsLayout
        ck::tensor_operation::device::instance::NHWGK,       // OutLayout
        ck::half_t,                                          // ADataType
        ck::half_t,                                          // BDataType
        ck::Tuple<>,                                         // DsDataType
        ck::half_t,                                          // EDataType
        ck::tensor_operation::element_wise::PassThrough,     // AElementwiseOperation
        ck::tensor_operation::element_wise::PassThrough,     // BElementwiseOperation
        ck::tensor_operation::element_wise::PassThrough,     // CDEElementwiseOperation
        ck::half_t,                                          // AComputeType
        ck::half_t>;                                         // BComputeType

    const auto instances = InstanceSet::from_factory<DeviceOp>();

    EXPECT_THAT(instances.instances.size(), testing::Gt(0));

    const auto* el =
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<2,NHWGC,GKYXC,EmptyTuple,NHWGK,fp16,fp16,"
        "fp32,fp16,EmptyTuple,fp16,PassThrough,PassThrough,PassThrough,Default,MNKPadding,1,128,"
        "128,128,32,8,8,32,32,4,2,Seq(4,32,1),Seq(1,0,2),Seq(1,0,2),2,8,8,true,Seq(4,32,1),"
        "Seq(1,0,2),Seq(1,0,2),2,8,8,true,1,1,Seq(1,16,1,8),8,fp16,fp16,Default,1>";
    EXPECT_THAT(instances.instances, testing::Contains(el));
}

TEST(InstanceMatcher, Basic)
{
    auto a = InstanceSet{
        "python",
        "cobra",
        "boa",
    };

    auto b = InstanceSet{
        "cobra",
        "boa",
        "python",
    };

    auto c = InstanceSet{
        "adder",
        "boa",
        "cobra",
    };

    auto d = InstanceSet{
        "boa",
        "python",
    };

    EXPECT_THAT(a, InstancesMatch(b));
    EXPECT_THAT(c, Not(InstancesMatch(b)));
    EXPECT_THAT(a, Not(InstancesMatch(d)));
    EXPECT_THAT(d, Not(InstancesMatch(b)));
}

TEST(InstanceMatcher, ExplainMatchResult)
{
    auto actual = InstanceSet{
        "python",
        "cobra",
        "boa",
    };

    auto expected = InstanceSet{
        "adder",
        "boa",
        "cobra",
        "rattlesnake",
    };

    testing::StringMatchResultListener listener;
    EXPECT_TRUE(!ExplainMatchResult(InstancesMatch(expected), actual, &listener));

    EXPECT_THAT(listener.str(),
                StringEqWithDiff("\n"
                                 " Missing: 2\n"
                                 "- adder\n"
                                 "- rattlesnake\n"
                                 "Unexpected: 1\n"
                                 "- python\n"));
}

TEST(HipStatusMatcher, Basic)
{
    EXPECT_THAT(hipSuccess, HipSuccess());
    EXPECT_THAT(hipErrorInvalidValue, HipError(hipErrorInvalidValue));
    EXPECT_THAT(hipErrorInvalidValue, Not(HipSuccess()));
    EXPECT_THAT(hipSuccess, Not(HipError(hipErrorInvalidValue)));
    EXPECT_THAT(hipErrorOutOfMemory, Not(HipError(hipErrorInvalidValue)));
}
