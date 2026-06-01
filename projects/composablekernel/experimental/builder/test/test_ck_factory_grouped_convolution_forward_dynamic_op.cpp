// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_dynamic_op.hpp>
#include "ck/utility/data_type.hpp"
#include "testing_utils.hpp"

using ck_tile::test::InstanceSet;
using ck_tile::test::InstancesMatch;

namespace {

using InLayout  = ck::tensor_layout::convolution::NDHWGC;
using WeiLayout = ck::tensor_layout::convolution::GKZYXC;
using OutLayout = ck::tensor_layout::convolution::NDHWGK;

using ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD;
using ck::tensor_operation::element_wise::DynamicUnaryOp;
using ck::tensor_operation::element_wise::PassThrough;

template <ck::index_t NumDimSpatial, typename T>
struct DeviceOpHelper;

template <typename T>
struct DeviceOpHelper<2, T>
{
    using InLayout  = ck::tensor_layout::convolution::NHWGC;
    using WeiLayout = ck::tensor_layout::convolution::GKYXC;
    using OutLayout = ck::tensor_layout::convolution::NHWGK;

    using Type = DeviceGroupedConvFwdMultipleABD<2,
                                                 InLayout,
                                                 WeiLayout,
                                                 ck::Tuple<>, // DsLayout
                                                 OutLayout,
                                                 T,           // InDataType
                                                 T,           // WeiDataType
                                                 ck::Tuple<>, // DsDataType
                                                 T,           // OutDataType
                                                 PassThrough,
                                                 PassThrough,
                                                 DynamicUnaryOp>;
};

template <typename T>
struct DeviceOpHelper<3, T>
{
    using InLayout  = ck::tensor_layout::convolution::NDHWGC;
    using WeiLayout = ck::tensor_layout::convolution::GKZYXC;
    using OutLayout = ck::tensor_layout::convolution::NDHWGK;

    using Type = DeviceGroupedConvFwdMultipleABD<3,
                                                 InLayout,
                                                 WeiLayout,
                                                 ck::Tuple<>, // DsLayout
                                                 OutLayout,
                                                 T,           // InDataType
                                                 T,           // WeiDataType
                                                 ck::Tuple<>, // DsDataType
                                                 T,           // OutDataType
                                                 PassThrough,
                                                 PassThrough,
                                                 DynamicUnaryOp>;
};

template <ck::index_t NumDimSpatial, typename T>
using DeviceOp = DeviceOpHelper<NumDimSpatial, T>::Type;

} // namespace

template <typename Case>
struct CkFactoryTestBilinearFwd : public testing::Test
{
    static auto get_actual_instances()
    {
        return InstanceSet::from_factory<typename Case::DeviceOp>();
    }

    static auto get_expected_instances() { return InstanceSet(Case::expected); }
};

struct DyOp_F32_2
{
    using DeviceOp = ::DeviceOp<2, float>;

    constexpr static auto expected = {
        // clang-format off
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<2,NHWGC,GKYXC,EmptyTuple,NHWGK,fp32,fp32,fp32,fp32,EmptyTuple,fp32,PassThrough,PassThrough,DynamicUnaryOp,Default,MNKPadding,1,256,128,128,16,4,4,32,32,2,2,Seq(4,64,1),Seq(1,0,2),Seq(1,0,2),2,1,4,true,Seq(4,64,1),Seq(1,0,2),Seq(1,0,2),2,1,4,true,1,1,Seq(1,16,1,16),4,fp32,fp32,Default,1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<2,NHWGC,GKYXC,EmptyTuple,NHWGK,fp32,fp32,fp32,fp32,EmptyTuple,fp32,PassThrough,PassThrough,DynamicUnaryOp,Default,MNKPadding,1,64,64,32,16,4,4,32,32,2,1,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,4,4,true,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,4,4,true,1,1,Seq(1,8,1,8),1,fp32,fp32,Default,1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<2,NHWGC,GKYXC,EmptyTuple,NHWGK,fp32,fp32,fp32,fp32,EmptyTuple,fp32,PassThrough,PassThrough,DynamicUnaryOp,Default,MNKPadding,1,64,64,64,16,4,4,32,32,2,2,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,1,4,true,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,1,4,true,1,1,Seq(1,8,1,8),1,fp32,fp32,Default,1>"
        // clang-format on
    };
};

struct DyOp_F32_3
{
    using DeviceOp = ::DeviceOp<3, float>;

    constexpr static auto expected = {
        // clang-format off
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<3,NDHWGC,GKZYXC,EmptyTuple,NDHWGK,fp32,fp32,fp32,fp32,EmptyTuple,fp32,PassThrough,PassThrough,DynamicUnaryOp,Default,MNKPadding,1,256,128,128,16,4,4,32,32,2,2,Seq(4,64,1),Seq(1,0,2),Seq(1,0,2),2,1,4,true,Seq(4,64,1),Seq(1,0,2),Seq(1,0,2),2,1,4,true,1,1,Seq(1,16,1,16),4,fp32,fp32,Default,1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<3,NDHWGC,GKZYXC,EmptyTuple,NDHWGK,fp32,fp32,fp32,fp32,EmptyTuple,fp32,PassThrough,PassThrough,DynamicUnaryOp,Default,MNKPadding,1,64,64,32,16,4,4,32,32,2,1,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,4,4,true,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,4,4,true,1,1,Seq(1,8,1,8),1,fp32,fp32,Default,1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<3,NDHWGC,GKZYXC,EmptyTuple,NDHWGK,fp32,fp32,fp32,fp32,EmptyTuple,fp32,PassThrough,PassThrough,DynamicUnaryOp,Default,MNKPadding,1,64,64,64,16,4,4,32,32,2,2,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,1,4,true,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,1,4,true,1,1,Seq(1,8,1,8),1,fp32,fp32,Default,1>"
        // clang-format on
    };
};

struct DyOp_F16_2
{
    using DeviceOp = ::DeviceOp<2, ck::half_t>;

    constexpr static auto expected = {
        // clang-format off
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<2,NHWGC,GKYXC,EmptyTuple,NHWGK,fp16,fp16,fp32,fp16,EmptyTuple,fp16,PassThrough,PassThrough,DynamicUnaryOp,Default,MNKPadding,1,256,128,128,32,8,8,32,32,2,2,Seq(4,64,1),Seq(1,0,2),Seq(1,0,2),2,1,8,true,Seq(4,64,1),Seq(1,0,2),Seq(1,0,2),2,1,8,true,1,1,Seq(1,32,1,8),8,fp16,fp16,Default,1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<2,NHWGC,GKYXC,EmptyTuple,NHWGK,fp16,fp16,fp32,fp16,EmptyTuple,fp16,PassThrough,PassThrough,DynamicUnaryOp,Default,MNKPadding,1,64,64,32,32,8,8,32,32,2,1,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,8,8,true,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,8,8,true,1,1,Seq(1,16,1,4),1,fp16,fp16,Default,1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<2,NHWGC,GKYXC,EmptyTuple,NHWGK,fp16,fp16,fp32,fp16,EmptyTuple,fp16,PassThrough,PassThrough,DynamicUnaryOp,Default,MNKPadding,1,64,64,64,32,8,8,32,32,2,2,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,1,8,true,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,1,8,true,1,1,Seq(1,16,1,4),1,fp16,fp16,Default,1>"
        // clang-format on
    };
};

struct DyOp_F16_3
{
    using DeviceOp = ::DeviceOp<3, ck::half_t>;

    constexpr static auto expected = {
        // clang-format off
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<3,NDHWGC,GKZYXC,EmptyTuple,NDHWGK,fp16,fp16,fp32,fp16,EmptyTuple,fp16,PassThrough,PassThrough,DynamicUnaryOp,Default,MNKPadding,1,256,128,128,32,8,8,32,32,2,2,Seq(4,64,1),Seq(1,0,2),Seq(1,0,2),2,1,8,true,Seq(4,64,1),Seq(1,0,2),Seq(1,0,2),2,1,8,true,1,1,Seq(1,32,1,8),8,fp16,fp16,Default,1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<3,NDHWGC,GKZYXC,EmptyTuple,NDHWGK,fp16,fp16,fp32,fp16,EmptyTuple,fp16,PassThrough,PassThrough,DynamicUnaryOp,Default,MNKPadding,1,64,64,32,32,8,8,32,32,2,1,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,8,8,true,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,8,8,true,1,1,Seq(1,16,1,4),1,fp16,fp16,Default,1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<3,NDHWGC,GKZYXC,EmptyTuple,NDHWGK,fp16,fp16,fp32,fp16,EmptyTuple,fp16,PassThrough,PassThrough,DynamicUnaryOp,Default,MNKPadding,1,64,64,64,32,8,8,32,32,2,2,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,1,8,true,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,1,8,true,1,1,Seq(1,16,1,4),1,fp16,fp16,Default,1>"
        // clang-format on
    };
};

struct DyOp_BF16_2
{
    using DeviceOp = ::DeviceOp<2, ck::bhalf_t>;

    constexpr static auto expected = {
        // clang-format off
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<2,NHWGC,GKYXC,EmptyTuple,NHWGK,bf16,bf16,fp32,bf16,EmptyTuple,bf16,PassThrough,PassThrough,DynamicUnaryOp,Default,MNKPadding,1,256,128,128,32,8,8,32,32,2,2,Seq(4,64,1),Seq(1,0,2),Seq(1,0,2),2,1,8,true,Seq(4,64,1),Seq(1,0,2),Seq(1,0,2),2,1,8,true,1,1,Seq(1,32,1,8),8,bf16,bf16,Default,1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<2,NHWGC,GKYXC,EmptyTuple,NHWGK,bf16,bf16,fp32,bf16,EmptyTuple,bf16,PassThrough,PassThrough,DynamicUnaryOp,Default,MNKPadding,1,64,64,32,32,8,8,32,32,2,1,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,8,8,true,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,8,8,true,1,1,Seq(1,16,1,4),1,bf16,bf16,Default,1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<2,NHWGC,GKYXC,EmptyTuple,NHWGK,bf16,bf16,fp32,bf16,EmptyTuple,bf16,PassThrough,PassThrough,DynamicUnaryOp,Default,MNKPadding,1,64,64,64,32,8,8,32,32,2,2,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,1,8,true,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,1,8,true,1,1,Seq(1,16,1,4),1,bf16,bf16,Default,1>"
        // clang-format on
    };
};

struct DyOp_BF16_3
{
    using DeviceOp = ::DeviceOp<3, ck::bhalf_t>;

    constexpr static auto expected = {
        // clang-format off
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<3,NDHWGC,GKZYXC,EmptyTuple,NDHWGK,bf16,bf16,fp32,bf16,EmptyTuple,bf16,PassThrough,PassThrough,DynamicUnaryOp,Default,MNKPadding,1,256,128,128,32,8,8,32,32,2,2,Seq(4,64,1),Seq(1,0,2),Seq(1,0,2),2,1,8,true,Seq(4,64,1),Seq(1,0,2),Seq(1,0,2),2,1,8,true,1,1,Seq(1,32,1,8),8,bf16,bf16,Default,1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<3,NDHWGC,GKZYXC,EmptyTuple,NDHWGK,bf16,bf16,fp32,bf16,EmptyTuple,bf16,PassThrough,PassThrough,DynamicUnaryOp,Default,MNKPadding,1,64,64,32,32,8,8,32,32,2,1,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,8,8,true,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,8,8,true,1,1,Seq(1,16,1,4),1,bf16,bf16,Default,1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<3,NDHWGC,GKZYXC,EmptyTuple,NDHWGK,bf16,bf16,fp32,bf16,EmptyTuple,bf16,PassThrough,PassThrough,DynamicUnaryOp,Default,MNKPadding,1,64,64,64,32,8,8,32,32,2,2,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,1,8,true,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,1,8,true,1,1,Seq(1,16,1,4),1,bf16,bf16,Default,1>"
        // clang-format on
    };
};

struct DyOp_INT8_2
{
    using DeviceOp = ::DeviceOp<2, int8_t>;

    constexpr static auto expected = {
        // clang-format off
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<2,NHWGC,GKYXC,EmptyTuple,NHWGK,s8,s8,s32,s8,EmptyTuple,s8,PassThrough,PassThrough,DynamicUnaryOp,Default,MNKPadding,1,256,128,128,32,8,8,32,32,2,2,Seq(4,64,1),Seq(1,0,2),Seq(1,0,2),2,1,8,true,Seq(4,64,1),Seq(1,0,2),Seq(1,0,2),2,1,8,true,1,1,Seq(1,32,1,8),8,s8,s8,Default,1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<2,NHWGC,GKYXC,EmptyTuple,NHWGK,s8,s8,s32,s8,EmptyTuple,s8,PassThrough,PassThrough,DynamicUnaryOp,Default,MNKPadding,1,64,64,32,32,8,8,32,32,2,1,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,8,8,true,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,8,8,true,1,1,Seq(1,16,1,4),1,s8,s8,Default,1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<2,NHWGC,GKYXC,EmptyTuple,NHWGK,s8,s8,s32,s8,EmptyTuple,s8,PassThrough,PassThrough,DynamicUnaryOp,Default,MNKPadding,1,64,64,64,32,8,8,32,32,2,2,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,1,8,true,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,1,8,true,1,1,Seq(1,16,1,4),1,s8,s8,Default,1>"
        // clang-format on
    };
};

struct DyOp_INT8_3
{
    using DeviceOp = ::DeviceOp<3, int8_t>;

    constexpr static auto expected = {
        // clang-format off
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<3,NDHWGC,GKZYXC,EmptyTuple,NDHWGK,s8,s8,s32,s8,EmptyTuple,s8,PassThrough,PassThrough,DynamicUnaryOp,Default,MNKPadding,1,256,128,128,32,8,8,32,32,2,2,Seq(4,64,1),Seq(1,0,2),Seq(1,0,2),2,1,8,true,Seq(4,64,1),Seq(1,0,2),Seq(1,0,2),2,1,8,true,1,1,Seq(1,32,1,8),8,s8,s8,Default,1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<3,NDHWGC,GKZYXC,EmptyTuple,NDHWGK,s8,s8,s32,s8,EmptyTuple,s8,PassThrough,PassThrough,DynamicUnaryOp,Default,MNKPadding,1,64,64,32,32,8,8,32,32,2,1,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,8,8,true,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,8,8,true,1,1,Seq(1,16,1,4),1,s8,s8,Default,1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<3,NDHWGC,GKZYXC,EmptyTuple,NDHWGK,s8,s8,s32,s8,EmptyTuple,s8,PassThrough,PassThrough,DynamicUnaryOp,Default,MNKPadding,1,64,64,64,32,8,8,32,32,2,2,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,1,8,true,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,1,8,true,1,1,Seq(1,16,1,4),1,s8,s8,Default,1>"
        // clang-format on
    };
};

using TestTypes = ::testing::Types<DyOp_F32_2,
                                   DyOp_F32_3,
                                   DyOp_F16_2,
                                   DyOp_F16_3,
                                   DyOp_BF16_2,
                                   DyOp_BF16_3,
                                   DyOp_INT8_2,
                                   DyOp_INT8_3>;

TYPED_TEST_SUITE(CkFactoryTestBilinearFwd, TestTypes);

TYPED_TEST(CkFactoryTestBilinearFwd, TestInstances)
{
    auto actual   = TestFixture::get_actual_instances();
    auto expected = TestFixture::get_expected_instances();

    EXPECT_THAT(actual, InstancesMatch(expected));
}
