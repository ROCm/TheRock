// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdlib>
#include <iostream>
#include <initializer_list>
#include <tuple>
#include <vector>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_conv_fwd_multiple_d.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_convolution_forward.hpp"

#include <gtest/gtest.h>

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

class TestGroupedConvndFwdMultiDInterfaceCompatibility : public ::testing::Test
{
    protected:
    static constexpr ck::index_t NDimSpatial = 3;

    using InDataType  = float;
    using WeiDataType = float;
    using OutDataType = float;
    using InLayout    = ck::tensor_layout::convolution::GNDHWC;
    using WeiLayout   = ck::tensor_layout::convolution::GKZYXC;
    using OutLayout   = ck::tensor_layout::convolution::GNDHWK;

    using DeviceOp = ck::tensor_operation::device::DeviceGroupedConvFwdMultipleD<NDimSpatial,
                                                                                 InLayout,
                                                                                 WeiLayout,
                                                                                 ck::Tuple<>,
                                                                                 OutLayout,
                                                                                 InDataType,
                                                                                 WeiDataType,
                                                                                 ck::Tuple<>,
                                                                                 OutDataType,
                                                                                 PassThrough,
                                                                                 PassThrough,
                                                                                 PassThrough>;

    bool Run()
    {
        const auto op_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
            DeviceOp>::GetInstances();
        return op_ptrs.size() != 0;
    }
};

TEST_F(TestGroupedConvndFwdMultiDInterfaceCompatibility, CompatibilityTest)
{
    EXPECT_TRUE(this->Run());
}
