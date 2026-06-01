// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iomanip>
#include <iostream>
#include <typeinfo>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_elementwise.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_elementwise_dynamic_vector_dims_impl.hpp"

#include "ck/library/tensor_operation_instance/gpu/transpose_3d.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"

namespace ck {
namespace profiler {

template <typename HostTensorA, typename HostTensorB, typename Functor>
void host_elementwise4D(HostTensorB& B_ndhwc, const HostTensorA& A_ncdhw, Functor functor)
{
    for(std::size_t n = 0; n < A_ncdhw.mDesc.GetLengths()[0]; ++n)
        for(std::size_t c = 0; c < A_ncdhw.mDesc.GetLengths()[1]; ++c)
            for(std::size_t d = 0; d < A_ncdhw.mDesc.GetLengths()[2]; ++d)
                for(std::size_t h = 0; h < A_ncdhw.mDesc.GetLengths()[3]; ++h)
                    for(std::size_t w = 0; w < A_ncdhw.mDesc.GetLengths()[4]; ++w)
                    {
                        auto a_val = A_ncdhw(n, c, d, h, w);
                        functor(B_ndhwc(n, d, h, w, c), a_val);
                    }
}

template <typename ADataType, typename BDataType, index_t NumDim>
bool profile_transpose_impl(int do_verification,
                            int init_method,
                            bool do_log,
                            bool time_kernel,
                            std::vector<index_t> lengths)
{
    bool pass = true;

    index_t N = lengths[0];
    index_t C = lengths[1];
    index_t D = lengths[2];
    index_t H = lengths[3];
    index_t W = lengths[4];

    std::vector<ck::index_t> ncdhw = {N, C, D, H, W};
    std::vector<ck::index_t> ndhwc = {N, D, H, W, C};
    Tensor<ADataType> a(ncdhw);
    Tensor<BDataType> b(ndhwc);
    Tensor<BDataType> host_b(ndhwc);

    // a.GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0});

    std::array<ck::index_t, 5> ab_lengths{N, C, H, W, D};
    std::array<ck::index_t, 5> a_strides = {C * D * H * W, H * W, W, 1, D * H * W}; // N, C, D, H, W
    std::array<ck::index_t, 5> b_strides = {C * H * W * D, H * W * D, W * D, D, 1}; // N, D, H, W, C

    std::cout << "A: " << a.mDesc << std::endl;
    std::cout << "B: " << b.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1: a.GenerateTensorValue(GeneratorTensor_2<ADataType>{-1, 2}); break;
    default: a.GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0});
    }

    using ElementOp = ck::tensor_operation::element_wise::PassThrough;

    DeviceMem a_device_buf(sizeof(ADataType) * a.mDesc.GetElementSpaceSize());
    DeviceMem b_device_buf(sizeof(BDataType) * b.mDesc.GetElementSpaceSize());

    a_device_buf.ToDevice(a.mData.data());

    std::array<const void*, 1> input = {a_device_buf.GetDeviceBuffer()};
    std::array<void*, 1> output      = {b_device_buf.GetDeviceBuffer()};
    using DeviceOp                   = ck::tensor_operation::device::
        DeviceElementwise<ck::Tuple<ADataType>, ck::Tuple<BDataType>, ElementOp, NumDim>;

    // get device op instances
    const auto op_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp>::GetInstances();

    std::cout << "found " << op_ptrs.size() << " instances" << std::endl;

    if(do_verification)
    {
        host_elementwise4D(host_b, a, ElementOp{});
    }

    std::string best_op_name;
    float best_ave_time   = 0;
    float best_tflops     = 0;
    float best_gb_per_sec = 0;

    for(auto& op_ptr : op_ptrs)
    {
        auto argument_ptr = op_ptr->MakeArgumentPointer(
            ab_lengths, {a_strides}, {b_strides}, input, output, ElementOp{});

        auto invoker_ptr = op_ptr->MakeInvokerPointer();

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {

            // re-init C to zero before profiling next kernel
            b_device_buf.SetZero();

            // run for verification
            invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, false});

            if(do_verification)
            {
                b_device_buf.FromDevice(b.mData.data());

                pass &= ck::utils::check_err(
                    b.mData, host_b.mData, "Error: Incorrect results b", 1e-3, 1e-3);

                if(do_log)
                {
                    LogRangeAsType<float>(std::cout << "a : ", a.mData, ",") << std::endl;
                    LogRangeAsType<float>(std::cout << "b: ", b.mData, ",") << std::endl;
                }
            }

            std::string op_name = op_ptr->GetTypeString();

            // run for timing purposes
            float ave_time =
                invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, time_kernel});

            std::size_t flop =
                std::size_t(2) * ncdhw[0] * ncdhw[1] * ncdhw[2] * ncdhw[3] * ncdhw[4];

            std::size_t num_btype =
                sizeof(ADataType) * (ncdhw[0] * ncdhw[1] * ncdhw[2] * ncdhw[3] * ncdhw[4]) +
                sizeof(BDataType) * (ncdhw[0] * ncdhw[1] * ncdhw[2] * ncdhw[3] * ncdhw[4]);

            float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

            float gb_per_sec = num_btype / 1.E6 / ave_time;

            std::cout << "Perf: " << std::setw(10) << ave_time << " ms, " << tflops << " TFlops, "
                      << gb_per_sec << " GB/s, " << op_name << std::endl;

            if(tflops > best_tflops)
            {
                best_op_name    = op_name;
                best_tflops     = tflops;
                best_ave_time   = ave_time;
                best_gb_per_sec = gb_per_sec;
            }
        }
        else
        {
            std::cout << op_ptr->GetTypeString() << " does not support this problem" << std::endl;
        }
    }

    std::cout << " N = " << N << " C = " << C << " D = " << D << " H = " << H << " W = " << W
              << " : " << best_ave_time << " ms, " << best_tflops << " TFlops, " << best_gb_per_sec
              << " GB/s, " << best_op_name << std::endl;

    return pass;
}

} // namespace profiler
} // namespace ck
