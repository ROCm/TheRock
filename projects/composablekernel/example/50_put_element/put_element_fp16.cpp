// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_put_element_impl.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"

using ::ck::DeviceMem;
using ::ck::HostTensorDescriptor;
using ::ck::Tensor;

using XDataType     = ck::half_t;
using YDataType     = ck::half_t;
using IndexDataType = int32_t;

using YElementwiseOp = ck::tensor_operation::element_wise::PassThrough;

using DeviceInstance =
    ck::tensor_operation::device::DevicePutElementImpl<XDataType,     // XDataType
                                                       IndexDataType, // IndexDataType
                                                       YDataType,     // YDataType
                                                       YElementwiseOp,
                                                       ck::InMemoryDataOperationEnum::Set,
                                                       1>;

int main()
{
    bool do_verification = true;
    bool time_kernel     = false;

    int N = 1024;

    Tensor<XDataType> x(HostTensorDescriptor{N});
    Tensor<IndexDataType> indices(HostTensorDescriptor{N});
    Tensor<YDataType> y(HostTensorDescriptor{N});

    x.GenerateTensorValue(GeneratorTensor_3<XDataType>{-1.0, 1.0});
    for(int i = 0; i < N; ++i)
        indices(i) = i;

    DeviceMem x_device_buf(sizeof(XDataType) * x.mDesc.GetElementSpaceSize());
    DeviceMem y_device_buf(sizeof(YDataType) * y.mDesc.GetElementSpaceSize());
    DeviceMem indices_device_buf(sizeof(IndexDataType) * indices.mDesc.GetElementSpaceSize());

    x_device_buf.ToDevice(x.mData.data());
    indices_device_buf.ToDevice(indices.mData.data());

    auto put_instance     = DeviceInstance{};
    auto put_invoker_ptr  = put_instance.MakeInvokerPointer();
    auto put_argument_ptr = put_instance.MakeArgumentPointer(
        static_cast<XDataType*>(x_device_buf.GetDeviceBuffer()),
        static_cast<IndexDataType*>(indices_device_buf.GetDeviceBuffer()),
        static_cast<YDataType*>(y_device_buf.GetDeviceBuffer()),
        N,
        N,
        YElementwiseOp{});

    if(!put_instance.IsSupportedArgument(put_argument_ptr.get()))
    {
        throw std::runtime_error("argument is not supported!");
    }

    float ave_time =
        put_invoker_ptr->Run(put_argument_ptr.get(), StreamConfig{nullptr, time_kernel});

    std::cout << "perf: " << ave_time << " ms" << std::endl;

    bool pass = true;
    if(do_verification)
    {
        Tensor<YDataType> y_host(HostTensorDescriptor{N});

        for(int i = 0; i < N; ++i)
        {
            IndexDataType idx = indices(i);
            y_host(idx)       = x(i);
        }

        y_device_buf.FromDevice(y.mData.data());
        pass = ck::utils::check_err(y, y_host);
    }

    return (pass ? 0 : 1);
}
