// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_data_sdk/data_objects/tensor_attributes_generated.h>
#include <miopen/miopen.h>

namespace miopen_legacy_plugin
{

class MiopenTensor
{
public:
    MiopenTensor(const hipdnn_data_sdk::data_objects::TensorAttributes& tensor);

    MiopenTensor(const MiopenTensor&) = delete;
    MiopenTensor& operator=(const MiopenTensor&) = delete;

    MiopenTensor(MiopenTensor&& other) noexcept;
    MiopenTensor& operator=(MiopenTensor&& other) noexcept;

    ~MiopenTensor();

    int64_t uid() const;

    miopenTensorDescriptor_t tensorDescriptor() const;

private:
    int64_t _uid;
    miopenTensorDescriptor_t _descriptor{nullptr};
};

}
