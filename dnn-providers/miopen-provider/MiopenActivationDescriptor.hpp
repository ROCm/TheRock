// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_data_sdk/data_objects/pointwise_attributes_generated.h>
#include <miopen/miopen.h>

namespace miopen_legacy_plugin
{

class MiopenActivationDescriptor
{
public:
    explicit MiopenActivationDescriptor(
        const hipdnn_data_sdk::data_objects::PointwiseAttributes& pointwiseAttrs);

    MiopenActivationDescriptor(const MiopenActivationDescriptor&) = delete;
    MiopenActivationDescriptor& operator=(const MiopenActivationDescriptor&) = delete;

    MiopenActivationDescriptor(MiopenActivationDescriptor&& other) noexcept;
    MiopenActivationDescriptor& operator=(MiopenActivationDescriptor&& other) noexcept;

    ~MiopenActivationDescriptor();

    miopenActivationDescriptor_t activationDescriptor() const;

private:
    miopenActivationDescriptor_t _descriptor{nullptr};
};

} // namespace miopen_legacy_plugin
