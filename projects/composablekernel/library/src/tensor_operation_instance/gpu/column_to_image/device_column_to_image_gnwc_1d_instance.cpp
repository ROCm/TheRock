// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/library/tensor_operation_instance/gpu/conv_tensor_rearrange/device_column_to_image_instance.hpp"
#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using namespace ck::conv_tensor_rearrange_op;

void add_device_column_to_image_gnwc_1d_bf16_instances(
    std::vector<std::unique_ptr<DeviceConvTensorRearrange<1, GNWC, BF16, BF16, ColumnToImage>>>&
        instances)
{
#ifdef CK_ENABLE_BF16
    add_device_operation_instances(instances, device_column_to_image_bf16_instances<1, GNWC>{});
#else
    ignore = instances;
#endif
}

void add_device_column_to_image_gnwc_1d_f16_instances(
    std::vector<std::unique_ptr<DeviceConvTensorRearrange<1, GNWC, F16, F16, ColumnToImage>>>&
        instances)
{
#ifdef CK_ENABLE_FP16
    add_device_operation_instances(instances, device_column_to_image_f16_instances<1, GNWC>{});
#else
    ignore = instances;
#endif
}

void add_device_column_to_image_gnwc_1d_f32_instances(
    std::vector<std::unique_ptr<DeviceConvTensorRearrange<1, GNWC, F32, F32, ColumnToImage>>>&
        instances)
{
#ifdef CK_ENABLE_FP32
    add_device_operation_instances(instances, device_column_to_image_f32_instances<1, GNWC>{});
#else
    ignore = instances;
#endif
}

void add_device_column_to_image_gnwc_1d_i8_instances(
    std::vector<std::unique_ptr<DeviceConvTensorRearrange<1, GNWC, int8_t, int8_t, ColumnToImage>>>&
        instances)
{
#ifdef CK_ENABLE_INT8
    add_device_operation_instances(instances, device_column_to_image_i8_instances<1, GNWC>{});
#else
    ignore = instances;
#endif
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
