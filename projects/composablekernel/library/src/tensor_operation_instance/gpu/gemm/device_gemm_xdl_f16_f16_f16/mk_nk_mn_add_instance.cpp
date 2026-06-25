// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "common.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using Instance  = InstanceTN;
using Instances = OwnerList<Instance>;

void add_device_gemm_xdl_f16_f16_f16_mk_nk_mn_default_pipeline_v1_instances(Instances&);
void add_device_gemm_xdl_f16_f16_f16_mk_nk_mn_interwave_pipeline_v1_instances(Instances&);
void add_device_gemm_xdl_f16_f16_f16_mk_nk_mn_default_pipeline_v2_instances(Instances&);
void add_device_gemm_xdl_f16_f16_f16_mk_nk_mn_default_pipeline_v2_opt_instances(Instances&);

void add_device_gemm_xdl_f16_f16_f16_mk_nk_mn_irregular_default_pipeline_v1_instances(Instances&);
void add_device_gemm_xdl_f16_f16_f16_mk_nk_mn_irregular_interwave_pipeline_v1_instances(Instances&);
void add_device_gemm_xdl_f16_f16_f16_mk_nk_mn_irregular_default_pipeline_v2_instances(Instances&);

void add_device_gemm_xdl_f16_f16_f16_mk_nk_mn_instances(Instances& instances)
{
    add_device_gemm_xdl_f16_f16_f16_mk_nk_mn_default_pipeline_v1_instances(instances);
    add_device_gemm_xdl_f16_f16_f16_mk_nk_mn_interwave_pipeline_v1_instances(instances);
    add_device_gemm_xdl_f16_f16_f16_mk_nk_mn_default_pipeline_v2_instances(instances);
    add_device_gemm_xdl_f16_f16_f16_mk_nk_mn_default_pipeline_v2_opt_instances(instances);

    add_device_gemm_xdl_f16_f16_f16_mk_nk_mn_irregular_default_pipeline_v1_instances(instances);
    add_device_gemm_xdl_f16_f16_f16_mk_nk_mn_irregular_interwave_pipeline_v1_instances(instances);
    add_device_gemm_xdl_f16_f16_f16_mk_nk_mn_irregular_default_pipeline_v2_instances(instances);
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
