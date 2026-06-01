// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_weight/device_grouped_conv_bwd_weight_wmma_bilinear_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

// Compilation parameters for in[n, hi, wi, g, c] * wei[g, k, y, x, c] = out[n, ho, wo, g, k]
void add_device_grouped_conv3d_bwd_weight_wmma_bilinear_ndhwgc_gkzyxc_ndhwgk_f16_instances(
    [[maybe_unused]] std::vector<std::unique_ptr<DeviceGroupedConvBwdWeightMultipleD<3,
                                                                                     NDHWGC,
                                                                                     GKZYXC,
                                                                                     NDHWGK,
                                                                                     Tuple<GKZYXC>,
                                                                                     F16,
                                                                                     F16,
                                                                                     F16,
                                                                                     Tuple<F16>,
                                                                                     PassThrough,
                                                                                     Bilinear,
                                                                                     PassThrough>>>&
        instances)
{
    // One of the kernels in this code block fails to compile, but only on Windows when building for
    // gfx1101. It succeeds on Linux for all gfx110X series GPU's, and on Windows for other gfx110X
    // series GPU's.
    // TODO: Remove this ifdef combo disabling these kernels once we have followed up with the
    // compiler team and they are able to be built again. This is the compilation error that
    // results:
    //
    // error: Illegal instruction detected: Operand has incorrect register class.
    // V_CMP_NE_U32_e32 0, $src_private_base, implicit-def $vcc, implicit $exec
    // Compiler version info:
    // AMD clang version 22.0.0git (https://github.com/ROCm/llvm-project.git
    // 8e85e3138dd485c4221cc12aff9eb60ab48ed3b5+PATCHED:93c451b46cc0dc23c47d67e394b370de65731aac)
#if !defined(_WIN32)
    // 1. Default
    add_device_operation_instances(
        instances,
        device_grouped_conv_bwd_weight_wmma_c_shuffle_f16_bilinear_instances<
            3,
            NDHWGC,
            GKZYXC,
            NDHWGK,
            ConvBwdWeightDefault>{});
    // 2. Filter1x1Stride1Pad0
    add_device_operation_instances(
        instances,
        device_grouped_conv_bwd_weight_wmma_c_shuffle_f16_bilinear_instances<
            3,
            NDHWGC,
            GKZYXC,
            NDHWGK,
            ConvBwdWeightFilter1x1Stride1Pad0>{});
#endif
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
