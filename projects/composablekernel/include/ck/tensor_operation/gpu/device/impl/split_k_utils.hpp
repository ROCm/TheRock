// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include <numeric>
#include <hip/hip_runtime.h>
#include "ck/utility/env.hpp"
#include "ck/utility/number.hpp"
#include "ck/host_utility/hip_check_error.hpp"
#include "ck/ck.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

struct DeviceProperties
{
    DeviceProperties()
    {
        hipDeviceProp_t dev_prop;
        hipDevice_t dev;
        hip_check_error(hipGetDevice(&dev));
        hip_check_error(hipGetDeviceProperties(&dev_prop, dev));

        num_cu_ = dev_prop.multiProcessorCount;
    };
    int num_cu_;
};

inline ck::index_t get_best_occupancy_k_batch_value(int max_occupancy, ck::index_t grid_size)
{
    static DeviceProperties device_properties;
    const int max_capacity = max_occupancy * device_properties.num_cu_;

    ck::index_t k_batch = 1;
    const auto optimal_split =
        static_cast<ck::index_t>(std::floor((1.0 * max_capacity) / grid_size));
    if(optimal_split > 1)
    {
        k_batch = optimal_split;
    }

    if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
    {
        std::cout << "[SPLIT-K AUTODEDUCE] Max active thread blocks per CU for GEMM kernel:  "
                  << max_occupancy << std::endl;
        std::cout << "[SPLIT-K AUTODEDUCE] Output grid size:  " << grid_size << std::endl;
        std::cout << "[SPLIT-K AUTODEDUCE] Optimal split-k value " << k_batch << std::endl;
    }
    return k_batch;
}

template <ck::index_t NDimSpatial>
inline auto
get_bwd_weight_gemm_sizes(const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_lengths,
                          const std::array<index_t, NDimSpatial + 3>& e_g_k_c_xs_lengths)
{
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};

    // The input array has elements in the order: G, N, K, Do, Ho, Wo
    // GemmK = N * Do * Ho * Wo for the BWD weight pass.
    constexpr index_t spatial_offset = 3;
    const index_t DoHoWo             = std::accumulate(begin(a_g_n_k_wos_lengths) + spatial_offset,
                                           end(a_g_n_k_wos_lengths),
                                           index_t{1},
                                           std::multiplies<>{});
    const auto gemmK                 = a_g_n_k_wos_lengths[I1] * DoHoWo;

    // The GEMM M dimension is the number of output channels.
    const auto gemmM = e_g_k_c_xs_lengths[I1];

    // The output array has elements in the order: G, K, C, X, Y, Z
    // GemmN = C * X * Y * Z for the BWD weight pass.
    const index_t XYZ = std::accumulate(begin(e_g_k_c_xs_lengths) + spatial_offset,
                                        end(e_g_k_c_xs_lengths),
                                        index_t{1},
                                        std::multiplies<>{});
    const auto gemmN  = e_g_k_c_xs_lengths[I2] * XYZ;
    return std::make_tuple(gemmM, gemmN, gemmK);
}

template <ck::index_t MPerBlock, ck::index_t NPerBlock>
inline ck::index_t calculate_mn_grid_size(ck::index_t gemmM, ck::index_t gemmN)
{
    const auto M0 = math::integer_divide_ceil(gemmM, MPerBlock);
    const auto N0 = math::integer_divide_ceil(gemmN, NPerBlock);
    return M0 * N0;
}

} // namespace device
} // namespace tensor_operation
} // namespace ck
