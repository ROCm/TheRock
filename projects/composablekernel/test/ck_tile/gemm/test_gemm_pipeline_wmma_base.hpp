// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_attribute_wmma_impl.hpp"
#include "test_gemm_pipeline_util.hpp"

template <typename Tuple, typename Derived>
class TestCkTileGemmPipelineWmmaBase : public TestCkTileGemmPipeline<Tuple, Derived>
{
    public:
    static constexpr bool check_data_type()
    {
        using Base = TestCkTileGemmPipeline<Tuple, Derived>;

#if defined(ARCH_GFX12)
        using DeviceIp = ck_tile::gfx12_t;
#elif defined(ARCH_GFX11)
        using DeviceIp = ck_tile::gfx11_t;
#else
#error "Unsupported architecture for WMMA"
#endif

        using BTypeToUse =
            std::conditional_t<std::is_same_v<typename Base::BDataType, ck_tile::pk_int4_t>,
                               typename Base::ADataType,
                               typename Base::BDataType>;
        return ck_tile::has_wmma_traits_v<DeviceIp,
                                          typename Base::ADataType,
                                          BTypeToUse,
                                          typename Base::AccDataType,
                                          ck_tile::constant<Base::M_Warp_Tile>::value,
                                          ck_tile::constant<Base::N_Warp_Tile>::value,
                                          ck_tile::constant<Base::K_Warp_Tile>::value>;
    }
};
