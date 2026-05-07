// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include "ck_tile/core/arch/arch.hpp"
namespace ck_tile {

template <typename CompilerTarget, typename Enabler = void>
struct StreamKCoherency
{
    static constexpr amd_buffer_coherence_enum BUFFER_COHERENCE =
        amd_buffer_coherence_enum::coherence_default;
};

template <typename CompilerTarget>
struct StreamKCoherency<CompilerTarget,
                        core::arch::enable_if_target_id_t<CompilerTarget,
                                                          core::arch::amdgcn_target_id::GFX942,
                                                          core::arch::amdgcn_target_id::GFX950>>
{
    static constexpr amd_buffer_coherence_enum BUFFER_COHERENCE =
        amd_buffer_coherence_enum::SYSTEM_NT0;
};

template <typename CompilerTarget>
struct StreamKCoherency<CompilerTarget,
                        core::arch::enable_if_target_id_t<CompilerTarget,
                                                          core::arch::amdgcn_target_id::GFX908,
                                                          core::arch::amdgcn_target_id::GFX90A>>
{
    static constexpr amd_buffer_coherence_enum BUFFER_COHERENCE =
        amd_buffer_coherence_enum::glc_slc;
};

} // namespace ck_tile
