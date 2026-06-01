// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace ck {

enum struct AmdBufferCoherenceEnum
{
    DefaultCoherence = 0, // default value
#if defined(__gfx12__)
    // Temporal hint
    RT    = 0, // regular temporal
    NT    = 1, // non temporal
    HT    = 2, // high priority temporal
    LU    = 3, // last use (load op)
    WB    = 3, // same as HT, overrides WR in far cache (store op)
    NT_RT = 4, // non temporal for near cache, regular for far cache
    RT_NT = 5, // regular for near cache, non-temporal for far cache
    NT_HT = 6, // non temporal for near cache, high priority for far cache
    NT_WB = 7, // non temporal for near cache, WB for far cache
               // (store op, reserved for load op)
    // Scope
    CU     = 0,
    SE     = 8,
    DEVICE = 16,
    SYSTEM = 24,
    // Temporal Hint for CU
    CU_RT    = RT | CU,
    CU_NT    = NT | CU,
    CU_HT    = HT | CU,
    CU_LU    = LU | CU,
    CU_WB    = WB | CU,
    CU_NT_RT = NT_RT | CU,
    CU_RT_NT = RT_NT | CU,
    CU_NT_HT = NT_HT | CU,
    CU_NT_WB = NT_WB | CU,
    // Temporal Hint for SE
    SE_RT    = RT | SE,
    SE_NT    = NT | SE,
    SE_HT    = HT | SE,
    SE_LU    = LU | SE,
    SE_WB    = WB | SE,
    SE_NT_RT = NT_RT | SE,
    SE_RT_NT = RT_NT | SE,
    SE_NT_HT = NT_HT | SE,
    SE_NT_WB = NT_WB | SE,
    // Temporal Hint for DEVICE
    DEVICE_RT    = RT | DEVICE,
    DEVICE_NT    = NT | DEVICE,
    DEVICE_HT    = HT | DEVICE,
    DEVICE_LU    = LU | DEVICE,
    DEVICE_WB    = WB | DEVICE,
    DEVICE_NT_RT = NT_RT | DEVICE,
    DEVICE_RT_NT = RT_NT | DEVICE,
    DEVICE_NT_HT = NT_HT | DEVICE,
    DEVICE_NT_WB = NT_WB | DEVICE,
    // Temporal Hint for SYSTEM
    SYSTEM_RT    = RT | SYSTEM,
    SYSTEM_NT    = NT | SYSTEM,
    SYSTEM_HT    = HT | SYSTEM,
    SYSTEM_LU    = LU | SYSTEM,
    SYSTEM_WB    = WB | SYSTEM,
    SYSTEM_NT_RT = NT_RT | SYSTEM,
    SYSTEM_RT_NT = RT_NT | SYSTEM,
    SYSTEM_NT_HT = NT_HT | SYSTEM,
    SYSTEM_NT_WB = NT_WB | SYSTEM,

    // GFX942 and GFX950 compatiblity
    GROUP_NT0  = CU_RT,
    GROUP_NT1  = CU_NT,
    DEVICE_NT0 = DEVICE_RT,
    DEVICE_NT1 = DEVICE_NT,
    SYSTEM_NT0 = SYSTEM_RT,
    SYSTEM_NT1 = SYSTEM_NT,
    // Other archs compatiblity
    GLC     = DEVICE_NT,
    SLC     = SYSTEM_NT,
    GLC_SLC = DEVICE_NT | SYSTEM_NT,

// gfx94: bit 0 = sc0, bit 1 = nt, bit 3 = swz, bit 4 = sc1
// SC[1:0] System Cache level: 0=wave, 1=group, 2=device, 3=system
// NT Non-Temporal: 0=expect temporal reuse; 1=do not expect temporal reuse
#elif defined(__gfx942__) || defined(__gfx950__)

    WAVE   = 0,
    GROUP  = 1,
    DEVICE = 16,
    SYSTEM = 17,
    NT0    = 0,
    NT1    = 2,

    WAVE_NT0   = NT0 | WAVE,
    WAVE_NT1   = NT1 | WAVE,
    GROUP_NT0  = NT0 | GROUP,
    GROUP_NT1  = NT1 | GROUP,
    DEVICE_NT0 = NT0 | DEVICE,
    DEVICE_NT1 = NT1 | DEVICE,
    SYSTEM_NT0 = NT0 | SYSTEM,
    SYSTEM_NT1 = NT1 | SYSTEM,

    // Other archs compatiblity
    GLC     = DEVICE_NT1,
    SLC     = SYSTEM_NT1,
    GLC_SLC = DEVICE_NT1 | SYSTEM_NT1,
#else
    GLC     = 1,
    SLC     = 2,
    GLC_SLC = 3,

    // Other archs compatiblity
    DEVICE_NT0 = 0,
    SYSTEM_NT0 = 0,
    DEVICE_NT1 = GLC,
    SYSTEM_NT1 = SLC,
#endif
};

} // namespace ck
