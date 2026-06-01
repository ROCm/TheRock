// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include "ck_tile/core/arch/arch.hpp"

using namespace ck_tile;
using namespace ck_tile::core::arch;

// Test address_space_enum string conversion
TEST(TestArch, AddressSpaceToString)
{
    EXPECT_STREQ(address_space_to_string(address_space_enum::generic), "generic");
    EXPECT_STREQ(address_space_to_string(address_space_enum::global), "global");
    EXPECT_STREQ(address_space_to_string(address_space_enum::lds), "lds");
    EXPECT_STREQ(address_space_to_string(address_space_enum::sgpr), "sgpr");
    EXPECT_STREQ(address_space_to_string(address_space_enum::constant), "constant");
    EXPECT_STREQ(address_space_to_string(address_space_enum::vgpr), "vgpr");
    EXPECT_STREQ(address_space_to_string(static_cast<address_space_enum>(999)), "unknown");
}

#if 1 // __cplusplus <= 201703L

// Tests make_amdgcn_gf9_target function
TEST(ArchTest, MakeGfx9TargetFields)
{
    constexpr auto target = make_amdgcn_gfx9_target<amdgcn_target_id::GFX908>();
    EXPECT_EQ(target.TARGET_ID, amdgcn_target_id::GFX908);
    EXPECT_EQ(target.FAMILY_ID, amdgcn_target_family_id::GFX9);
    EXPECT_EQ(target.ARCH_ID, amdgcn_target_arch_id::CDNA);
    EXPECT_EQ(target.WAVE_SIZE_ID, amdgcn_target_wave_size_id::WAVE64);
}

// Tests make_amdgcn_gfx11_target function
TEST(ArchTest, MakeGfx11TargetFields)
{
    constexpr auto target = make_amdgcn_gfx11_target<amdgcn_target_id::GFX1100>();
    EXPECT_EQ(target.TARGET_ID, amdgcn_target_id::GFX1100);
    EXPECT_EQ(target.FAMILY_ID, amdgcn_target_family_id::GFX11);
    EXPECT_EQ(target.ARCH_ID, amdgcn_target_arch_id::RDNA);
    EXPECT_EQ(target.WAVE_SIZE_ID, amdgcn_target_wave_size_id::WAVE32);
}

// Tests make_amdgcn_gfx12_target function
TEST(ArchTest, MakeGfx12TargetFields)
{
    constexpr auto target = make_amdgcn_gfx12_target<amdgcn_target_id::GFX1200>();
    EXPECT_EQ(target.TARGET_ID, amdgcn_target_id::GFX1200);
    EXPECT_EQ(target.FAMILY_ID, amdgcn_target_family_id::GFX12);
    EXPECT_EQ(target.ARCH_ID, amdgcn_target_arch_id::RDNA);
    EXPECT_EQ(target.WAVE_SIZE_ID, amdgcn_target_wave_size_id::WAVE32);
}

// Tests default amdgcn_target
TEST(ArchTest, DefaultTargetIsHost)
{
    constexpr auto target = amdgcn_target<>{};
    EXPECT_EQ(target.TARGET_ID, amdgcn_target_id::HOST);
    EXPECT_EQ(target.FAMILY_ID, amdgcn_target_family_id::HOST);
    EXPECT_EQ(target.ARCH_ID, amdgcn_target_arch_id::HOST);
    EXPECT_EQ(target.WAVE_SIZE_ID, amdgcn_target_wave_size_id::HOST);
}

// Tests get_compiler_target function on host
TEST(ArchTest, GetCompilerTargetDefaultIsHost)
{
    // By default, get_compiler_target should return HOST arch id because we aren't on device
    auto target = get_compiler_target();
    EXPECT_EQ(target.TARGET_ID, amdgcn_target_id::HOST);
    EXPECT_EQ(target.FAMILY_ID, amdgcn_target_family_id::HOST);
    EXPECT_EQ(target.ARCH_ID, amdgcn_target_arch_id::HOST);
    EXPECT_EQ(target.WAVE_SIZE_ID, amdgcn_target_wave_size_id::HOST);
}

// SFINAE test setup for incoming acceptable target architecture ids
template <typename Target, typename = void>
struct SFINAETestTargetIdGfx908OrGfx90a
{
    static constexpr bool value = false;
};

// Acceptable target arch ids: GFX908, GFX90A
template <typename Target>
struct SFINAETestTargetIdGfx908OrGfx90a<
    Target,
    enable_if_target_id_t<Target, amdgcn_target_id::GFX908, amdgcn_target_id::GFX90A>>
{
    static constexpr bool value = true;
};

// SFINAE test setup for incoming acceptable target family ids
template <typename Target, typename = void>
struct SFINAETestFamilyIdGfx9
{
    static constexpr bool value = false;
};

// Acceptable target arch family ids: GFX9
template <typename Target>
struct SFINAETestFamilyIdGfx9<Target,
                              enable_if_target_family_id_t<Target, amdgcn_target_family_id::GFX9>>
{
    static constexpr bool value = true;
};

// SFINAE test setup for incoming acceptable target architecture ids
template <typename Target, typename = void>
struct SFINAETestArchIdCdna
{
    static constexpr bool value = false;
};

// Acceptable target arch ids: CDNA
template <typename Target>
struct SFINAETestArchIdCdna<Target, enable_if_target_arch_id_t<Target, amdgcn_target_arch_id::CDNA>>
{
    static constexpr bool value = true;
};

// SFINAE test setup for incoming acceptable target wave size ids
template <typename Target, typename = void>
struct SFINAETestWaveSizeIdWave64
{
    static constexpr bool value = false;
};

// Acceptable target arch wave size ids: WAVE64
template <typename Target>
struct SFINAETestWaveSizeIdWave64<
    Target,
    enable_if_target_wave_size_id_t<Target, amdgcn_target_wave_size_id::WAVE64>>
{
    static constexpr bool value = true;
};

// Test SFINAE enablers with various architectures
TEST(ArchTest, TestSFINAEEnablersGfx9CdnaWave64)
{
    static constexpr auto target = make_amdgcn_gfx9_target<amdgcn_target_id::GFX908>();
    using Target                 = decltype(target);
    EXPECT_EQ(true, SFINAETestTargetIdGfx908OrGfx90a<Target>::value);
    EXPECT_EQ(true, SFINAETestFamilyIdGfx9<Target>::value);
    EXPECT_EQ(true, SFINAETestArchIdCdna<Target>::value);
    EXPECT_EQ(true, SFINAETestWaveSizeIdWave64<Target>::value);
}

TEST(ArchTest, TestSFINAEEnablersGfx11RdnaWave32)
{
    static constexpr auto target = make_amdgcn_gfx11_target<amdgcn_target_id::GFX1100>();
    using Target                 = decltype(target);
    EXPECT_EQ(false, SFINAETestTargetIdGfx908OrGfx90a<Target>::value);
    EXPECT_EQ(false, SFINAETestFamilyIdGfx9<Target>::value);
    EXPECT_EQ(false, SFINAETestArchIdCdna<Target>::value);
    EXPECT_EQ(false, SFINAETestWaveSizeIdWave64<Target>::value);
}

TEST(ArchTest, TestSFINAEEnablersGfx12RdnaWave32)
{
    static constexpr auto target = make_amdgcn_gfx12_target<amdgcn_target_id::GFX1200>();
    using Target                 = decltype(target);
    EXPECT_EQ(false, SFINAETestTargetIdGfx908OrGfx90a<Target>::value);
    EXPECT_EQ(false, SFINAETestFamilyIdGfx9<Target>::value);
    EXPECT_EQ(false, SFINAETestArchIdCdna<Target>::value);
    EXPECT_EQ(false, SFINAETestWaveSizeIdWave64<Target>::value);
}

TEST(ArchTest, TestSFINAEEnablersHost)
{
    static constexpr auto target = amdgcn_target<>{};
    using Target                 = decltype(target);
    EXPECT_EQ(false, SFINAETestTargetIdGfx908OrGfx90a<Target>::value);
    EXPECT_EQ(false, SFINAETestFamilyIdGfx9<Target>::value);
    EXPECT_EQ(false, SFINAETestArchIdCdna<Target>::value);
    // TODO: Should host be considered as WAVE64 or not? For now, we will consider it as WAVE64
    EXPECT_EQ(true, SFINAETestWaveSizeIdWave64<Target>::value);
}

TEST(ArchTest, TestSFINAEEnablersGfx9CdnaWave32)
{
    static constexpr auto target = amdgcn_target<amdgcn_target_id::GFX908,
                                                 amdgcn_target_family_id::GFX9,
                                                 amdgcn_target_arch_id::CDNA,
                                                 amdgcn_target_wave_size_id::WAVE32>{};
    using Target                 = decltype(target);
    EXPECT_EQ(true, SFINAETestTargetIdGfx908OrGfx90a<Target>::value);
    EXPECT_EQ(true, SFINAETestFamilyIdGfx9<Target>::value);
    EXPECT_EQ(true, SFINAETestArchIdCdna<Target>::value);
    EXPECT_EQ(false, SFINAETestWaveSizeIdWave64<Target>::value);
}

TEST(ArchTest, TestSFINAEEnablersMix)
{
    static constexpr auto target = amdgcn_target<amdgcn_target_id::GFX90A,
                                                 amdgcn_target_family_id::GFX12,
                                                 amdgcn_target_arch_id::CDNA,
                                                 amdgcn_target_wave_size_id::WAVE32>{};
    using Target                 = decltype(target);
    EXPECT_EQ(true, SFINAETestTargetIdGfx908OrGfx90a<Target>::value);
    EXPECT_EQ(false, SFINAETestFamilyIdGfx9<Target>::value);
    EXPECT_EQ(true, SFINAETestArchIdCdna<Target>::value);
    EXPECT_EQ(false, SFINAETestWaveSizeIdWave64<Target>::value);
}

#elif 0 // TODO: c++20 tests

// Tests make_amdgcn_gf9_target function
TEST(ArchTest, MakeGfx9TargetFields)
{
    constexpr auto target = make_amdgcn_gfx9_target(amdgcn_target_id::GFX908);
    EXPECT_EQ(target.TARGET_ID, amdgcn_target_id::GFX908);
    EXPECT_EQ(target.FAMILY_ID, amdgcn_target_family_id::GFX9);
    EXPECT_EQ(target.ARCH_ID, amdgcn_target_arch_id::CDNA);
    EXPECT_EQ(target.WAVE_SIZE_ID, amdgcn_target_wave_size_id::WAVE64);
}

// Tests make_amdgcn_gfx11_target function
TEST(ArchTest, MakeGfx11TargetFields)
{
    constexpr auto target = make_amdgcn_gfx11_target(amdgcn_target_id::GFX1100);
    EXPECT_EQ(target.TARGET_ID, amdgcn_target_id::GFX1100);
    EXPECT_EQ(target.FAMILY_ID, amdgcn_target_family_id::GFX11);
    EXPECT_EQ(target.ARCH_ID, amdgcn_target_arch_id::RDNA);
    EXPECT_EQ(target.WAVE_SIZE_ID, amdgcn_target_wave_size_id::WAVE32);
}

// Tests make_amdgcn_gfx12_target function
TEST(ArchTest, MakeGfx12TargetFields)
{
    constexpr auto target = make_amdgcn_gfx12_target(amdgcn_target_id::GFX1200);
    EXPECT_EQ(target.TARGET_ID, amdgcn_target_id::GFX1200);
    EXPECT_EQ(target.FAMILY_ID, amdgcn_target_family_id::GFX12);
    EXPECT_EQ(target.ARCH_ID, amdgcn_target_arch_id::RDNA);
    EXPECT_EQ(target.WAVE_SIZE_ID, amdgcn_target_wave_size_id::WAVE32);
}

// Tests default amdgcn_target
TEST(ArchTest, DefaultTargetIsHost)
{
    constexpr amdgcn_target target{};
    EXPECT_EQ(target.TARGET_ID, amdgcn_target_id::HOST);
    EXPECT_EQ(target.FAMILY_ID, amdgcn_target_family_id::HOST);
    EXPECT_EQ(target.ARCH_ID, amdgcn_target_arch_id::HOST);
    EXPECT_EQ(target.WAVE_SIZE_ID, amdgcn_target_wave_size_id::HOST);
}

// Tests get_compiler_target function on host
TEST(ArchTest, GetCompilerTargetDefaultIsHost)
{
    // By default, get_compiler_target should return HOST arch id because we aren't on device
    auto target = get_compiler_target();
    EXPECT_EQ(target.TARGET_ID, amdgcn_target_id::HOST);
    EXPECT_EQ(target.FAMILY_ID, amdgcn_target_family_id::HOST);
    EXPECT_EQ(target.ARCH_ID, amdgcn_target_arch_id::HOST);
    EXPECT_EQ(target.WAVE_SIZE_ID, amdgcn_target_wave_size_id::HOST);
}

// SFINAE test setup for incoming acceptable target architecture ids
template <amdgcn_target Target, typename = void>
struct SFINAETestTargetIdGfx908OrGfx90a
{
    static constexpr bool value = false;
};

// Acceptable target arch ids: GFX908, GFX90A
template <amdgcn_target Target>
struct SFINAETestTargetIdGfx908OrGfx90a<
    Target,
    enable_if_target_id_t<Target, amdgcn_target_id::GFX908, amdgcn_target_id::GFX90A>>
{
    static constexpr bool value = true;
};

// SFINAE test setup for incoming acceptable target family ids
template <amdgcn_target Target, typename = void>
struct SFINAETestFamilyIdGfx9
{
    static constexpr bool value = false;
};

// Acceptable target arch family ids: GFX9
template <amdgcn_target Target>
struct SFINAETestFamilyIdGfx9<Target,
                              enable_if_target_family_id_t<Target, amdgcn_target_family_id::GFX9>>
{
    static constexpr bool value = true;
};

// SFINAE test setup for incoming acceptable target architecture ids
template <amdgcn_target Target, typename = void>
struct SFINAETestArchIdCdna
{
    static constexpr bool value = false;
};

// Acceptable target arch ids: CDNA
template <amdgcn_target Target>
struct SFINAETestArchIdCdna<Target, enable_if_target_arch_id_t<Target, amdgcn_target_arch_id::CDNA>>
{
    static constexpr bool value = true;
};

// SFINAE test setup for incoming acceptable target wave size ids
template <amdgcn_target Target, typename = void>
struct SFINAETestWaveSizeIdWave64
{
    static constexpr bool value = false;
};

// Acceptable target arch wave size ids: WAVE64
template <amdgcn_target Target>
struct SFINAETestWaveSizeIdWave64<
    Target,
    enable_if_target_wave_size_id_t<Target, amdgcn_target_wave_size_id::WAVE64>>
{
    static constexpr bool value = true;
};

// Test SFINAE enablers with various architectures
TEST(ArchTest, TestSFINAEEnablersGfx9CdnaWave64)
{
    static constexpr auto target =
        amdgcn_target{.TARGET_ID    = amdgcn_target_id::GFX908,
                      .FAMILY_ID    = amdgcn_target_family_id::GFX9,
                      .ARCH_ID      = amdgcn_target_arch_id::CDNA,
                      .WAVE_SIZE_ID = amdgcn_target_wave_size_id::WAVE64};
    EXPECT_EQ(true, SFINAETestTargetIdGfx908OrGfx90a<target>::value);
    EXPECT_EQ(true, SFINAETestFamilyIdGfx9<target>::value);
    EXPECT_EQ(true, SFINAETestArchIdCdna<target>::value);
    EXPECT_EQ(true, SFINAETestWaveSizeIdWave64<target>::value);
}

TEST(ArchTest, TestSFINAEEnablersGfx11RdnaWave32)
{
    static constexpr auto target =
        amdgcn_target{.TARGET_ID    = amdgcn_target_id::GFX1100,
                      .FAMILY_ID    = amdgcn_target_family_id::GFX11,
                      .ARCH_ID      = amdgcn_target_arch_id::RDNA,
                      .WAVE_SIZE_ID = amdgcn_target_wave_size_id::WAVE32};
    EXPECT_EQ(false, SFINAETestTargetIdGfx908OrGfx90a<target>::value);
    EXPECT_EQ(false, SFINAETestFamilyIdGfx9<target>::value);
    EXPECT_EQ(false, SFINAETestArchIdCdna<target>::value);
    EXPECT_EQ(false, SFINAETestWaveSizeIdWave64<target>::value);
}

TEST(ArchTest, TestSFINAEEnablersGfx12RdnaWave32)
{
    static constexpr auto target =
        amdgcn_target{.TARGET_ID    = amdgcn_target_id::GFX1200,
                      .FAMILY_ID    = amdgcn_target_family_id::GFX12,
                      .ARCH_ID      = amdgcn_target_arch_id::RDNA,
                      .WAVE_SIZE_ID = amdgcn_target_wave_size_id::WAVE32};
    EXPECT_EQ(false, SFINAETestTargetIdGfx908OrGfx90a<target>::value);
    EXPECT_EQ(false, SFINAETestFamilyIdGfx9<target>::value);
    EXPECT_EQ(false, SFINAETestArchIdCdna<target>::value);
    EXPECT_EQ(false, SFINAETestWaveSizeIdWave64<target>::value);
}

TEST(ArchTest, TestSFINAEEnablersHost)
{
    static constexpr auto target = amdgcn_target{.TARGET_ID    = amdgcn_target_id::HOST,
                                                 .FAMILY_ID    = amdgcn_target_family_id::HOST,
                                                 .ARCH_ID      = amdgcn_target_arch_id::HOST,
                                                 .WAVE_SIZE_ID = amdgcn_target_wave_size_id::HOST};
    EXPECT_EQ(false, SFINAETestTargetIdGfx908OrGfx90a<target>::value);
    EXPECT_EQ(false, SFINAETestFamilyIdGfx9<target>::value);
    EXPECT_EQ(false, SFINAETestArchIdCdna<target>::value);
    EXPECT_EQ(false, SFINAETestWaveSizeIdWave64<target>::value);
}

TEST(ArchTest, TestSFINAEEnablersGfx9CdnaWave32)
{
    static constexpr auto target =
        amdgcn_target{.TARGET_ID    = amdgcn_target_id::GFX908,
                      .FAMILY_ID    = amdgcn_target_family_id::GFX9,
                      .ARCH_ID      = amdgcn_target_arch_id::CDNA,
                      .WAVE_SIZE_ID = amdgcn_target_wave_size_id::WAVE32};
    EXPECT_EQ(true, SFINAETestTargetIdGfx908OrGfx90a<target>::value);
    EXPECT_EQ(true, SFINAETestFamilyIdGfx9<target>::value);
    EXPECT_EQ(true, SFINAETestArchIdCdna<target>::value);
    EXPECT_EQ(false, SFINAETestWaveSizeIdWave64<target>::value);
}

TEST(ArchTest, TestSFINAEEnablersMix)
{
    static constexpr auto target =
        amdgcn_target{.TARGET_ID    = amdgcn_target_id::GFX90A,
                      .FAMILY_ID    = amdgcn_target_family_id::GFX12,
                      .ARCH_ID      = amdgcn_target_arch_id::CDNA,
                      .WAVE_SIZE_ID = amdgcn_target_wave_size_id::WAVE32};
    EXPECT_EQ(true, SFINAETestTargetIdGfx908OrGfx90a<target>::value);
    EXPECT_EQ(false, SFINAETestFamilyIdGfx9<target>::value);
    EXPECT_EQ(true, SFINAETestArchIdCdna<target>::value);
    EXPECT_EQ(false, SFINAETestWaveSizeIdWave64<target>::value);
}

#endif // __cplusplus <= 201703L
