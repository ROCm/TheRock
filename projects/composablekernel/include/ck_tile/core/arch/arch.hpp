// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Address Space for AMDGCN
// https://llvm.org/docs/AMDGPUUsage.html#address-space

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/utility/type_traits.hpp"
#include "ck_tile/core/arch/amd_buffer_addressing_builtins.hpp"
#include "ck_tile/core/arch/amd_buffer_addressing.hpp"
#include "ck_tile/core/utility/ignore.hpp"

#define CK_TILE_S_CNT_MAX 0b1100'1111'0111'1111
#define CK_TILE_VMCNT(cnt)                                              \
    ([]() { static_assert(!((cnt) >> 6), "VMCNT only has 6 bits"); }(), \
     ((cnt) & 0b1111) | (((cnt) & 0b110000) << 10))
#define CK_TILE_EXPCNT(cnt) \
    ([]() { static_assert(!((cnt) >> 3), "EXP only has 3 bits"); }(), ((cnt) << 4))
#define CK_TILE_LGKMCNT(cnt) \
    ([]() { static_assert(!((cnt) >> 4), "LGKM only has 4 bits"); }(), ((cnt) << 8))

namespace ck_tile {

template <typename, bool>
struct safe_underlying_type;

template <typename T>
struct safe_underlying_type<T, true>
{
    using type = std::underlying_type_t<T>;
};

template <typename T>
struct safe_underlying_type<T, false>
{
    using type = void;
};

template <typename T>
using safe_underlying_type_t = typename safe_underlying_type<T, std::is_enum<T>::value>::type;

enum struct address_space_enum : std::uint16_t
{
    generic = 0,
    global,
    lds,
    sgpr,
    constant,
    vgpr
};

enum struct memory_operation_enum : std::uint16_t
{
    set = 0,
    atomic_add,
    atomic_max,
    add
};

namespace core::arch {

/**
 * @enum amdgcn_target_id
 * @brief Defines constants for AMDGCN architecture target IDs
 */
enum struct amdgcn_target_id
{
    GFX908         = 0x0908, // MI-100...
    GFX90A         = 0x090A,
    GFX942         = 0x0942,
    GFX950         = 0x0950,
    GFX1030        = 0x1030,
    GFX1031        = 0x1031,
    GFX1032        = 0x1032,
    GFX1034        = 0x1034,
    GFX1035        = 0x1035,
    GFX1036        = 0x1036,
    GFX103_GENERIC = 0x103F,
    GFX1100        = 0x1100,
    GFX1101        = 0x1101,
    GFX1102        = 0x1102,
    GFX1103        = 0x1103,
    GFX1150        = 0x1150,
    GFX1151        = 0x1151,
    GFX1152        = 0x1152,
    GFX1153        = 0x1153,
    GFX11_GENERIC  = 0x11FF,
    GFX1200        = 0x1200,
    GFX1201        = 0x1201,
    GFX12_GENERIC  = 0x12FF,
    HOST           = 0x0000,
};

enum struct amdgcn_target_family_id
{
    GFX9    = 0x09,
    GFX10_3 = 0x10,
    GFX11   = 0x11,
    GFX12   = 0x12,
    HOST    = 0x00,
};

enum struct amdgcn_target_arch_id
{
    CDNA = 0x01,
    RDNA = 0x02,
    HOST = 0x00,
};

enum struct amdgcn_target_wave_size_id
{
    WAVE32 = 32u,
    WAVE64 = 64u,
    HOST   = 64u, // TODO: Is this correct? Should the host default to 64 or 1?
};

#if 1 //__cplusplus <= 201703L

template <amdgcn_target_id TargetId             = amdgcn_target_id::HOST,
          amdgcn_target_family_id FamilyId      = amdgcn_target_family_id::HOST,
          amdgcn_target_arch_id ArchId          = amdgcn_target_arch_id::HOST,
          amdgcn_target_wave_size_id WaveSizeId = amdgcn_target_wave_size_id::HOST>
struct amdgcn_target
{
    static constexpr amdgcn_target_id TARGET_ID              = TargetId;
    static constexpr amdgcn_target_family_id FAMILY_ID       = FamilyId;
    static constexpr amdgcn_target_arch_id ARCH_ID           = ArchId;
    static constexpr amdgcn_target_wave_size_id WAVE_SIZE_ID = WaveSizeId;
};

template <amdgcn_target_id targetId>
static constexpr auto make_amdgcn_gfx9_target()
{
    return amdgcn_target<targetId,
                         amdgcn_target_family_id::GFX9,
                         amdgcn_target_arch_id::CDNA,
                         amdgcn_target_wave_size_id::WAVE64>{};
}

template <amdgcn_target_id targetId>
static constexpr auto make_amdgcn_gfx10_3_target()
{
    return amdgcn_target<targetId,
                         amdgcn_target_family_id::GFX10_3,
                         amdgcn_target_arch_id::RDNA,
                         amdgcn_target_wave_size_id::WAVE32>{};
}

template <amdgcn_target_id targetId>
static constexpr auto make_amdgcn_gfx11_target()
{
    return amdgcn_target<targetId,
                         amdgcn_target_family_id::GFX11,
                         amdgcn_target_arch_id::RDNA,
                         amdgcn_target_wave_size_id::WAVE32>{};
}

template <amdgcn_target_id targetId>
static constexpr auto make_amdgcn_gfx12_target()
{
    return amdgcn_target<targetId,
                         amdgcn_target_family_id::GFX12,
                         amdgcn_target_arch_id::RDNA,
                         amdgcn_target_wave_size_id::WAVE32>{};
}

template <typename CompilerTarget, amdgcn_target_id... TargetIds>
static constexpr auto is_target_id_any_of()
{
    return is_any_value_of(CompilerTarget::TARGET_ID, TargetIds...);
}

template <typename CompilerTarget, amdgcn_target_family_id... FamilyIds>
static constexpr auto is_target_family_any_of()
{
    return is_any_value_of(CompilerTarget::FAMILY_ID, FamilyIds...);
}

template <typename CompilerTarget>
static constexpr bool is_target_family_gfx9()
{
    return CompilerTarget::FAMILY_ID == amdgcn_target_family_id::GFX9;
}

template <typename CompilerTarget>
static constexpr bool is_target_family_gfx10_3()
{
    return CompilerTarget::FAMILY_ID == amdgcn_target_family_id::GFX10_3;
}

template <typename CompilerTarget>
static constexpr bool is_target_family_gfx11()
{
    return CompilerTarget::FAMILY_ID == amdgcn_target_family_id::GFX11;
}

template <typename CompilerTarget>
static constexpr bool is_target_family_gfx12()
{
    return CompilerTarget::FAMILY_ID == amdgcn_target_family_id::GFX12;
}

template <typename CompilerTarget>
static constexpr bool is_target_arch_cdna()
{
    return CompilerTarget::ARCH_ID == amdgcn_target_arch_id::CDNA;
}

template <typename CompilerTarget>
static constexpr bool is_target_arch_rdna()
{
    return CompilerTarget::ARCH_ID == amdgcn_target_arch_id::RDNA;
}

template <typename CompilerTarget>
static constexpr bool is_target_wave_size_32()
{
    return CompilerTarget::WAVE_SIZE_ID == amdgcn_target_wave_size_id::WAVE32;
}

template <typename CompilerTarget>
static constexpr bool is_target_wave_size_64()
{
    return CompilerTarget::WAVE_SIZE_ID == amdgcn_target_wave_size_id::WAVE64;
}

// Helper to map compiler state to target arch id

#define MAP_COMPILER_STATE_TO_GFX9_TARGET(COMPILER_STATE, TARGET_ID)   \
    if constexpr(amdgcn_compiler_target_state::COMPILER_STATE)         \
    {                                                                  \
        return make_amdgcn_gfx9_target<amdgcn_target_id::TARGET_ID>(); \
    }                                                                  \
    else

#define MAP_COMPILER_STATE_TO_GFX10_3_TARGET(COMPILER_STATE, TARGET_ID)   \
    if constexpr(amdgcn_compiler_target_state::COMPILER_STATE)            \
    {                                                                     \
        return make_amdgcn_gfx10_3_target<amdgcn_target_id::TARGET_ID>(); \
    }                                                                     \
    else

#define MAP_COMPILER_STATE_TO_GFX11_TARGET(COMPILER_STATE, TARGET_ID)   \
    if constexpr(amdgcn_compiler_target_state::COMPILER_STATE)          \
    {                                                                   \
        return make_amdgcn_gfx11_target<amdgcn_target_id::TARGET_ID>(); \
    }                                                                   \
    else

#define MAP_COMPILER_STATE_TO_GFX12_TARGET(COMPILER_STATE, TARGET_ID)   \
    if constexpr(amdgcn_compiler_target_state::COMPILER_STATE)          \
    {                                                                   \
        return make_amdgcn_gfx12_target<amdgcn_target_id::TARGET_ID>(); \
    }                                                                   \
    else

/**
 * @brief Returns the amdgcn_target of the current compiler pass.
 * @note This is where we tie the compiler state to our internal target architecture representation
 * at compile time.
 */
constexpr auto get_compiler_target()
{
    MAP_COMPILER_STATE_TO_GFX9_TARGET(CK_TILE_ARCH_GFX908, GFX908);
    MAP_COMPILER_STATE_TO_GFX9_TARGET(CK_TILE_ARCH_GFX90A, GFX90A);
    MAP_COMPILER_STATE_TO_GFX9_TARGET(CK_TILE_ARCH_GFX942, GFX942);
    MAP_COMPILER_STATE_TO_GFX9_TARGET(CK_TILE_ARCH_GFX950, GFX950);
    MAP_COMPILER_STATE_TO_GFX10_3_TARGET(CK_TILE_ARCH_GFX1030, GFX1030);
    MAP_COMPILER_STATE_TO_GFX10_3_TARGET(CK_TILE_ARCH_GFX1031, GFX1031);
    MAP_COMPILER_STATE_TO_GFX10_3_TARGET(CK_TILE_ARCH_GFX1032, GFX1032);
    MAP_COMPILER_STATE_TO_GFX10_3_TARGET(CK_TILE_ARCH_GFX1034, GFX1034);
    MAP_COMPILER_STATE_TO_GFX10_3_TARGET(CK_TILE_ARCH_GFX1035, GFX1035);
    MAP_COMPILER_STATE_TO_GFX10_3_TARGET(CK_TILE_ARCH_GFX1036, GFX1036);
    MAP_COMPILER_STATE_TO_GFX10_3_TARGET(CK_TILE_ARCH_GFX10_3_GENERIC, GFX103_GENERIC);
    MAP_COMPILER_STATE_TO_GFX11_TARGET(CK_TILE_ARCH_GFX1100, GFX1100);
    MAP_COMPILER_STATE_TO_GFX11_TARGET(CK_TILE_ARCH_GFX1101, GFX1101);
    MAP_COMPILER_STATE_TO_GFX11_TARGET(CK_TILE_ARCH_GFX1102, GFX1102);
    MAP_COMPILER_STATE_TO_GFX11_TARGET(CK_TILE_ARCH_GFX1103, GFX1103);
    MAP_COMPILER_STATE_TO_GFX11_TARGET(CK_TILE_ARCH_GFX1150, GFX1150);
    MAP_COMPILER_STATE_TO_GFX11_TARGET(CK_TILE_ARCH_GFX1151, GFX1151);
    MAP_COMPILER_STATE_TO_GFX11_TARGET(CK_TILE_ARCH_GFX1152, GFX1152);
    MAP_COMPILER_STATE_TO_GFX11_TARGET(CK_TILE_ARCH_GFX1153, GFX1153);
    MAP_COMPILER_STATE_TO_GFX11_TARGET(CK_TILE_ARCH_GFX11_GENERIC, GFX11_GENERIC);
    MAP_COMPILER_STATE_TO_GFX12_TARGET(CK_TILE_ARCH_GFX1200, GFX1200);
    MAP_COMPILER_STATE_TO_GFX12_TARGET(CK_TILE_ARCH_GFX1201, GFX1201);
    MAP_COMPILER_STATE_TO_GFX12_TARGET(CK_TILE_ARCH_GFX12_GENERIC, GFX12_GENERIC);

    // Return HOST by default
    if constexpr(amdgcn_compiler_target_state::CK_TILE_HOST_COMPILE)
    {
        return amdgcn_target<>{};
    }
}

// Cleanup
#undef MAP_COMPILER_STATE_TO_GFX9_TARGET
#undef MAP_COMPILER_STATE_TO_GFX10_3_TARGET
#undef MAP_COMPILER_STATE_TO_GFX11_TARGET
#undef MAP_COMPILER_STATE_TO_GFX12_TARGET

// Sanity check: device compile must have a valid target architecture
static_assert(!amdgcn_compiler_target_state::CK_TILE_DEVICE_COMPILE ||
                  get_compiler_target().TARGET_ID != amdgcn_target_id::HOST,
              "Device compile must have a valid target device architecture");

// Sanity check: host compile must have HOST target architecture
static_assert(!amdgcn_compiler_target_state::CK_TILE_HOST_COMPILE ||
                  get_compiler_target().TARGET_ID == amdgcn_target_id::HOST,
              "Host compile must target HOST architecture");

// TODO: c++20 use the make functions and constexpr if to avoid string construction and find at
// runtime
#define MAP_HIP_DEVICE_PROP_GCN_ARCH_NAME_STRING_TO_TARGET_ID(NAME_STRING, TARGET_ID) \
    if(str.find(NAME_STRING) != std::string::npos)                                    \
    {                                                                                 \
        return amdgcn_target_id::TARGET_ID;                                           \
    }                                                                                 \
    else

/**
 * @brief Converts a lower-case string to the corresponding amdgcn_target_arch_id value.
 *        Returns amdgcn_target_arch_id::HOST if no match is found.
 *        Matches if the input contains the architecture substring.
 *        Example: "gfx908", "gfx90a", "gfx1100", etc. can be parsed from hip runtime info.
 */
// TODO: c++20 constexpr if and string_view to avoid std::string construction and find at runtime
// TODO: c++20 return amdgcn_target instance instead of just the target id
CK_TILE_HOST auto hip_device_prop_gcn_arch_name_to_amdgcn_target_id(char const* testStr)
{
    auto str = std::string(testStr);
    MAP_HIP_DEVICE_PROP_GCN_ARCH_NAME_STRING_TO_TARGET_ID("gfx908", GFX908);
    MAP_HIP_DEVICE_PROP_GCN_ARCH_NAME_STRING_TO_TARGET_ID("gfx90a", GFX90A);
    MAP_HIP_DEVICE_PROP_GCN_ARCH_NAME_STRING_TO_TARGET_ID("gfx942", GFX942);
    MAP_HIP_DEVICE_PROP_GCN_ARCH_NAME_STRING_TO_TARGET_ID("gfx950", GFX950);
    MAP_HIP_DEVICE_PROP_GCN_ARCH_NAME_STRING_TO_TARGET_ID("gfx1030", GFX1030);
    MAP_HIP_DEVICE_PROP_GCN_ARCH_NAME_STRING_TO_TARGET_ID("gfx1031", GFX1031);
    MAP_HIP_DEVICE_PROP_GCN_ARCH_NAME_STRING_TO_TARGET_ID("gfx1032", GFX1032);
    MAP_HIP_DEVICE_PROP_GCN_ARCH_NAME_STRING_TO_TARGET_ID("gfx1034", GFX1034);
    MAP_HIP_DEVICE_PROP_GCN_ARCH_NAME_STRING_TO_TARGET_ID("gfx1035", GFX1035);
    MAP_HIP_DEVICE_PROP_GCN_ARCH_NAME_STRING_TO_TARGET_ID("gfx1036", GFX1036);
    MAP_HIP_DEVICE_PROP_GCN_ARCH_NAME_STRING_TO_TARGET_ID("gfx10_3_generic", GFX103_GENERIC);
    MAP_HIP_DEVICE_PROP_GCN_ARCH_NAME_STRING_TO_TARGET_ID("gfx1100", GFX1100);
    MAP_HIP_DEVICE_PROP_GCN_ARCH_NAME_STRING_TO_TARGET_ID("gfx1101", GFX1101);
    MAP_HIP_DEVICE_PROP_GCN_ARCH_NAME_STRING_TO_TARGET_ID("gfx1102", GFX1102);
    MAP_HIP_DEVICE_PROP_GCN_ARCH_NAME_STRING_TO_TARGET_ID("gfx1103", GFX1103);
    MAP_HIP_DEVICE_PROP_GCN_ARCH_NAME_STRING_TO_TARGET_ID("gfx1150", GFX1150);
    MAP_HIP_DEVICE_PROP_GCN_ARCH_NAME_STRING_TO_TARGET_ID("gfx1151", GFX1151);
    MAP_HIP_DEVICE_PROP_GCN_ARCH_NAME_STRING_TO_TARGET_ID("gfx1152", GFX1152);
    MAP_HIP_DEVICE_PROP_GCN_ARCH_NAME_STRING_TO_TARGET_ID("gfx1153", GFX1153);
    MAP_HIP_DEVICE_PROP_GCN_ARCH_NAME_STRING_TO_TARGET_ID("gfx11_generic", GFX11_GENERIC);
    MAP_HIP_DEVICE_PROP_GCN_ARCH_NAME_STRING_TO_TARGET_ID("gfx1200", GFX1200);
    MAP_HIP_DEVICE_PROP_GCN_ARCH_NAME_STRING_TO_TARGET_ID("gfx1201", GFX1201);
    MAP_HIP_DEVICE_PROP_GCN_ARCH_NAME_STRING_TO_TARGET_ID("gfx12_generic", GFX12_GENERIC);

    // Default case: return HOST target if no match is found
    return amdgcn_target_id::HOST;
}

#undef MAP_HIP_DEVICE_PROP_GCN_ARCH_NAME_STRING_TO_TARGET_ID

/**
 * @brief SFINAE enabler for a compiler target if the target id is in the list of supported target
 * ids
 * @tparam CompilerTarget The compiler target to check
 * @tparam SupportedTargetIds The list of supported target ids, e.g., amdgcn_target_id::GFX908
 */
template <typename CompilerTarget, amdgcn_target_id... SupportedTargetIds>
using enable_if_target_id_t =
    std::enable_if_t<is_any_value_of(CompilerTarget::TARGET_ID, SupportedTargetIds...)>;

/**
 * @brief SFINAE enabler for a compiler target if the family id is in the list of supported family
 * ids
 * @tparam CompilerTarget The compiler target to check
 * @tparam SupportedTargetFamilyIds The list of supported family ids, e.g.,
 * amdgcn_target_family_id::GFX9
 */
template <typename CompilerTarget, amdgcn_target_family_id... SupportedTargetFamilyIds>
using enable_if_target_family_id_t =
    std::enable_if_t<is_any_value_of(CompilerTarget::FAMILY_ID, SupportedTargetFamilyIds...)>;

/**
 * @brief SFINAE enabler for a compiler target if the arch id is in the list of supported arch ids
 * @tparam CompilerTarget The compiler target to check
 * @tparam SupportedTargetArchIds The list of supported arch ids, e.g., amdgcn_target_arch_id::CDNA
 */
template <typename CompilerTarget, amdgcn_target_arch_id... SupportedTargetArchIds>
using enable_if_target_arch_id_t =
    std::enable_if_t<is_any_value_of(CompilerTarget::ARCH_ID, SupportedTargetArchIds...)>;

/**
 * @brief SFINAE enabler for a compiler target if the wave size id is in the list of supported wave
 * size ids
 * @tparam CompilerTarget The compiler target to check
 * @tparam SupportedTargetWaveSizeIds The list of supported wave size ids, e.g.,
 * amdgcn_target_wave_size_id::WAVE64
 */
template <typename CompilerTarget, amdgcn_target_wave_size_id... SupportedTargetWaveSizeIds>
using enable_if_target_wave_size_id_t =
    std::enable_if_t<is_any_value_of(CompilerTarget::WAVE_SIZE_ID, SupportedTargetWaveSizeIds...)>;

/// Specialized enablers for common families, architectures, and wave sizes ///

/**
 * @brief SFINAE enabler for GFX9 family targets
 * @tparam CompilerTarget The compiler target to check
 */
template <typename CompilerTarget>
using enable_if_target_family_gfx9_t =
    enable_if_target_family_id_t<CompilerTarget, amdgcn_target_family_id::GFX9>;

/**
 * @brief SFINAE enabler for GFX10.3 family targets
 * @tparam CompilerTarget The compiler target to check
 */
template <typename CompilerTarget>
using enable_if_target_family_gfx10_3_t =
    enable_if_target_family_id_t<CompilerTarget, amdgcn_target_family_id::GFX10_3>;

/**
 * @brief SFINAE enabler for GFX11 family targets
 * @tparam CompilerTarget The compiler target to check
 */
template <typename CompilerTarget>
using enable_if_target_family_gfx11_t =
    enable_if_target_family_id_t<CompilerTarget, amdgcn_target_family_id::GFX11>;

/**
 * @brief SFINAE enabler for GFX12 family targets
 * @tparam CompilerTarget The compiler target to check
 */
template <typename CompilerTarget>
using enable_if_target_family_gfx12_t =
    enable_if_target_family_id_t<CompilerTarget, amdgcn_target_family_id::GFX12>;

/**
 * @brief SFINAE enabler for CDNA architecture targets
 * @tparam CompilerTarget The compiler target to check
 */
template <typename CompilerTarget>
using enable_if_target_arch_cdna_t =
    enable_if_target_arch_id_t<CompilerTarget, amdgcn_target_arch_id::CDNA>;

/**
 * @brief SFINAE enabler for RDNA architecture targets
 * @tparam CompilerTarget The compiler target to check
 */
template <typename CompilerTarget>
using enable_if_target_arch_rdna_t =
    enable_if_target_arch_id_t<CompilerTarget, amdgcn_target_arch_id::RDNA>;

/**
 * @brief SFINAE enabler for WAVE32 size targets
 * @tparam CompilerTarget The compiler target to check
 */
template <typename CompilerTarget>
using enable_if_target_wave32_t =
    enable_if_target_wave_size_id_t<CompilerTarget, amdgcn_target_wave_size_id::WAVE32>;

/**
 * @brief SFINAE enabler for WAVE64 size targets
 * @tparam CompilerTarget The compiler target to check
 */
template <typename CompilerTarget>
using enable_if_target_wave64_t =
    enable_if_target_wave_size_id_t<CompilerTarget, amdgcn_target_wave_size_id::WAVE64>;

#elif __cplusplus >= 202002L

struct amdgcn_target
{
    // Target architecture identifiers
    // These are set to HOST (0) by default
    // TARGET_ID is the specific architecture id (e.g., GFX908)
    // FAMILY_ID is the architecture family id (e.g., GFX9)
    // ARCH_ID is the architecture class id (e.g., CDNA, RDNA)
    // WAVE_SIZE_ID is the wavefront size id (e.g., WAVE32, WAVE64)
    const amdgcn_target_id TARGET_ID              = amdgcn_target_id::HOST;
    const amdgcn_target_family_id FAMILY_ID       = amdgcn_target_family_id::HOST;
    const amdgcn_target_arch_id ARCH_ID           = amdgcn_target_arch_id::HOST;
    const amdgcn_target_wave_size_id WAVE_SIZE_ID = amdgcn_target_wave_size_id::HOST;
};

static constexpr auto make_amdgcn_gfx10_3_target(amdgcn_target_id targetId)
{
    return amdgcn_target{.TARGET_ID    = targetId,
                         .FAMILY_ID    = amdgcn_target_family_id::GFX10_3,
                         .ARCH_ID      = amdgcn_target_arch_id::RDNA,
                         .WAVE_SIZE_ID = amdgcn_target_wave_size_id::WAVE32};
}

static constexpr auto make_amdgcn_gfx9_target(amdgcn_target_id targetId)
{
    return amdgcn_target{.TARGET_ID    = targetId,
                         .FAMILY_ID    = amdgcn_target_family_id::GFX9,
                         .ARCH_ID      = amdgcn_target_arch_id::CDNA,
                         .WAVE_SIZE_ID = amdgcn_target_wave_size_id::WAVE64};
}

static constexpr auto make_amdgcn_gfx11_target(amdgcn_target_id targetId)
{
    return amdgcn_target{.TARGET_ID    = targetId,
                         .FAMILY_ID    = amdgcn_target_family_id::GFX11,
                         .ARCH_ID      = amdgcn_target_arch_id::RDNA,
                         .WAVE_SIZE_ID = amdgcn_target_wave_size_id::WAVE32};
}

static constexpr auto make_amdgcn_gfx12_target(amdgcn_target_id targetId)
{
    return amdgcn_target{.TARGET_ID    = targetId,
                         .FAMILY_ID    = amdgcn_target_family_id::GFX12,
                         .ARCH_ID      = amdgcn_target_arch_id::RDNA,
                         .WAVE_SIZE_ID = amdgcn_target_wave_size_id::WAVE32};
}

static constexpr bool is_target_family_gfx9(amdgcn_target target)
{
    return target.FAMILY_ID == amdgcn_target_family_id::GFX9;
}

static constexpr bool is_target_family_gfx10_3(amdgcn_target target)
{
    return target.FAMILY_ID == amdgcn_target_family_id::GFX10_3;
}

static constexpr bool is_target_family_gfx11(amdgcn_target target)
{
    return target.FAMILY_ID == amdgcn_target_family_id::GFX11;
}

static constexpr bool is_target_family_gfx12(amdgcn_target target)
{
    return target.FAMILY_ID == amdgcn_target_family_id::GFX12;
}

static constexpr bool is_target_arch_cdna(amdgcn_target target)
{
    return target.ARCH_ID == amdgcn_target_arch_id::CDNA;
}

static constexpr bool is_target_arch_rdna(amdgcn_target target)
{
    return target.ARCH_ID == amdgcn_target_arch_id::RDNA;
}

static constexpr bool is_target_wave_size_32(amdgcn_target target)
{
    return target.WAVE_SIZE_ID == amdgcn_target_wave_size_id::WAVE32;
}

static constexpr bool is_target_wave_size_64(amdgcn_target target)
{
    return target.WAVE_SIZE_ID == amdgcn_target_wave_size_id::WAVE64;
}

// Helper to map compiler state to target arch id
#define MAP_COMPILER_STATE_TO_GFX10_3_TARGET(COMPILER_STATE, TARGET_ID) \
    if constexpr(amdgcn_compiler_target_state::COMPILER_STATE)          \
    {                                                                   \
        return make_amdgcn_gfx9_target(amdgcn_target_id::TARGET_ID);    \
    }

#define MAP_COMPILER_STATE_TO_GFX9_TARGET(COMPILER_STATE, TARGET_ID) \
    if constexpr(amdgcn_compiler_target_state::COMPILER_STATE)       \
    {                                                                \
        return make_amdgcn_gfx9_target(amdgcn_target_id::TARGET_ID); \
    }

#define MAP_COMPILER_STATE_TO_GFX11_TARGET(COMPILER_STATE, TARGET_ID) \
    if constexpr(amdgcn_compiler_target_state::COMPILER_STATE)        \
    {                                                                 \
        return make_amdgcn_gfx11_target(amdgcn_target_id::TARGET_ID); \
    }

#define MAP_COMPILER_STATE_TO_GFX12_TARGET(COMPILER_STATE, TARGET_ID) \
    if constexpr(amdgcn_compiler_target_state::COMPILER_STATE)        \
    {                                                                 \
        return make_amdgcn_gfx12_target(amdgcn_target_id::TARGET_ID); \
    }

/*! @brief Returns the amdgcn_target of the current compiler pass.
 * @note This is where we tie the compiler state to our internal target architecture representation
 * at compile time.
 */
CK_TILE_HOST_DEVICE constexpr auto get_compiler_target()
{
    MAP_COMPILER_STATE_TO_GFX9_TARGET(CK_TILE_ARCH_GFX908, GFX908);
    MAP_COMPILER_STATE_TO_GFX9_TARGET(CK_TILE_ARCH_GFX90A, GFX90A);
    MAP_COMPILER_STATE_TO_GFX9_TARGET(CK_TILE_ARCH_GFX942, GFX942);
    MAP_COMPILER_STATE_TO_GFX9_TARGET(CK_TILE_ARCH_GFX950, GFX950);
    MAP_COMPILER_STATE_TO_GFX10_3_TARGET(CK_TILE_ARCH_GFX1030, GFX1030);
    MAP_COMPILER_STATE_TO_GFX10_3_TARGET(CK_TILE_ARCH_GFX1031, GFX1031);
    MAP_COMPILER_STATE_TO_GFX10_3_TARGET(CK_TILE_ARCH_GFX1032, GFX1032);
    MAP_COMPILER_STATE_TO_GFX10_3_TARGET(CK_TILE_ARCH_GFX1034, GFX1034);
    MAP_COMPILER_STATE_TO_GFX10_3_TARGET(CK_TILE_ARCH_GFX1035, GFX1035);
    MAP_COMPILER_STATE_TO_GFX10_3_TARGET(CK_TILE_ARCH_GFX1036, GFX1036);
    MAP_COMPILER_STATE_TO_GFX10_3_TARGET(CK_TILE_ARCH_GFX10_3_GENERIC, GFX103_GENERIC);
    MAP_COMPILER_STATE_TO_GFX11_TARGET(CK_TILE_ARCH_GFX1100, GFX1100);
    MAP_COMPILER_STATE_TO_GFX11_TARGET(CK_TILE_ARCH_GFX1101, GFX1101);
    MAP_COMPILER_STATE_TO_GFX11_TARGET(CK_TILE_ARCH_GFX1102, GFX1102);
    MAP_COMPILER_STATE_TO_GFX11_TARGET(CK_TILE_ARCH_GFX1103, GFX1103);
    MAP_COMPILER_STATE_TO_GFX11_TARGET(CK_TILE_ARCH_GFX1150, GFX1150);
    MAP_COMPILER_STATE_TO_GFX11_TARGET(CK_TILE_ARCH_GFX1151, GFX1151);
    MAP_COMPILER_STATE_TO_GFX11_TARGET(CK_TILE_ARCH_GFX1152, GFX1152);
    MAP_COMPILER_STATE_TO_GFX11_TARGET(CK_TILE_ARCH_GFX1153, GFX1153);
    MAP_COMPILER_STATE_TO_GFX11_TARGET(CK_TILE_ARCH_GFX11_GENERIC, GFX11_GENERIC);
    MAP_COMPILER_STATE_TO_GFX12_TARGET(CK_TILE_ARCH_GFX1200, GFX1200);
    MAP_COMPILER_STATE_TO_GFX12_TARGET(CK_TILE_ARCH_GFX1201, GFX1201);
    MAP_COMPILER_STATE_TO_GFX12_TARGET(CK_TILE_ARCH_GFX12_GENERIC, GFX12_GENERIC);

    // Default to HOST
    return amdgcn_target{};
}

// Cleanup
#undef MAP_COMPILER_STATE_TO_GFX9_TARGET
#undef MAP_COMPILER_STATE_TO_GFX10_3_TARGET
#undef MAP_COMPILER_STATE_TO_GFX11_TARGET
#undef MAP_COMPILER_STATE_TO_GFX12_TARGET

// Sanity check: device compile must have a valid target architecture
static_assert(!amdgcn_compiler_target_state::CK_TILE_DEVICE_COMPILE ||
                  get_compiler_target().TARGET_ID != amdgcn_target_id::HOST,
              "Device compile must have a valid target device architecture");

// Sanity check: host compile must have HOST target architecture
static_assert(!amdgcn_compiler_target_state::CK_TILE_HOST_COMPILE ||
                  get_compiler_target().TARGET_ID == amdgcn_target_id::HOST,
              "Host compile must target HOST architecture");

#define MAP_HIP_DEVICE_PROP_GCN_ARCH_NAME_STRING_TO_GFX9_TARGET(NAME_STRING, TARGET_ID) \
    if constexpr(str.find(NAME_STRING) != std::string::npos)                            \
    {                                                                                   \
        return make_amdgcn_gfx9_target(amdgcn_target_id::TARGET_ID);                    \
    }                                                                                   \
    else

#define MAP_HIP_DEVICE_PROP_GCN_ARCH_NAME_STRING_TO_GFX10_3_TARGET(NAME_STRING, TARGET_ID) \
    if constexpr(str.find(NAME_STRING) != std::string::npos)                               \
    {                                                                                      \
        return make_amdgcn_gfx10_3_target(amdgcn_target_id::TARGET_ID);                    \
    }                                                                                      \
    else

#define MAP_HIP_DEVICE_PROP_GCN_ARCH_NAME_STRING_TO_GFX11_TARGET(NAME_STRING, TARGET_ID) \
    if constexpr(str.find(NAME_STRING) != std::string::npos)                             \
    {                                                                                    \
        return make_amdgcn_gfx11_target(amdgcn_target_id::TARGET_ID);                    \
    }                                                                                    \
    else

#define MAP_HIP_DEVICE_PROP_GCN_ARCH_NAME_STRING_TO_GFX12_TARGET(NAME_STRING, TARGET_ID) \
    if constexpr(str.find(NAME_STRING) != std::string::npos)                             \
    {                                                                                    \
        return make_amdgcn_gfx12_target(amdgcn_target_id::TARGET_ID);                    \
    }                                                                                    \
    else

/**
 * @brief Converts a lower-case string to the corresponding amdgcn_target_arch_id value.
 *        Returns amdgcn_target_arch_id::HOST if no match is found.
 *        Matches if the input contains the architecture substring.
 *        Example: "gfx908", "gfx90a", "gfx1100", etc. can be parsed from hip runtime info.
 */
CK_TILE_HOST auto hip_device_prop_gcn_arch_name_to_amdgcn_target(char const* testStr)
{
    auto str = std::string(testStr);
    MAP_HIP_DEVICE_PROP_GCN_ARCH_NAME_STRING_TO_GFX9_TARGET("gfx908", GFX908);
    MAP_HIP_DEVICE_PROP_GCN_ARCH_NAME_STRING_TO_GFX9_TARGET("gfx90a", GFX90A);
    MAP_HIP_DEVICE_PROP_GCN_ARCH_NAME_STRING_TO_GFX9_TARGET("gfx942", GFX942);
    MAP_HIP_DEVICE_PROP_GCN_ARCH_NAME_STRING_TO_GFX9_TARGET("gfx950", GFX950);
    MAP_HIP_DEVICE_PROP_GCN_ARCH_NAME_STRING_TO_GFX10_3_TARGET("gfx1030", GFX1030);
    MAP_HIP_DEVICE_PROP_GCN_ARCH_NAME_STRING_TO_GFX10_3_TARGET("gfx1031", GFX1031);
    MAP_HIP_DEVICE_PROP_GCN_ARCH_NAME_STRING_TO_GFX10_3_TARGET("gfx1032", GFX1032);
    MAP_HIP_DEVICE_PROP_GCN_ARCH_NAME_STRING_TO_GFX10_3_TARGET("gfx1034", GFX1034);
    MAP_HIP_DEVICE_PROP_GCN_ARCH_NAME_STRING_TO_GFX10_3_TARGET("gfx1035", GFX1035);
    MAP_HIP_DEVICE_PROP_GCN_ARCH_NAME_STRING_TO_GFX10_3_TARGET("gfx1036", GFX1036);
    MAP_HIP_DEVICE_PROP_GCN_ARCH_NAME_STRING_TO_GFX10_3_TARGET("gfx10_3_generic", GFX103_GENERIC);
    MAP_HIP_DEVICE_PROP_GCN_ARCH_NAME_STRING_TO_GFX11_TARGET("gfx1100", GFX1100);
    MAP_HIP_DEVICE_PROP_GCN_ARCH_NAME_STRING_TO_GFX11_TARGET("gfx1101", GFX1101);
    MAP_HIP_DEVICE_PROP_GCN_ARCH_NAME_STRING_TO_GFX11_TARGET("gfx1102", GFX1102);
    MAP_HIP_DEVICE_PROP_GCN_ARCH_NAME_STRING_TO_GFX11_TARGET("gfx1103", GFX1103);
    MAP_HIP_DEVICE_PROP_GCN_ARCH_NAME_STRING_TO_GFX11_TARGET("gfx1150", GFX1150);
    MAP_HIP_DEVICE_PROP_GCN_ARCH_NAME_STRING_TO_GFX11_TARGET("gfx1151", GFX1151);
    MAP_HIP_DEVICE_PROP_GCN_ARCH_NAME_STRING_TO_GFX11_TARGET("gfx1152", GFX1152);
    MAP_HIP_DEVICE_PROP_GCN_ARCH_NAME_STRING_TO_GFX11_TARGET("gfx1153", GFX1153);
    MAP_HIP_DEVICE_PROP_GCN_ARCH_NAME_STRING_TO_GFX11_TARGET("gfx11_generic", GFX11_GENERIC);
    MAP_HIP_DEVICE_PROP_GCN_ARCH_NAME_STRING_TO_GFX12_TARGET("gfx1200", GFX1200);
    MAP_HIP_DEVICE_PROP_GCN_ARCH_NAME_STRING_TO_GFX12_TARGET("gfx1201", GFX1201);
    MAP_HIP_DEVICE_PROP_GCN_ARCH_NAME_STRING_TO_GFX12_TARGET("gfx12_generic", GFX12_GENERIC);

    // Default case
    return amdgcn_target{};
}

#undef MAP_HIP_DEVICE_PROP_GCN_ARCH_NAME_STRING_TO_GFX9_TARGET
#undef MAP_HIP_DEVICE_PROP_GCN_ARCH_NAME_STRING_TO_GFX10_3_TARGET
#undef MAP_HIP_DEVICE_PROP_GCN_ARCH_NAME_STRING_TO_GFX11_TARGET
#undef MAP_HIP_DEVICE_PROP_GCN_ARCH_NAME_STRING_TO_GFX12_TARGET

/**
 * @brief SFINAE enabler for a compiler target if the target id is in the list of supported target
 * ids
 * @tparam CompilerTarget The compiler target to check
 * @tparam SupportedTargetIds The list of supported target ids, e.g., amdgcn_target_id::GFX908
 */
template <amdgcn_target CompilerTarget, amdgcn_target_id... SupportedTargetIds>
using enable_if_target_id_t =
    std::enable_if_t<is_any_value_of(CompilerTarget.TARGET_ID, SupportedTargetIds...)>;

/**
 * @brief SFINAE enabler for a compiler target if the family id is in the list of supported family
 * ids
 * @tparam CompilerTarget The compiler target to check
 * @tparam SupportedTargetFamilyIds The list of supported family ids, e.g.,
 * amdgcn_target_family_id::GFX9
 */
template <amdgcn_target CompilerTarget, amdgcn_target_family_id... SupportedTargetFamilyIds>
using enable_if_target_family_id_t =
    std::enable_if_t<is_any_value_of(CompilerTarget.FAMILY_ID, SupportedTargetFamilyIds...)>;

/**
 * @brief SFINAE enabler for a compiler target if the arch id is in the list of supported arch ids
 * @tparam CompilerTarget The compiler target to check
 * @tparam SupportedTargetArchIds The list of supported arch ids, e.g., amdgcn_target_arch_id::CDNA
 */
template <amdgcn_target CompilerTarget, amdgcn_target_arch_id... SupportedTargetArchIds>
using enable_if_target_arch_id_t =
    std::enable_if_t<is_any_value_of(CompilerTarget.ARCH_ID, SupportedTargetArchIds...)>;

/**
 * @brief SFINAE enabler for a compiler target if the wave size id is in the list of supported wave
 * size ids
 * @tparam CompilerTarget The compiler target to check
 * @tparam SupportedTargetWaveSizeIds The list of supported wave size ids, e.g.,
 * amdgcn_target_wave_size_id::WAVE64
 */
template <amdgcn_target CompilerTarget, amdgcn_target_wave_size_id... SupportedTargetWaveSizeIds>
using enable_if_target_wave_size_id_t =
    std::enable_if_t<is_any_value_of(CompilerTarget.WAVE_SIZE_ID, SupportedTargetWaveSizeIds...)>;

/// Specialized enablers for common families, architectures, and wave sizes ///

/**
 * @brief SFINAE enabler for GFX9 family targets
 * @tparam CompilerTarget The compiler target to check
 */
template <amdgcn_target CompilerTarget>
using enable_if_target_family_gfx9_t =
    enable_if_target_family_id_t<CompilerTarget, amdgcn_target_family_id::GFX9>;

/**
 * @brief SFINAE enabler for GFX10.3 family targets
 * @tparam CompilerTarget The compiler target to check
 */
template <amdgcn_target CompilerTarget>
using enable_if_target_family_gfx10_3_t =
    enable_if_target_family_id_t<CompilerTarget, amdgcn_target_family_id::GFX10_3>;

/**
 * @brief SFINAE enabler for GFX11 family targets
 * @tparam CompilerTarget The compiler target to check
 */
template <amdgcn_target CompilerTarget>
using enable_if_target_family_gfx11_t =
    enable_if_target_family_id_t<CompilerTarget, amdgcn_target_family_id::GFX11>;

/**
 * @brief SFINAE enabler for GFX12 family targets
 * @tparam CompilerTarget The compiler target to check
 */
template <amdgcn_target CompilerTarget>
using enable_if_target_family_gfx12_t =
    enable_if_target_family_id_t<CompilerTarget, amdgcn_target_family_id::GFX12>;

/**
 * @brief SFINAE enabler for CDNA architecture targets
 * @tparam CompilerTarget The compiler target to check
 */
template <amdgcn_target CompilerTarget>
using enable_if_target_arch_cdna_t =
    enable_if_target_arch_id_t<CompilerTarget, amdgcn_target_arch_id::CDNA>;

/**
 * @brief SFINAE enabler for RDNA architecture targets
 * @tparam CompilerTarget The compiler target to check
 */
template <amdgcn_target CompilerTarget>
using enable_if_target_arch_rdna_t =
    enable_if_target_arch_id_t<CompilerTarget, amdgcn_target_arch_id::RDNA>;

/**
 * @brief SFINAE enabler for WAVE32 size targets
 * @tparam CompilerTarget The compiler target to check
 */
template <amdgcn_target CompilerTarget>
using enable_if_target_wave32_t =
    enable_if_target_wave_size_id_t<CompilerTarget, amdgcn_target_wave_size_id::WAVE32>;

/**
 * @brief SFINAE enabler for WAVE64 size targets
 * @tparam CompilerTarget The compiler target to check
 */
template <amdgcn_target CompilerTarget>
using enable_if_target_wave64_t =
    enable_if_target_wave_size_id_t<CompilerTarget, amdgcn_target_wave_size_id::WAVE64>;

#endif // __cplusplus <= 201703L

} // namespace core::arch

CK_TILE_HOST bool is_wave32()
{
    hipDeviceProp_t props{};
    int device;
    auto status = hipGetDevice(&device);
    if(status != hipSuccess)
    {
        return false;
    }
    status = hipGetDeviceProperties(&props, device);
    if(status != hipSuccess)
    {
        return false;
    }
    return props.major > 9;
}

/*! @brief Returns the amdgcn_wave_size of the current compiler pass
 */
CK_TILE_HOST_DEVICE constexpr index_t get_warp_size()
{
    return static_cast<index_t>(core::arch::get_compiler_target().WAVE_SIZE_ID);
}

CK_TILE_DEVICE index_t get_grid_size() { return gridDim.x; }

CK_TILE_DEVICE index_t get_block_size() { return blockDim.x; }

// TODO: deprecate these
CK_TILE_DEVICE index_t get_thread_local_1d_id() { return threadIdx.x; }

CK_TILE_DEVICE index_t get_thread_global_1d_id() { return blockIdx.x * blockDim.x + threadIdx.x; }

CK_TILE_DEVICE index_t get_block_1d_id() { return blockIdx.x; }

// Use these instead
CK_TILE_DEVICE index_t get_lane_id() { return __lane_id(); }

template <bool ReturnSgpr = true>
CK_TILE_DEVICE index_t get_warp_id(bool_constant<ReturnSgpr> = {})
{
    const index_t warp_id = threadIdx.x / get_warp_size();
    if constexpr(ReturnSgpr)
    {
        return amd_wave_read_first_lane(warp_id);
    }
    else
    {
        return warp_id;
    }
}

CK_TILE_DEVICE index_t get_thread_id() { return threadIdx.x; }

CK_TILE_DEVICE index_t get_block_id() { return blockIdx.x; }

CK_TILE_DEVICE void block_sync_load_raw(index_t cnt = 0)
{
#ifdef __gfx12__
    asm volatile("s_wait_loadcnt %0 \n"
                 "s_barrier_signal -1 \n"
                 "s_barrier_wait -1"
                 :
                 : "n"(cnt)
                 : "memory");
#else
    asm volatile("s_waitcnt vmcnt(%0) \n"
                 "s_barrier"
                 :
                 : "n"(cnt)
                 : "memory");
#endif
}

struct WaitcntLayoutGfx12
{ // s_wait_loadcnt_dscnt: mem[13:8], ds[5:0]
    CK_TILE_DEVICE static constexpr index_t VM_MASK   = 0x3F; // mem
    CK_TILE_DEVICE static constexpr index_t LGKM_MASK = 0x3F; // ds
    CK_TILE_DEVICE static constexpr bool HAS_EXP      = false;

    CK_TILE_DEVICE static constexpr index_t pack_vm(index_t c) { return ((c & VM_MASK) << 8); }
    CK_TILE_DEVICE static constexpr index_t pack_lgkm(index_t c) { return ((c & LGKM_MASK) << 0); }
    CK_TILE_DEVICE static constexpr index_t pack_exp(index_t) { return 0; }
};

struct WaitcntLayoutGfx11
{ // vm[15:10] (6), lgkm[9:4] (6), exp unused
    CK_TILE_DEVICE static constexpr index_t VM_MASK   = 0x3F;
    CK_TILE_DEVICE static constexpr index_t LGKM_MASK = 0x3F;
    CK_TILE_DEVICE static constexpr bool HAS_EXP      = false;

    CK_TILE_DEVICE static constexpr index_t pack_vm(index_t c) { return ((c & VM_MASK) << 10); }
    CK_TILE_DEVICE static constexpr index_t pack_lgkm(index_t c) { return ((c & LGKM_MASK) << 4); }
    CK_TILE_DEVICE static constexpr index_t pack_exp(index_t) { return 0; }
};

struct WaitcntLayoutLegacy
{ // FE'DC'BA98'7'654'3210 => VV'UU'LLLL'U'EEE'VVVV
    CK_TILE_DEVICE static constexpr index_t VM_MASK   = 0x3F; // split: low4 + hi2
    CK_TILE_DEVICE static constexpr index_t LGKM_MASK = 0x0F; // [11:8]
    CK_TILE_DEVICE static constexpr index_t EXP_MASK  = 0x07; // [6:4]
    CK_TILE_DEVICE static constexpr bool HAS_EXP      = true;

    CK_TILE_DEVICE static constexpr index_t pack_vm(index_t c)
    {
        c &= VM_MASK;
        return ((c & 0xF) << 0) | ((c & 0x30) << 10);
    }
    CK_TILE_DEVICE static constexpr index_t pack_lgkm(index_t c) { return ((c & LGKM_MASK) << 8); }
    CK_TILE_DEVICE static constexpr index_t pack_exp(index_t c) { return ((c & EXP_MASK) << 4); }
};

// Select active layout
#if defined(__gfx12__)
using Waitcnt = WaitcntLayoutGfx12;
#elif defined(__gfx11__)
using Waitcnt = WaitcntLayoutGfx11;
#else
using Waitcnt = WaitcntLayoutLegacy;
#endif

//----------------------------------------------
// Public API: only from_* (constexpr templates)
//----------------------------------------------
struct waitcnt_arg
{
    // kMax* exposed for callers; match field widths per-arch
#if defined(__gfx12__) || defined(__gfx11__)
    CK_TILE_DEVICE static constexpr index_t kMaxVmCnt   = 0x3F; // 6 bits
    CK_TILE_DEVICE static constexpr index_t kMaxLgkmCnt = 0x3F; // 6 bits
    CK_TILE_DEVICE static constexpr index_t kMaxExpCnt  = 0x0;  // none
#else
    CK_TILE_DEVICE static constexpr index_t kMaxVmCnt   = 0x3F; // 6 bits (split)
    CK_TILE_DEVICE static constexpr index_t kMaxLgkmCnt = 0x0F; // 4 bits
    CK_TILE_DEVICE static constexpr index_t kMaxExpCnt  = 0x07; // 3 bits
#endif

    template <index_t cnt>
    CK_TILE_DEVICE static constexpr index_t from_vmcnt()
    {
        static_assert((cnt & ~Waitcnt::VM_MASK) == 0, "vmcnt out of range");
        return Waitcnt::pack_vm(cnt);
    }

    template <index_t cnt>
    CK_TILE_DEVICE static constexpr index_t from_lgkmcnt()
    {
        static_assert((cnt & ~Waitcnt::LGKM_MASK) == 0, "lgkmcnt out of range");
        return Waitcnt::pack_lgkm(cnt);
    }

    template <index_t cnt>
    CK_TILE_DEVICE static constexpr index_t from_expcnt()
    {
        if constexpr(Waitcnt::HAS_EXP)
        {
            // EXP_MASK only exists on legacy
#if !defined(__gfx12__) && !defined(__gfx11__)
            static_assert((cnt & ~Waitcnt::EXP_MASK) == 0, "expcnt out of range");
            return Waitcnt::pack_exp(cnt);
#else
            (void)cnt;
            return 0;
#endif
        }
        else
        {
            static_assert(cnt == 0, "expcnt unsupported on this arch");
            return 0;
        }
    }
};

template <index_t vmcnt   = waitcnt_arg::kMaxVmCnt,
          index_t expcnt  = waitcnt_arg::kMaxExpCnt,
          index_t lgkmcnt = waitcnt_arg::kMaxLgkmCnt>
CK_TILE_DEVICE void s_waitcnt()
{
#if defined(__gfx12__)
    // GFX12 do't use __builtin_amdgcn_s_waitcnt
    constexpr index_t wait_mask = waitcnt_arg::from_vmcnt<vmcnt>() |
                                  waitcnt_arg::from_expcnt<expcnt>() |
                                  waitcnt_arg::from_lgkmcnt<lgkmcnt>();

    asm volatile("s_wait_loadcnt_dscnt %0" : : "n"(wait_mask) : "memory");
#else
    __builtin_amdgcn_s_waitcnt(waitcnt_arg::from_vmcnt<vmcnt>() |
                               waitcnt_arg::from_expcnt<expcnt>() |
                               waitcnt_arg::from_lgkmcnt<lgkmcnt>());
#endif
}

template <index_t vmcnt   = waitcnt_arg::kMaxVmCnt,
          index_t expcnt  = waitcnt_arg::kMaxExpCnt,
          index_t lgkmcnt = waitcnt_arg::kMaxLgkmCnt>
CK_TILE_DEVICE void s_waitcnt_barrier()
{
#if defined(__gfx12__)
    // GFX12 optimization: Manual barrier implementation avoids performance penalty
    // from __builtin_amdgcn_s_barrier which inserts extra s_wait_loadcnt_dscnt 0x0
    constexpr index_t wait_mask = waitcnt_arg::from_vmcnt<vmcnt>() |
                                  waitcnt_arg::from_expcnt<expcnt>() |
                                  waitcnt_arg::from_lgkmcnt<lgkmcnt>();

    asm volatile("s_wait_loadcnt_dscnt %0\n"
                 "s_barrier_signal -1\n"
                 "s_barrier_wait -1"
                 :
                 : "n"(wait_mask)
                 : "memory");
#else
    s_waitcnt<vmcnt, expcnt, lgkmcnt>();
    __builtin_amdgcn_s_barrier();
#endif
}

template <index_t lgkmcnt = 0>
CK_TILE_DEVICE void block_sync_lds()
{
    s_waitcnt_barrier<waitcnt_arg::kMaxVmCnt, waitcnt_arg::kMaxExpCnt, lgkmcnt>();
}

template <index_t vmcnt = 0>
CK_TILE_DEVICE void block_sync_lds_direct_load()
{
    s_waitcnt_barrier<vmcnt, waitcnt_arg::kMaxExpCnt, waitcnt_arg::kMaxLgkmCnt>();
}

CK_TILE_DEVICE void s_nop(index_t cnt = 0)
{
#if 1
    asm volatile("s_nop %0" : : "n"(cnt) :);
#else
    __builtin_amdgcn_sched_barrier(cnt);
#endif
}

#define CK_TILE_CONSTANT_ADDRESS_SPACE \
    __attribute__((address_space(      \
        static_cast<safe_underlying_type_t<address_space_enum>>(address_space_enum::constant))))

template <typename T>
__device__ T* cast_pointer_to_generic_address_space(T CK_TILE_CONSTANT_ADDRESS_SPACE* p)
{
    // cast a pointer in "Constant" address space (4) to "Generic" address space (0)
    // only c-style pointer cast seems be able to be compiled
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
    return (T*)(p); // NOLINT(old-style-cast)
#pragma clang diagnostic pop
}

template <typename T>
__host__ __device__ T CK_TILE_CONSTANT_ADDRESS_SPACE* cast_pointer_to_constant_address_space(T* p)
{
    // cast a pointer in "Generic" address space (0) to "Constant" address space (4)
    // only c-style pointer cast seems be able to be compiled;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
    return (T CK_TILE_CONSTANT_ADDRESS_SPACE*)p; // NOLINT(old-style-cast)
#pragma clang diagnostic pop
}

CK_TILE_HOST_DEVICE constexpr index_t get_smem_capacity()
{
#if defined(__gfx950__)
    return 163840;
#else
    return 65536;
#endif
}

/// Helper function to convert address space enum to string
CK_TILE_HOST_DEVICE constexpr const char* address_space_to_string(address_space_enum addr_space)
{
    switch(addr_space)
    {
    case address_space_enum::generic: return "generic";
    case address_space_enum::global: return "global";
    case address_space_enum::lds: return "lds";
    case address_space_enum::sgpr: return "sgpr";
    case address_space_enum::constant: return "constant";
    case address_space_enum::vgpr: return "vgpr";
    default: return "unknown";
    }
}

// Architecture tags
struct gfx9_t
{
};
struct gfx950_t
{
};
struct gfx103_t
{
};
struct gfx11_t
{
};
struct gfx12_t
{
};
struct gfx_invalid_t
{
};

CK_TILE_DEVICE static constexpr auto get_device_arch()
{
// FIXME(0): on all devices except gfx11 it returns gfx12_t
// FIXME(1): during the host compilation pass it returns gfx12_t
#if defined(__gfx103__)
    return gfx103_t{};
#elif defined(__gfx11__)
    return gfx11_t{};
#elif defined(__gfx950__)
    return gfx950_t{};
#elif defined(__gfx9__)
    return gfx9_t{};
#else
    return gfx12_t{};
#endif
}

CK_TILE_DEVICE static constexpr auto get_n_words_per_128b() { return 4; }

namespace detail {
CK_TILE_DEVICE static constexpr auto get_n_lds_banks(gfx9_t) { return 32; }

CK_TILE_DEVICE static constexpr auto get_n_lds_banks(gfx103_t) { return 32; }

CK_TILE_DEVICE static constexpr auto get_n_lds_banks(gfx11_t) { return 32; }

CK_TILE_DEVICE static constexpr auto get_n_lds_banks(gfx12_t) { return 32; }

CK_TILE_DEVICE static constexpr auto get_n_lds_banks(gfx950_t) { return 64; }

CK_TILE_DEVICE static constexpr auto get_n_lds_banks(gfx_invalid_t) { return 0; }

} // namespace detail
CK_TILE_DEVICE static constexpr auto get_n_lds_banks()
{
    return detail::get_n_lds_banks(get_device_arch());
}

enum LLVMSchedGroupMask : int32_t
{
    NONE       = 0,
    ALU        = 1 << 0,
    VALU       = 1 << 1,
    SALU       = 1 << 2,
    MFMA       = 1 << 3,
    VMEM       = 1 << 4,
    VMEM_READ  = 1 << 5,
    VMEM_WRITE = 1 << 6,
    DS         = 1 << 7,
    DS_READ    = 1 << 8,
    DS_WRITE   = 1 << 9,
    ALL        = (DS_WRITE << 1) - 1,
};
} // namespace ck_tile
