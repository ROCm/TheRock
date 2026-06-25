// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#ifndef __HIPCC_RTC__
#include <string>
#include <string_view>
#include <hip/hip_runtime.h>

namespace ck {

constexpr unsigned int fnv1a_hash(std::string_view str, unsigned int h = 2166136261u)
{
    return str.empty() ? h
                       : fnv1a_hash(str.substr(1),
                                    (h ^ static_cast<unsigned char>(str.front())) * 16777619u);
}
inline std::string get_device_name()
{
    hipDeviceProp_t props{};
    int device;
    auto status = hipGetDevice(&device);
    if(status != hipSuccess)
    {
        return std::string();
    }
    status = hipGetDeviceProperties(&props, device);
    if(status != hipSuccess)
    {
        return std::string();
    }
    const std::string raw_name(props.gcnArchName);
    const auto name = raw_name.substr(0, raw_name.find(':')); // str.substr(0, npos) returns str.
    switch(fnv1a_hash(name))
    {
    // https://github.com/ROCm/MIOpen/blob/8498875aef84878e04c1eabefdf6571514891086/src/target_properties.cpp#L40
    case fnv1a_hash("Ellesmere"):
    case fnv1a_hash("Baffin"):
    case fnv1a_hash("RacerX"):
    case fnv1a_hash("Polaris10"):
    case fnv1a_hash("Polaris11"):
    case fnv1a_hash("Tonga"):
    case fnv1a_hash("Fiji"):
    case fnv1a_hash("gfx800"):
    case fnv1a_hash("gfx802"):
    case fnv1a_hash("gfx804"): return "gfx803";
    case fnv1a_hash("Vega10"):
    case fnv1a_hash("gfx901"): return "gfx900";
    case fnv1a_hash("10.3.0 Sienna_Cichlid 18"): return "gfx1030";
    default: return name;
    }
}

inline bool is_gfx12_supported()
{
    return ck::get_device_name() == "gfx1200" || ck::get_device_name() == "gfx1201";
}

inline bool is_gfx11_supported()
{
    return ck::get_device_name() == "gfx1100" || ck::get_device_name() == "gfx1101" ||
           ck::get_device_name() == "gfx1102" || ck::get_device_name() == "gfx1103" ||
           ck::get_device_name() == "gfx1150" || ck::get_device_name() == "gfx1151" ||
           ck::get_device_name() == "gfx1152" || ck::get_device_name() == "gfx1153";
}

inline bool is_xdl_supported()
{
    return ck::get_device_name() == "gfx908" || ck::get_device_name() == "gfx90a" ||
           ck::get_device_name() == "gfx942" || ck::get_device_name() == "gfx950" ||
           is_gfx12_supported() || is_gfx11_supported();
}

template <typename ADataType,
          typename BDataType,
          index_t MPerXDL64,
          index_t NPerXDL64,
          index_t MPerXDL32 = MPerXDL64,
          index_t NPerXDL32 = NPerXDL64>
inline bool is_xdl_wmma_supported()
{
    if(ck::get_device_name() == "gfx908" || ck::get_device_name() == "gfx90a" ||
       ck::get_device_name() == "gfx942" || ck::get_device_name() == "gfx950")
    {
        return true;
    }
    else if(is_gfx12_supported() || is_gfx11_supported())
    {
        if constexpr((MPerXDL32 != 16) || (NPerXDL32 != 16))
        {
            return false;
        }
        if constexpr(sizeof(ADataType) > 2 || sizeof(BDataType) > 2)
        {
            return false;
        }
        return true;
    }
    else
    {
        return false;
    }
}

inline bool is_lds_direct_load_supported()
{
    // Check if direct loads from global memory to LDS are supported.
    return ck::get_device_name() == "gfx90a" || ck::get_device_name() == "gfx942" ||
           ck::get_device_name() == "gfx950";
}

inline bool is_bf16_atomic_supported()
{
    return ck::get_device_name() == "gfx942" || ck::get_device_name() == "gfx950" ||
           is_gfx12_supported();
}

inline bool is_gfx101_supported()
{
    return ck::get_device_name() == "gfx1010" || ck::get_device_name() == "gfx1011" ||
           ck::get_device_name() == "gfx1012";
}

inline bool is_gfx103_supported()
{
    return ck::get_device_name() == "gfx1030" || ck::get_device_name() == "gfx1031" ||
           ck::get_device_name() == "gfx1032" || ck::get_device_name() == "gfx1034" ||
           ck::get_device_name() == "gfx1035" || ck::get_device_name() == "gfx1036";
}

inline bool is_wmma_supported()
{
    return is_gfx103_supported() || is_gfx11_supported() || is_gfx12_supported();
}

inline bool is_tf32_supported()
{
    return ck::get_device_name() == "gfx942" || ck::get_device_name() == "gfx950";
}

} // namespace ck
#endif
