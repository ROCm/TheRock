// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#ifndef __HIPCC_RTC__
#include <string>
#include <string_view>
#include <hip/hip_runtime.h>

namespace ck_tile {

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

inline bool is_gfx11_supported()
{
    return get_device_name() == "gfx1100" || get_device_name() == "gfx1101" ||
           get_device_name() == "gfx1102" || get_device_name() == "gfx1103" ||
           get_device_name() == "gfx1150" || get_device_name() == "gfx1151" ||
           get_device_name() == "gfx1152" || get_device_name() == "gfx1153";
}

inline bool is_gfx12_supported()
{
    return get_device_name() == "gfx1200" || get_device_name() == "gfx1201";
}

inline bool is_load_tr_supported()
{
    // Check if load transpose is supported.
    return get_device_name() == "gfx950";
}

inline size_t get_num_cus()
{
    hipDeviceProp_t props{};
    int device;
    auto status = hipGetDevice(&device);
    if(status != hipSuccess)
    {
        return 0;
    }
    status = hipGetDeviceProperties(&props, device);
    if(status != hipSuccess)
    {
        return 0;
    }
    return static_cast<size_t>(props.multiProcessorCount);
}

} // namespace ck_tile

#endif
