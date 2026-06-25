// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hip/hip_runtime.h>
#include <string>

namespace ck {
namespace test {

struct DeviceResources
{
    int computeUnits;
    size_t totalMemory;
    std::string deviceName;
    // Add other relevant properties as needed
};

inline DeviceResources GetDeviceResources()
{
    DeviceResources res;
    hipDeviceProp_t props;

    hipError_t status = hipGetDeviceProperties(&props, 0);
    if(status != hipSuccess)
    {
        props.multiProcessorCount = 0;
        res.computeUnits          = 0;
        res.totalMemory           = 0;
        res.deviceName            = "Unknown";
        return res;
    }

    res.computeUnits = props.multiProcessorCount;
    res.totalMemory  = props.totalGlobalMem;
    res.deviceName   = props.name;

    return res;
}

// Device capability tiers
enum class DeviceCapabilityTier
{
    LOW,    // Low resources devices (CU less than 80)
    MEDIUM, // Mid-range devices
    HIGH    // High resources devices (CU hiher than 100)
};

inline DeviceCapabilityTier DetermineDeviceTier()
{
    DeviceResources res = GetDeviceResources();

    // Adjust these thresholds based on your device specifics
    if(res.computeUnits < 80)
    {
        return DeviceCapabilityTier::LOW;
    }
    else if(res.computeUnits < 100)
    {
        return DeviceCapabilityTier::MEDIUM;
    }
    else
    {
        return DeviceCapabilityTier::HIGH;
    }
}

} // namespace test
} // namespace ck
