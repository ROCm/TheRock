// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

namespace ck_tile {

enum struct ConvolutionSpecialization
{
    Default,
    Filter1x1Pad0,
    Filter1x1Stride1Pad0,
    Filter3x3,
};

CK_TILE_HOST std::string getConvSpecializationString(const ConvolutionSpecialization& s)
{
    switch(s)
    {
    case ConvolutionSpecialization::Default: return "Default";
    case ConvolutionSpecialization::Filter1x1Pad0: return "Filter1x1Pad0";
    case ConvolutionSpecialization::Filter1x1Stride1Pad0: return "Filter1x1Stride1Pad0";
    case ConvolutionSpecialization::Filter3x3: return "Filter3x3";
    default: return "Unrecognized specialization!";
    }
}

} // namespace ck_tile
