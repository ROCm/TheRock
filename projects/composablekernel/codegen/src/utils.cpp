// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/host/utils.hpp"

namespace ck {
namespace host {

std::size_t integer_divide_ceil(std::size_t x, std::size_t y)
{
    return (x + y - std::size_t{1}) / y;
}

const std::unordered_set<std::string>& get_xdlop_archs()
{
    static std::unordered_set<std::string> supported_archs{
        "gfx90a", "gfx908", "gfx942", "gfx1100", "gfx1101", "gfx1102", "gfx1200", "gfx1201"};
    return supported_archs;
}

} // namespace host
} // namespace ck
