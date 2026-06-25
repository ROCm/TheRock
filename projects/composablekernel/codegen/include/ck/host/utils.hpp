// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <unordered_set>
#include <numeric>
#include <iterator>

namespace ck {
namespace host {

std::size_t integer_divide_ceil(std::size_t x, std::size_t y);

const std::unordered_set<std::string>& get_xdlop_archs();
} // namespace host
} // namespace ck
