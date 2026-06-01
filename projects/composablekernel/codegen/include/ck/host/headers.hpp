// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <utility>
#include <unordered_map>
#include <vector>

namespace ck {
namespace host {

std::unordered_map<std::string_view, std::string_view> GetHeaders();

} // namespace host
} // namespace ck
