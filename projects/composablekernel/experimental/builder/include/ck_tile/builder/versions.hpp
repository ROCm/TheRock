// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <concepts>
#include <string_view>

#include "ck_tile/builder/builder_utils.hpp"

namespace ck_tile::builder {

static constexpr StringLiteral V0_0_0 = "0.0.0";
static constexpr StringLiteral V0_1_0 = "0.1.0";

static constexpr StringLiteral LATEST_API_VERSION = V0_1_0;

template <StringLiteral V>
concept SupportedVersion = (V == V0_0_0) || (V == V0_1_0);

} // namespace ck_tile::builder
