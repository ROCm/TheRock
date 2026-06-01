// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

template <typename DataType_>
struct BlockSoftmax2DProblem
{
    using DataType = remove_cvref_t<DataType_>;
};

} // namespace ck_tile
