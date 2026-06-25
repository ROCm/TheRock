// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "multi_reduce2d_kernel.hpp"
namespace ck_tile {

template <typename Problem_, typename Policy_ = Reduce2dDefaultPolicy>
using MultiReduceThreadWise = MultiReduce2d<Problem_, Policy_, false>;

} // namespace ck_tile
