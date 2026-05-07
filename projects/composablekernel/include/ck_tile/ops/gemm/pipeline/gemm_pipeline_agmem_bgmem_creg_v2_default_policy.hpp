// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

// Default policy for GemmPipelineAGmemBGmemCRegV2
// Default policy class should not be templated, put template on member functions instead
// NOTE: policy should be binded to its corresponding operation. It's just a coincidence that
//   GemmPipelineAGmemBGmemCRegV2DefaultPolicy is the same as
//   GemmPipelineAGmemBGmemCRegV1DefaultPolicy
using GemmPipelineAGmemBGmemCRegV2DefaultPolicy = GemmPipelineAGmemBGmemCRegV1DefaultPolicy;

} // namespace ck_tile
