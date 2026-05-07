// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "smoothquant_instance_common.hpp"

// clang-format off
//                                                  rm rn  tm  tn  vn   pd    2p
template float smoothquant_<trait_<ck_tile::fp16_t, 1,  1,  4, 64, 8,  true , false>>(const S&, A);
template float smoothquant_<trait_<ck_tile::fp16_t, 1,  2,  4, 64, 4,  true , false>>(const S&, A);
template float smoothquant_<trait_<ck_tile::fp16_t, 1,  4,  4, 64, 2,  true , false>>(const S&, A);
template float smoothquant_<trait_<ck_tile::fp16_t, 1,  8,  4, 64, 1,  true , false>>(const S&, A);
// clang-format on
