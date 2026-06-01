// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "add_rmsnorm2d_rdquant_fwd_instance_common.hpp"

// clang-format off
//                                                               rm  rn  tm  tn  vn    pd     x      3p
template float add_rmsnorm2d_rdquant_fwd_<trait_<ck_tile::fp16_t, ck_tile::int8_t, 1,  1,  4,  64, 1,  true , true, false>>(const S&, A);
template float add_rmsnorm2d_rdquant_fwd_<trait_<ck_tile::fp16_t, ck_tile::int8_t, 1,  1,  4,  64, 2,  true , true, false>>(const S&, A);
template float add_rmsnorm2d_rdquant_fwd_<trait_<ck_tile::fp16_t, ck_tile::int8_t, 1,  2,  4,  64, 1,  true , true, false>>(const S&, A);
template float add_rmsnorm2d_rdquant_fwd_<trait_<ck_tile::fp16_t, ck_tile::fp8_t, 1,  1,  4,  64, 1,  true , true, false>>(const S&, A);
template float add_rmsnorm2d_rdquant_fwd_<trait_<ck_tile::fp16_t, ck_tile::fp8_t, 1,  1,  4,  64, 2,  true , true, false>>(const S&, A);
template float add_rmsnorm2d_rdquant_fwd_<trait_<ck_tile::fp16_t, ck_tile::fp8_t, 1,  2,  4,  64, 1,  true , true, false>>(const S&, A);
// clang-format on
