// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "add_rmsnorm2d_rdquant_fwd_instance_common.hpp"

// clang-format off
//                                                               rm  rn  tm  tn  vn     pd    x     3p
template float add_rmsnorm2d_rdquant_fwd_<trait_<ck_tile::bf16_t,  ck_tile::int8_t, 1, 3, 1,  128, 8,  true,  true, false>>(const S&, A);
template float add_rmsnorm2d_rdquant_fwd_<trait_<ck_tile::bf16_t,  ck_tile::int8_t, 1, 3, 1,  256, 4,  true,  true, false>>(const S&, A);
template float add_rmsnorm2d_rdquant_fwd_<trait_<ck_tile::bf16_t,  ck_tile::int8_t, 1, 6, 1,  256, 2,  true,  true, false>>(const S&, A);
template float add_rmsnorm2d_rdquant_fwd_<trait_<ck_tile::bf16_t,  ck_tile::int8_t, 1, 3, 1, 1024, 1,  true,  true, false>>(const S&, A);
template float add_rmsnorm2d_rdquant_fwd_<trait_<ck_tile::bf16_t,  ck_tile::fp8_t, 1, 3, 1,  128, 8,  true,  true, false>>(const S&, A);
template float add_rmsnorm2d_rdquant_fwd_<trait_<ck_tile::bf16_t,  ck_tile::fp8_t, 1, 3, 1,  256, 4,  true,  true, false>>(const S&, A);
template float add_rmsnorm2d_rdquant_fwd_<trait_<ck_tile::bf16_t,  ck_tile::fp8_t, 1, 6, 1,  256, 2,  true,  true, false>>(const S&, A);
template float add_rmsnorm2d_rdquant_fwd_<trait_<ck_tile::bf16_t,  ck_tile::fp8_t, 1, 3, 1, 1024, 1,  true,  true, false>>(const S&, A);
// clang-format on
