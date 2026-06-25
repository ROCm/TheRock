// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "add_rmsnorm2d_rdquant_fwd_instance_common.hpp"

// clang-format off
//                                                                rm  rn  tm  tn  vn  pd      x     3p
#if 0
template float add_rmsnorm2d_rdquant_fwd_<trait_<ck_tile::bf16_t, 1,  2,  4,  64, 8,  true , true, false>>(const S&, A);
template float add_rmsnorm2d_rdquant_fwd_<trait_<ck_tile::bf16_t, 1,  4,  4,  64, 4,  true , true, false>>(const S&, A);
template float add_rmsnorm2d_rdquant_fwd_<trait_<ck_tile::bf16_t, 1,  8,  4,  64, 2,  true , true, false>>(const S&, A);
template float add_rmsnorm2d_rdquant_fwd_<trait_<ck_tile::bf16_t, 1, 16,  4,  64, 1,  true , true, false>>(const S&, A);

template float add_rmsnorm2d_rdquant_fwd_<trait_<ck_tile::bf16_t, 1,  1,  1, 256, 4,  true , true, false>>(const S&, A);
#endif

template float add_rmsnorm2d_rdquant_fwd_<trait_<ck_tile::bf16_t,  ck_tile::int8_t, 1, 1, 2,  128, 8,  true,  true, false>>(const S&, A);
template float add_rmsnorm2d_rdquant_fwd_<trait_<ck_tile::bf16_t,  ck_tile::int8_t, 1, 2, 2,  128, 4,  true,  true, false>>(const S&, A);
template float add_rmsnorm2d_rdquant_fwd_<trait_<ck_tile::bf16_t,  ck_tile::int8_t, 1, 4, 2,  128, 2,  true,  true, false>>(const S&, A);
template float add_rmsnorm2d_rdquant_fwd_<trait_<ck_tile::bf16_t,  ck_tile::int8_t, 1, 4, 1,  256, 1,  true,  true, false>>(const S&, A);
template float add_rmsnorm2d_rdquant_fwd_<trait_<ck_tile::bf16_t,  ck_tile::fp8_t, 1, 1, 2,  128, 8,  true,  true, false>>(const S&, A);
template float add_rmsnorm2d_rdquant_fwd_<trait_<ck_tile::bf16_t,  ck_tile::fp8_t, 1, 2, 2,  128, 4,  true,  true, false>>(const S&, A);
template float add_rmsnorm2d_rdquant_fwd_<trait_<ck_tile::bf16_t,  ck_tile::fp8_t, 1, 4, 2,  128, 2,  true,  true, false>>(const S&, A);
template float add_rmsnorm2d_rdquant_fwd_<trait_<ck_tile::bf16_t,  ck_tile::fp8_t, 1, 4, 1,  256, 1,  true,  true, false>>(const S&, A);
// clang-format on
