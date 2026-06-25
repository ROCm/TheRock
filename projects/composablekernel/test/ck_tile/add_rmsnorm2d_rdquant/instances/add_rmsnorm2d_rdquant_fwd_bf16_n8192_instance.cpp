// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "add_rmsnorm2d_rdquant_fwd_instance_common.hpp"

// clang-format off
//                                                               rm  rn  tm  tn  vn     pd    x     3p
template float add_rmsnorm2d_rdquant_fwd_<trait_<ck_tile::bf16_t,  ck_tile::int8_t, 1, 2, 1,  512, 8,  true,  true, false>>(const S&, A);
template float add_rmsnorm2d_rdquant_fwd_<trait_<ck_tile::bf16_t,  ck_tile::int8_t, 1, 4, 1,  512, 4,  true,  true, false>>(const S&, A);
template float add_rmsnorm2d_rdquant_fwd_<trait_<ck_tile::bf16_t,  ck_tile::int8_t, 1, 4, 1, 1024, 2,  true,  true, false>>(const S&, A);
template float add_rmsnorm2d_rdquant_fwd_<trait_<ck_tile::bf16_t,  ck_tile::int8_t, 1, 8, 1, 1024, 1,  true,  true, false>>(const S&, A);
template float add_rmsnorm2d_rdquant_fwd_<trait_<ck_tile::bf16_t,  ck_tile::int8_t, 1, 2, 1,  512, 8,  true,  false, false>>(const S&, A);
template float add_rmsnorm2d_rdquant_fwd_<trait_<ck_tile::bf16_t,  ck_tile::int8_t, 1, 4, 1,  512, 4,  true,  false, false>>(const S&, A);
template float add_rmsnorm2d_rdquant_fwd_<trait_<ck_tile::bf16_t,  ck_tile::int8_t, 1, 4, 1, 1024, 2,  true,  false, false>>(const S&, A);
template float add_rmsnorm2d_rdquant_fwd_<trait_<ck_tile::bf16_t,  ck_tile::int8_t, 1, 8, 1, 1024, 1,  true,  false, false>>(const S&, A);
template float add_rmsnorm2d_rdquant_fwd_<trait_<ck_tile::bf16_t,  ck_tile::int8_t, 1, 2, 1,  512, 8,  false,  true, false>>(const S&, A);
template float add_rmsnorm2d_rdquant_fwd_<trait_<ck_tile::bf16_t,  ck_tile::int8_t, 1, 4, 1,  512, 4,  false,  true, false>>(const S&, A);
template float add_rmsnorm2d_rdquant_fwd_<trait_<ck_tile::bf16_t,  ck_tile::int8_t, 1, 4, 1, 1024, 2,  false,  true, false>>(const S&, A);
template float add_rmsnorm2d_rdquant_fwd_<trait_<ck_tile::bf16_t,  ck_tile::int8_t, 1, 8, 1, 1024, 1,  false,  true, false>>(const S&, A);
template float add_rmsnorm2d_rdquant_fwd_<trait_<ck_tile::bf16_t,  ck_tile::int8_t, 1, 2, 1,  512, 8,  false,  false, false>>(const S&, A);
template float add_rmsnorm2d_rdquant_fwd_<trait_<ck_tile::bf16_t,  ck_tile::int8_t, 1, 4, 1,  512, 4,  false,  false, false>>(const S&, A);
template float add_rmsnorm2d_rdquant_fwd_<trait_<ck_tile::bf16_t,  ck_tile::int8_t, 1, 4, 1, 1024, 2,  false,  false, false>>(const S&, A);
template float add_rmsnorm2d_rdquant_fwd_<trait_<ck_tile::bf16_t,  ck_tile::int8_t, 1, 8, 1, 1024, 1,  false,  false, false>>(const S&, A);
template float add_rmsnorm2d_rdquant_fwd_<trait_<ck_tile::bf16_t,  ck_tile::fp8_t, 1, 2, 1,  512, 8,  true,  true, false>>(const S&, A);
template float add_rmsnorm2d_rdquant_fwd_<trait_<ck_tile::bf16_t,  ck_tile::fp8_t, 1, 4, 1,  512, 4,  true,  true, false>>(const S&, A);
template float add_rmsnorm2d_rdquant_fwd_<trait_<ck_tile::bf16_t,  ck_tile::fp8_t, 1, 4, 1, 1024, 2,  true,  true, false>>(const S&, A);
template float add_rmsnorm2d_rdquant_fwd_<trait_<ck_tile::bf16_t,  ck_tile::fp8_t, 1, 8, 1, 1024, 1,  true,  true, false>>(const S&, A);
template float add_rmsnorm2d_rdquant_fwd_<trait_<ck_tile::bf16_t,  ck_tile::fp8_t, 1, 2, 1,  512, 8,  true,  false, false>>(const S&, A);
template float add_rmsnorm2d_rdquant_fwd_<trait_<ck_tile::bf16_t,  ck_tile::fp8_t, 1, 4, 1,  512, 4,  true,  false, false>>(const S&, A);
template float add_rmsnorm2d_rdquant_fwd_<trait_<ck_tile::bf16_t,  ck_tile::fp8_t, 1, 4, 1, 1024, 2,  true,  false, false>>(const S&, A);
template float add_rmsnorm2d_rdquant_fwd_<trait_<ck_tile::bf16_t,  ck_tile::fp8_t, 1, 8, 1, 1024, 1,  true,  false, false>>(const S&, A);
template float add_rmsnorm2d_rdquant_fwd_<trait_<ck_tile::bf16_t,  ck_tile::fp8_t, 1, 2, 1,  512, 8,  false,  true, false>>(const S&, A);
template float add_rmsnorm2d_rdquant_fwd_<trait_<ck_tile::bf16_t,  ck_tile::fp8_t, 1, 4, 1,  512, 4,  false,  true, false>>(const S&, A);
template float add_rmsnorm2d_rdquant_fwd_<trait_<ck_tile::bf16_t,  ck_tile::fp8_t, 1, 4, 1, 1024, 2,  false,  true, false>>(const S&, A);
template float add_rmsnorm2d_rdquant_fwd_<trait_<ck_tile::bf16_t,  ck_tile::fp8_t, 1, 8, 1, 1024, 1,  false,  true, false>>(const S&, A);
template float add_rmsnorm2d_rdquant_fwd_<trait_<ck_tile::bf16_t,  ck_tile::fp8_t, 1, 2, 1,  512, 8,  false,  false, false>>(const S&, A);
template float add_rmsnorm2d_rdquant_fwd_<trait_<ck_tile::bf16_t,  ck_tile::fp8_t, 1, 4, 1,  512, 4,  false,  false, false>>(const S&, A);
template float add_rmsnorm2d_rdquant_fwd_<trait_<ck_tile::bf16_t,  ck_tile::fp8_t, 1, 4, 1, 1024, 2,  false,  false, false>>(const S&, A);
template float add_rmsnorm2d_rdquant_fwd_<trait_<ck_tile::bf16_t,  ck_tile::fp8_t, 1, 8, 1, 1024, 1,  false,  false, false>>(const S&, A);

// clang-format on
