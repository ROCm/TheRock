// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <tuple>
#include <type_traits>
#include "gtest/gtest.h"
#include "ck_tile/host.hpp"

using F8   = ck_tile::fp8_t;
using F16  = ck_tile::half_t;
using BF16 = ck_tile::bf16_t;
using BF8  = ck_tile::bf8_t;
using F32  = float;

using Row = ck_tile::tensor_layout::gemm::RowMajor;
using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

using Persistent    = std::true_type;
using NonPersistent = std::false_type;

using I32  = ck_tile::number<32>;
using I128 = ck_tile::number<128>;
using I256 = ck_tile::number<256>;

// clang-format off
using KernelTypesStreamKFp16Persistent = ::testing::Types<
//                ALayout  BLayout  CLayout   ADataType  BDataType  AccDataType  CDataType  M_MacroTile  N_MacroTile  K_MacroTile  Persistent

    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,     Persistent>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,     Persistent>,
    std::tuple<    Col,     Col,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,     Persistent>,
    std::tuple<    Col,     Row,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,     Persistent>
>;

using KernelTypesStreamKFp16Reduction = ::testing::Types<
//                ALayout  BLayout  CLayout   ADataType  BDataType  AccDataType  CDataType  M_MacroTile  N_MacroTile  K_MacroTile  Persistent
    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,     Persistent>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,     Persistent>,
    std::tuple<    Col,     Col,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,     Persistent>,
    std::tuple<    Col,     Row,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,     Persistent>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,     NonPersistent>>;

using KernelTypesStreamKBf16Persistent = ::testing::Types<
    std::tuple<    Row,     Row,     Row,       BF16,      BF16,        F32,       BF16,         I256,        I256,        I32,    Persistent>,
    std::tuple<    Row,     Col,     Row,       BF16,      BF16,        F32,       BF16,         I256,        I256,        I32,    Persistent>,
    std::tuple<    Col,     Col,     Row,       BF16,      BF16,        F32,       BF16,         I256,        I256,        I32,    Persistent>,
    std::tuple<    Col,     Row,     Row,       BF16,      BF16,        F32,       BF16,         I256,        I256,        I32,    Persistent>
>;

using KernelTypesStreamKBf8Persistent = ::testing::Types<
    std::tuple<    Row,     Row,     Row,       BF8,      BF8,        F32,       BF16,         I128,        I128,        I32,    Persistent>,
    std::tuple<    Row,     Col,     Row,       BF8,      BF8,        F32,       BF16,         I128,        I128,        I32,    Persistent>,
    std::tuple<    Col,     Col,     Row,       BF8,      BF8,        F32,       BF16,         I128,        I128,        I32,    Persistent>,
    std::tuple<    Col,     Row,     Row,       BF8,      BF8,        F32,       BF16,         I128,        I128,        I32,    Persistent>
>;

using KernelTypesStreamKFp8Persistent = ::testing::Types<
    std::tuple<    Row,     Row,     Row,       F8,        F8,        F32,        F16,         I128,        I128,        I32,    Persistent>,
    std::tuple<    Row,     Col,     Row,       F8,        F8,        F32,        F16,         I128,        I128,        I32,    Persistent>,
    std::tuple<    Col,     Col,     Row,       F8,        F8,        F32,        F16,         I128,        I128,        I32,    Persistent>,
    std::tuple<    Col,     Row,     Row,       F8,        F8,        F32,        F16,         I128,        I128,        I32,    Persistent>
>;

using KernelTypesStreamKFp16NonPersistent = ::testing::Types<
//                ALayout  BLayout  CLayout   ADataType  BDataType  AccDataType  CDataType  M_MacroTile  N_MacroTile  K_MacroTile  Persistent

    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,     NonPersistent>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,     NonPersistent>,
    std::tuple<    Col,     Col,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,     NonPersistent>,
    std::tuple<    Col,     Row,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,     NonPersistent>
>;

using KernelTypesStreamKBf16NonPersistent = ::testing::Types<
    std::tuple<    Row,     Row,     Row,       BF16,      BF16,        F32,       BF16,         I256,        I256,        I32,    NonPersistent>,
    std::tuple<    Row,     Col,     Row,       BF16,      BF16,        F32,       BF16,         I256,        I256,        I32,    NonPersistent>,
    std::tuple<    Col,     Col,     Row,       BF16,      BF16,        F32,       BF16,         I256,        I256,        I32,    NonPersistent>,
    std::tuple<    Col,     Row,     Row,       BF16,      BF16,        F32,       BF16,         I256,        I256,        I32,    NonPersistent>
>;

using KernelTypesStreamKBf8NonPersistent = ::testing::Types<
    std::tuple<    Row,     Row,     Row,        BF8,      BF8,        F32,       BF16,         I128,        I128,        I32,    NonPersistent>,
    std::tuple<    Row,     Col,     Row,        BF8,      BF8,        F32,       BF16,         I128,        I128,        I32,    NonPersistent>,
    std::tuple<    Col,     Col,     Row,        BF8,      BF8,        F32,       BF16,         I128,        I128,        I32,    NonPersistent>,
    std::tuple<    Col,     Row,     Row,        BF8,      BF8,        F32,       BF16,         I128,        I128,        I32,    NonPersistent>
>;

using KernelTypesStreamKFp8NonPersistent = ::testing::Types<
    std::tuple<    Row,     Row,     Row,       F8,         F8,        F32,        F16,         I128,        I128,        I32,    NonPersistent>,
    std::tuple<    Row,     Col,     Row,       F8,         F8,        F32,        F16,         I128,        I128,        I32,    NonPersistent>,
    std::tuple<    Col,     Col,     Row,       F8,         F8,        F32,        F16,         I128,        I128,        I32,    NonPersistent>,
    std::tuple<    Col,     Row,     Row,       F8,         F8,        F32,        F16,         I128,        I128,        I32,    NonPersistent>
>;

// clang-format on
