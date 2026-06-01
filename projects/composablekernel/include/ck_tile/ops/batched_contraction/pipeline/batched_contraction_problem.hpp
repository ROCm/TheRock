// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once
#include "ck_tile/core.hpp"

namespace ck_tile {

template <typename ADataType_,
          typename BDataType_,
          typename DsDataType_,
          typename EDataType_,
          ck_tile::index_t NumDimG_,
          ck_tile::index_t NumDimM_,
          ck_tile::index_t NumDimN_,
          ck_tile::index_t NumDimK_,
          ck_tile::index_t NumDTensor_>
struct BatchedContractionProblem
{
    using ADataType  = ck_tile::remove_cvref_t<ADataType_>;
    using BDataType  = ck_tile::remove_cvref_t<BDataType_>;
    using DsDataType = ck_tile::remove_cvref_t<DsDataType_>;
    using EDataType  = ck_tile::remove_cvref_t<EDataType_>;

    static constexpr ck_tile::index_t NumDimG    = NumDimG_;
    static constexpr ck_tile::index_t NumDimM    = NumDimM_;
    static constexpr ck_tile::index_t NumDimN    = NumDimN_;
    static constexpr ck_tile::index_t NumDimK    = NumDimK_;
    static constexpr ck_tile::index_t NumDTensor = NumDTensor_;
};

} // namespace ck_tile
