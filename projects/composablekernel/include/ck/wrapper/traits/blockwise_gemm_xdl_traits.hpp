// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/ck.hpp"

// Disable from doxygen docs generation
/// @cond INTERNAL
namespace ck {
namespace wrapper {
/// @endcond

/**
 * \brief Traits for blockwise gemm xdl.
 *
 * \tparam MPerXDLValue The MFMA instruction size in M dimension.
 * \tparam NPerXDLValue The MFMA instruction size in N dimension.
 * \tparam MXdlPerWaveValue  The number of MFMA instructions run by single
 * wave in M dimension.
 * \tparam NXdlPerWaveValue  The number of MFMA instructions run by single
 * wave in N dimension.
 * \tparam K1Value The number of K-dim elements that are packed together as
 * a separate logical dimension. Usually aligns with vector load size.
 */
template <typename MPerXDLValue,
          typename NPerXDLValue,
          typename MXdlPerWaveValue,
          typename NXdlPerWaveValue,
          typename K1Value>
struct BlockwisGemmXdlTraits
{
    static constexpr auto MPerXDL     = MPerXDLValue{};
    static constexpr auto NPerXDL     = NPerXDLValue{};
    static constexpr auto MXdlPerWave = MXdlPerWaveValue{};
    static constexpr auto NXdlPerWave = NXdlPerWaveValue{};
    static constexpr auto K1          = K1Value{};
};

// K1 = 4
struct BlockwisGemmXdlTraits_32x32Xdl_4x2XdlPerWave_4K1
    : BlockwisGemmXdlTraits<Number<32>, Number<32>, Number<4>, Number<2>, Number<4>>
{
};
struct BlockwisGemmXdlTraits_32x32Xdl_2x4XdlPerWave_4K1
    : BlockwisGemmXdlTraits<Number<32>, Number<32>, Number<2>, Number<4>, Number<4>>
{
};
struct BlockwisGemmXdlTraits_32x32Xdl_2x2XdlPerWave_4K1
    : BlockwisGemmXdlTraits<Number<32>, Number<32>, Number<2>, Number<2>, Number<4>>
{
};
// K1 = 8
struct BlockwisGemmXdlTraits_32x32Xdl_4x2XdlPerWave_8K1
    : BlockwisGemmXdlTraits<Number<32>, Number<32>, Number<4>, Number<2>, Number<8>>
{
};
struct BlockwisGemmXdlTraits_32x32Xdl_2x4XdlPerWave_8K1
    : BlockwisGemmXdlTraits<Number<32>, Number<32>, Number<2>, Number<4>, Number<8>>
{
};
struct BlockwisGemmXdlTraits_32x32Xdl_2x2XdlPerWave_8K1
    : BlockwisGemmXdlTraits<Number<32>, Number<32>, Number<2>, Number<2>, Number<8>>
{
};
// K1 = 16
struct BlockwisGemmXdlTraits_32x32Xdl_4x2XdlPerWave_16K1
    : BlockwisGemmXdlTraits<Number<32>, Number<32>, Number<4>, Number<2>, Number<16>>
{
};
struct BlockwisGemmXdlTraits_32x32Xdl_2x4XdlPerWave_16K1
    : BlockwisGemmXdlTraits<Number<32>, Number<32>, Number<2>, Number<4>, Number<16>>
{
};
struct BlockwisGemmXdlTraits_32x32Xdl_2x2XdlPerWave_16K1
    : BlockwisGemmXdlTraits<Number<32>, Number<32>, Number<2>, Number<2>, Number<16>>
{
};

} // namespace wrapper
} // namespace ck
