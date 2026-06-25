// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/ck.hpp"

#include "ck/utility/data_type.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/warp/xdlops_gemm.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_mx_gemm.hpp"
#include "ck/library/utility/check_err.hpp"

namespace ck {

// MFMA instructions supported in this test
enum class MFMA_F8F6F4
{
    F32_16x16x128 =
        static_cast<int>(MfmaInstr::mfma_f32_16x16x128f8f6f4), // V_MFMA_F32_16X16X128_F8F6F4
    F32_32x32x64 =
        static_cast<int>(MfmaInstr::mfma_f32_32x32x64f8f6f4), // V_MFMA_F32_32X32X64_F8F6F4

    SCALE_F32_16x16x128 = static_cast<int>(
        MfmaInstr::mfma_scale_f32_16x16x128f8f6f4), // V_MFMA_SCALE_F32_16X16X128_F8F6F4
    SCALE_F32_32x32x64 = static_cast<int>(
        MfmaInstr::mfma_scale_f32_32x32x64f8f6f4) // V_MFMA_SCALE_F32_32X32X64_F8F6F4

};

template <int32_t BLOCK_M, int32_t BLOCK_N>
struct mfma_type_selector;

template <>
struct mfma_type_selector<16, 16>
{
    template <typename AFragT, typename BFragT, typename AccumFragT>
    __device__ static void run(AFragT const& fragA, BFragT const& fragB, AccumFragT& fragAcc)
    {
        auto op = mfma_type<MfmaInstr::mfma_f32_16x16x128f8f6f4>{};
        op.template run<16, 16>(fragA, fragB, fragAcc);
    }
};

template <>
struct mfma_type_selector<32, 32>
{
    template <typename AFragT, typename BFragT, typename AccumFragT>
    __device__ static void run(AFragT const& fragA, BFragT const& fragB, AccumFragT& fragAcc)
    {
        auto op = mfma_type<MfmaInstr::mfma_f32_32x32x64f8f6f4>{};
        op.template run<32, 32>(fragA, fragB, fragAcc);
    }
};

template <int32_t BLOCK_M, int32_t BLOCK_N>
struct mfma_scale_type_selector;

template <>
struct mfma_scale_type_selector<16, 16>
{
    template <typename AFragT,
              typename AScaleFragT,
              typename BFragT,
              typename BScaleFragT,
              typename AccumFragT>
    __device__ static void run(AFragT const& fragA,
                               AScaleFragT const& scale_a,
                               BFragT const& fragB,
                               BScaleFragT const& scale_b,
                               AccumFragT& fragAcc)
    {
        auto op = mfma_type<MfmaInstr::mfma_scale_f32_16x16x128f8f6f4>{};
        op.template run<16, 16, 0, 0>(fragA,
                                      ck::utils::get_exponent_value(scale_a[Number<0>{}]),
                                      fragB,
                                      ck::utils::get_exponent_value(scale_b[Number<0>{}]),
                                      fragAcc);
    }
};

template <>
struct mfma_scale_type_selector<32, 32>
{
    template <typename AFragT,
              typename AScaleFragT,
              typename BFragT,
              typename BScaleFragT,
              typename AccumFragT>
    __device__ static void run(AFragT const& fragA,
                               AScaleFragT const& scale_a,
                               BFragT const& fragB,
                               BScaleFragT const& scale_b,
                               AccumFragT& fragAcc)
    {
        auto op = mfma_type<MfmaInstr::mfma_scale_f32_32x32x64f8f6f4>{};
        op.template run<32, 32, 0, 0>(fragA,
                                      ck::utils::get_exponent_value(scale_a[Number<0>{}]),
                                      fragB,
                                      ck::utils::get_exponent_value(scale_b[Number<0>{}]),
                                      fragAcc);
    }
};

template <typename VecT>
static constexpr int32_t vectorSize(const VecT&)
{
    return scalar_type<VecT>::vector_size;
}

// Define a load function for input A blocks:
// Size: (BLOCK_M x BLOCK_K)
// - Data is in column major format
// - Rows are loaded in contiguous chunks that map to corresponding microscales
// - Each row is loaded in chunks of size 16 and each thread loads 32 elements
template <typename AType, typename AFragT, int32_t BLOCK_M, int32_t BLOCK_K>
__device__ AFragT load_A_col_major(AType const* input_ptr)
{
    // clang-format off
    // Register Mapping for 16x128 for FP8:                                                ||    Register Mapping for 32x64 for FP8:
    // Size              |   BLOCK_M  |   BLOCK_M   |   BLOCK_M  |   BLOCK_M   |           ||    Size              |   BLOCK_M  |   BLOCK_M   |        |
    // M                 | 0  ...  15 |  0  ...  15 | 0  ...  15 |  0  ...  15 | Vector    ||    M                 | 0  ...  31 |  0  ...  31 | Vector |
    // Thread Id         | 0  ...  15 | 16  ...  31 | 32  ... 47 | 48  ...  63 | Element   ||    Thread Id         | 0  ...  31 | 32  ...  63 | Element|
    // Register Element  |------------|-------------|------------|-------------|-----------||    Register Element  |------------|-------------|--------|
    // Reg 0 [0:7]       |     K0     |     K16     |     K32    |     K48     |  v[0]     ||    Reg 0 [0:7]       |     K0     |     K16     |  v[0]  |
    // Reg 0 [8:15]      |     K1     |     K17     |     K33    |     K49     |  v[1]     ||    Reg 0 [8:15]      |     K1     |     K17     |  v[1]  |
    // Reg 0 [16:23]     |     K2     |     K18     |     K34    |     K50     |  v[2]     ||    Reg 0 [16:23]     |     K2     |     K18     |  v[2]  |
    // Reg 0 [24:31]     |     K3     |     K19     |     K35    |     K51     |  v[3]     ||    Reg 0 [24:31]     |     K3     |     K19     |  v[3]  |
    // Reg 1 [0:7]       |     K4     |     K20     |     K36    |     K52     |  v[4]     ||    Reg 1 [0:7]       |     K4     |     K20     |  v[4]  |
    // Reg 1 [8:15]      |     K5     |     K21     |     K37    |     K53     |  v[5]     ||    Reg 1 [8:15]      |     K5     |     K21     |  v[5]  |
    // Reg 1 [16:23]     |     K6     |     K22     |     K38    |     K54     |  v[6]     ||    Reg 1 [16:23]     |     K6     |     K22     |  v[6]  |
    // Reg 1 [24:31]     |     K7     |     K23     |     K39    |     K55     |  v[7]     ||    Reg 1 [24:31]     |     K7     |     K23     |  v[7]  |
    // Reg 2 [0:7]       |     K8     |     K24     |     K40    |     K56     |  v[8]     ||    Reg 2 [0:7]       |     K8     |     K24     |  v[8]  |
    // Reg 2 [8:15]      |     K9     |     K25     |     K41    |     K57     |  v[9]     ||    Reg 2 [8:15]      |     K9     |     K25     |  v[9]  |
    // Reg 2 [16:23]     |     K10    |     K26     |     K42    |     K58     |  v[10]    ||    Reg 2 [16:23]     |     K10    |     K26     |  v[10] |
    // Reg 2 [24:31]     |     K11    |     K27     |     K43    |     K59     |  v[11]    ||    Reg 2 [24:31]     |     K11    |     K27     |  v[11] |
    // Reg 3 [0:7]       |     K12    |     K28     |     K44    |     K60     |  v[12]    ||    Reg 3 [0:7]       |     K12    |     K28     |  v[12] |
    // Reg 3 [8:15]      |     K13    |     K29     |     K45    |     K61     |  v[13]    ||    Reg 3 [8:15]      |     K13    |     K29     |  v[13] |
    // Reg 3 [16:23]     |     K14    |     K30     |     K46    |     K62     |  v[14]    ||    Reg 3 [16:23]     |     K14    |     K30     |  v[14] |
    // Reg 3 [24:31]     |     K15    |     K31     |     K47    |     K63     |  v[15]    ||    Reg 3 [24:31]     |     K15    |     K31     |  v[15] |
    // Reg 4 [0:7]       |     K64    |     K80     |     K96    |     K112    |  v[16]    ||    Reg 4 [0:7]       |     K32    |     K48     |  v[16] |
    // Reg 4 [8:15]      |     K65    |     K81     |     K97    |     K113    |  v[17]    ||    Reg 4 [8:15]      |     K33    |     K49     |  v[17] |
    // Reg 4 [16:23]     |     K66    |     K82     |     K98    |     K114    |  v[18]    ||    Reg 4 [16:23]     |     K34    |     K50     |  v[18] |
    // Reg 4 [24:31]     |     K67    |     K83     |     K99    |     K115    |  v[19]    ||    Reg 4 [24:31]     |     K35    |     K51     |  v[19] |
    // Reg 5 [0:7]       |     K68    |     K84     |     K100   |     K116    |  v[20]    ||    Reg 5 [0:7]       |     K36    |     K52     |  v[20] |
    // Reg 5 [8:15]      |     K69    |     K85     |     K101   |     K117    |  v[21]    ||    Reg 5 [8:15]      |     K37    |     K53     |  v[21] |
    // Reg 5 [16:23]     |     K70    |     K86     |     K102   |     K118    |  v[22]    ||    Reg 5 [16:23]     |     K38    |     K54     |  v[22] |
    // Reg 5 [24:31]     |     K71    |     K87     |     K103   |     K119    |  v[23]    ||    Reg 5 [24:31]     |     K39    |     K55     |  v[23] |
    // Reg 6 [0:7]       |     K72    |     K88     |     K104   |     K120    |  v[24]    ||    Reg 6 [0:7]       |     K40    |     K56     |  v[24] |
    // Reg 6 [8:15]      |     K73    |     K89     |     K105   |     K121    |  v[25]    ||    Reg 6 [8:15]      |     K41    |     K57     |  v[25] |
    // Reg 6 [16:23]     |     K74    |     K90     |     K106   |     K122    |  v[26]    ||    Reg 6 [16:23]     |     K42    |     K58     |  v[26] |
    // Reg 6 [24:31]     |     K75    |     K91     |     K107   |     K123    |  v[27]    ||    Reg 6 [24:31]     |     K43    |     K59     |  v[27] |
    // Reg 7 [0:7]       |     K76    |     K92     |     K108   |     K124    |  v[28]    ||    Reg 7 [0:7]       |     K44    |     K60     |  v[28] |
    // Reg 7 [8:15]      |     K77    |     K93     |     K109   |     K125    |  v[29]    ||    Reg 7 [8:15]      |     K45    |     K61     |  v[29] |
    // Reg 7 [16:23]     |     K78    |     K94     |     K110   |     K126    |  v[30]    ||    Reg 7 [16:23]     |     K46    |     K62     |  v[30] |
    // Reg 7 [24:31]     |     K79    |     K95     |     K111   |     K127    |  v[31]    ||    Reg 7 [24:31]     |     K47    |     K63     |  v[31] |
    // clang-format on

    static_assert(!is_packed_type_v<AType>, "Packed type is not supported");

    static constexpr int32_t WAVE_SIZE = 64;

    // Here we want to load from rows of A in chunks of 16 elements each.
    static constexpr uint32_t chunk_size = 16;

    // each chunk is separated by offset
    static constexpr uint32_t chunk_offset = chunk_size * WAVE_SIZE / BLOCK_M;

    // To start the loading process, let's visualize in 2D coords.
    // Each thread will load 32 elements.
    // We need to know where they start, and where the next elements are.
    auto startCoord2D =
        std::make_pair(threadIdx.x % BLOCK_M,                 // Row {0-31}  |  {0-15}
                       (threadIdx.x / BLOCK_M) * chunk_size); // Col {0, 16} |  {0, 16, 32, 48}

    auto minorStepCoord2D = std::make_pair(0u, 1u);          // read rows
    auto majorStepCoord2D = std::make_pair(0, chunk_offset); // read a chunk from a row

    // Flatten to 1D col_major offsets.
    auto col_major = [](auto const& coord, auto ld) { return coord.first + coord.second * ld; };

    // BLOCK_M is a stride in A matrix
    auto startOffset  = col_major(startCoord2D, BLOCK_M);
    auto kMinorOffset = col_major(minorStepCoord2D, BLOCK_M);
    auto kMajorOffset = col_major(majorStepCoord2D, BLOCK_M);

    using ARawT        = typename scalar_type<AFragT>::type;
    using AScalarFragT = typename vector_type<
        ARawT,
        BLOCK_M * BLOCK_K / WAVE_SIZE /
            (ck::is_same_v<ck::remove_cvref_t<AType>, ck::f4x2_pk_t> ? 2 : 1)>::type;

    AScalarFragT fragA{};

    constexpr index_t num_chunks =
        (ck::is_same_v<ck::remove_cvref_t<AType>, ck::f4x2_pk_t> ? 1 : 2);

#pragma unroll
    for(int chunk = 0; chunk < num_chunks; chunk++)
    {
#pragma unroll
        for(uint32_t i = 0; i < chunk_size; i++)
        {
            fragA[chunk * chunk_size + i] =
                bit_cast<ARawT>(input_ptr[startOffset + chunk * kMajorOffset + i * kMinorOffset]);
        }
    }

    return fragA;
}

// Define a load function for input A blocks:
// Size: (BLOCK_M x BLOCK_K)
// - Data is in row major format
// - Rows are loaded in contiguous chunks that map to corresponding microscales
// - Each row is loaded in chunks of size 16 and each thread loads 32 elements
template <typename AType, typename AFragT, int32_t BLOCK_M, int32_t BLOCK_K>
__device__ AFragT load_A_row_major(AType const* input_ptr)
{
    // clang-format off
    // Register Mapping for 16x128:                                                        ||    Register Mapping for 32x64:
    // Size              |   BLOCK_M  |   BLOCK_M   |   BLOCK_M  |   BLOCK_M   |           ||    Size              |   BLOCK_M  |   BLOCK_M   |        |
    // M                 | 0  ...  15 |  0  ...  15 | 0  ...  15 |  0  ...  15 | Vector    ||    M                 | 0  ...  31 |  0  ...  31 | Vector |
    // Thread Id         | 0  ...  15 | 16  ...  31 | 32  ... 47 | 48  ...  63 | Element   ||    Thread Id         | 0  ...  31 | 32  ...  63 | Element|
    // Register Element  |------------|-------------|------------|-------------|-----------||    Register Element  |------------|-------------|--------|
    // Reg 0 [0:7]       |     K0     |     K16     |     K32    |     K48     |  v[0]     ||    Reg 0 [0:7]       |     K0     |     K16     |  v[0]  |
    // Reg 0 [8:15]      |     K1     |     K17     |     K33    |     K49     |  v[1]     ||    Reg 0 [8:15]      |     K1     |     K17     |  v[1]  |
    // Reg 0 [16:23]     |     K2     |     K18     |     K34    |     K50     |  v[2]     ||    Reg 0 [16:23]     |     K2     |     K18     |  v[2]  |
    // Reg 0 [24:31]     |     K3     |     K19     |     K35    |     K51     |  v[3]     ||    Reg 0 [24:31]     |     K3     |     K19     |  v[3]  |
    // Reg 1 [0:7]       |     K4     |     K20     |     K36    |     K52     |  v[4]     ||    Reg 1 [0:7]       |     K4     |     K20     |  v[4]  |
    // Reg 1 [8:15]      |     K5     |     K21     |     K37    |     K53     |  v[5]     ||    Reg 1 [8:15]      |     K5     |     K21     |  v[5]  |
    // Reg 1 [16:23]     |     K6     |     K22     |     K38    |     K54     |  v[6]     ||    Reg 1 [16:23]     |     K6     |     K22     |  v[6]  |
    // Reg 1 [24:31]     |     K7     |     K23     |     K39    |     K55     |  v[7]     ||    Reg 1 [24:31]     |     K7     |     K23     |  v[7]  |
    // Reg 2 [0:7]       |     K8     |     K24     |     K40    |     K56     |  v[8]     ||    Reg 2 [0:7]       |     K8     |     K24     |  v[8]  |
    // Reg 2 [8:15]      |     K9     |     K25     |     K41    |     K57     |  v[9]     ||    Reg 2 [8:15]      |     K9     |     K25     |  v[9]  |
    // Reg 2 [16:23]     |     K10    |     K26     |     K42    |     K58     |  v[10]    ||    Reg 2 [16:23]     |     K10    |     K26     |  v[10] |
    // Reg 2 [24:31]     |     K11    |     K27     |     K43    |     K59     |  v[11]    ||    Reg 2 [24:31]     |     K11    |     K27     |  v[11] |
    // Reg 3 [0:7]       |     K12    |     K28     |     K44    |     K60     |  v[12]    ||    Reg 3 [0:7]       |     K12    |     K28     |  v[12] |
    // Reg 3 [8:15]      |     K13    |     K29     |     K45    |     K61     |  v[13]    ||    Reg 3 [8:15]      |     K13    |     K29     |  v[13] |
    // Reg 3 [16:23]     |     K14    |     K30     |     K46    |     K62     |  v[14]    ||    Reg 3 [16:23]     |     K14    |     K30     |  v[14] |
    // Reg 3 [24:31]     |     K15    |     K31     |     K47    |     K63     |  v[15]    ||    Reg 3 [24:31]     |     K15    |     K31     |  v[15] |
    // Reg 4 [0:7]       |     K64    |     K80     |     K96    |     K112    |  v[16]    ||    Reg 4 [0:7]       |     K32    |     K48     |  v[16] |
    // Reg 4 [8:15]      |     K65    |     K81     |     K97    |     K113    |  v[17]    ||    Reg 4 [8:15]      |     K33    |     K49     |  v[17] |
    // Reg 4 [16:23]     |     K66    |     K82     |     K98    |     K114    |  v[18]    ||    Reg 4 [16:23]     |     K34    |     K50     |  v[18] |
    // Reg 4 [24:31]     |     K67    |     K83     |     K99    |     K115    |  v[19]    ||    Reg 4 [24:31]     |     K35    |     K51     |  v[19] |
    // Reg 5 [0:7]       |     K68    |     K84     |     K100   |     K116    |  v[20]    ||    Reg 5 [0:7]       |     K36    |     K52     |  v[20] |
    // Reg 5 [8:15]      |     K69    |     K85     |     K101   |     K117    |  v[21]    ||    Reg 5 [8:15]      |     K37    |     K53     |  v[21] |
    // Reg 5 [16:23]     |     K70    |     K86     |     K102   |     K118    |  v[22]    ||    Reg 5 [16:23]     |     K38    |     K54     |  v[22] |
    // Reg 5 [24:31]     |     K71    |     K87     |     K103   |     K119    |  v[23]    ||    Reg 5 [24:31]     |     K39    |     K55     |  v[23] |
    // Reg 6 [0:7]       |     K72    |     K88     |     K104   |     K120    |  v[24]    ||    Reg 6 [0:7]       |     K40    |     K56     |  v[24] |
    // Reg 6 [8:15]      |     K73    |     K89     |     K105   |     K121    |  v[25]    ||    Reg 6 [8:15]      |     K41    |     K57     |  v[25] |
    // Reg 6 [16:23]     |     K74    |     K90     |     K106   |     K122    |  v[26]    ||    Reg 6 [16:23]     |     K42    |     K58     |  v[26] |
    // Reg 6 [24:31]     |     K75    |     K91     |     K107   |     K123    |  v[27]    ||    Reg 6 [24:31]     |     K43    |     K59     |  v[27] |
    // Reg 7 [0:7]       |     K76    |     K92     |     K108   |     K124    |  v[28]    ||    Reg 7 [0:7]       |     K44    |     K60     |  v[28] |
    // Reg 7 [8:15]      |     K77    |     K93     |     K109   |     K125    |  v[29]    ||    Reg 7 [8:15]      |     K45    |     K61     |  v[29] |
    // Reg 7 [16:23]     |     K78    |     K94     |     K110   |     K126    |  v[30]    ||    Reg 7 [16:23]     |     K46    |     K62     |  v[30] |
    // Reg 7 [24:31]     |     K79    |     K95     |     K111   |     K127    |  v[31]    ||    Reg 7 [24:31]     |     K47    |     K63     |  v[31] |

    // Register Mapping for 16x128 for FP4:                                                ||    Register Mapping for 32x64 for FP4:
    // Size              |   BLOCK_M  |   BLOCK_M   |   BLOCK_M  |   BLOCK_M   |           ||    Size              |   BLOCK_M  |   BLOCK_M   |        |
    // M                 | 0  ...  15 |  0  ...  15 | 0  ...  15 |  0  ...  15 | Vector    ||    M                 | 0  ...  31 |  0  ...  31 | Vector |
    // Thread Id         | 0  ...  15 | 16  ...  31 | 32  ... 47 | 48  ...  63 | Element   ||    Thread Id         | 0  ...  31 | 32  ...  63 | Element|
    // Register Element  |------------|-------------|------------|-------------|-----------||    Register Element  |------------|-------------|--------|
    // Reg 0 [0:7]       |     K0K1   |     K32K33  |     K64K65 |    K96K97   |  v[0]     ||    Reg 0 [0:7]       |     K0K1   |     K32K33  |  v[0]  |
    // Reg 0 [8:15]      |     K2K3   |     K34K35  |     K66K67 |    K98K99   |  v[1]     ||    Reg 0 [8:15]      |     K2K3   |     K34K35  |  v[1]  |
    // Reg 0 [16:23]     |     K4K5   |     K36K37  |     K68K69 |    K100K101 |  v[2]     ||    Reg 0 [16:23]     |     K4K5   |     K36K37  |  v[2]  |
    // Reg 0 [24:31]     |     K6K7   |     K38K39  |     K70K71 |    K102K103 |  v[3]     ||    Reg 0 [24:31]     |     K6K7   |     K38K39  |  v[3]  |
    // Reg 1 [0:7]       |     K8K9   |     K40K41  |     K72K73 |    K104K105 |  v[4]     ||    Reg 1 [0:7]       |     K8K9   |     K40K41  |  v[4]  |
    // Reg 1 [8:15]      |     K10K11 |     K42K43  |     K74K75 |    K106K107 |  v[5]     ||    Reg 1 [8:15]      |     K10K11 |     K42K43  |  v[5]  |
    // Reg 1 [16:23]     |     K12K13 |     K44K45  |     K76K77 |    K108K109 |  v[6]     ||    Reg 1 [16:23]     |     K12K13 |     K44K45  |  v[6]  |
    // Reg 1 [24:31]     |     K14K15 |     K46K47  |     K78K79 |    K110K111 |  v[7]     ||    Reg 1 [24:31]     |     K14K15 |     K46K47  |  v[7]  |
    // Reg 2 [0:7]       |     K16K17 |     K48K49  |     K80K81 |    K112K113 |  v[8]     ||    Reg 2 [0:7]       |     K16K17 |     K48K49  |  v[8]  |
    // Reg 2 [8:15]      |     K18K19 |     K50K51  |     K82K83 |    K114K115 |  v[9]     ||    Reg 2 [8:15]      |     K18K19 |     K50K51  |  v[9]  |
    // Reg 2 [16:23]     |     K20K21 |     K52K53  |     K84K85 |    K116K117 |  v[10]    ||    Reg 2 [16:23]     |     K20K21 |     K52K53  |  v[10] |
    // Reg 2 [24:31]     |     K22K23 |     K54K55  |     K86K87 |    K118K119 |  v[11]    ||    Reg 2 [24:31]     |     K22K23 |     K54K55  |  v[11] |
    // Reg 3 [0:7]       |     K24K25 |     K56K57  |     K88K89 |    K120K121 |  v[12]    ||    Reg 3 [0:7]       |     K24K25 |     K56K57  |  v[12] |
    // Reg 3 [8:15]      |     K26K27 |     K58K59  |     K90K91 |    K122K123 |  v[13]    ||    Reg 3 [8:15]      |     K26K27 |     K58K59  |  v[13] |
    // Reg 3 [16:23]     |     K28K29 |     K60K61  |     K92K93 |    K124K125 |  v[14]    ||    Reg 3 [16:23]     |     K28K29 |     K60K61  |  v[14] |
    // Reg 3 [24:31]     |     K30K31 |     K62K63  |     K94K95 |    K126K127 |  v[15]    ||    Reg 3 [24:31]     |     K30K31 |     K62K63  |  v[15] |


    // Register Mapping for 16x128 for FP6:                                                ||    Register Mapping for 32x64 for FP6:
    // Size              |   BLOCK_M  |   BLOCK_M   |   BLOCK_M  |   BLOCK_M   |           ||    Size              |   BLOCK_M  |   BLOCK_M   |        |
    // M                 | 0  ...  15 |  0  ...  15 | 0  ...  15 |  0  ...  15 | Vector    ||    M                 | 0  ...  31 |  0  ...  31 | Vector |
    // Thread Id         | 0  ...  15 | 16  ...  31 | 32  ... 47 | 48  ...  63 | Element   ||    Thread Id         | 0  ...  31 | 32  ...  63 | Element|
    // Register Element  |------------|-------------|------------|-------------|-----------||    Register Element  |------------|-------------|--------|
    // Reg 0-2 [0:95]    | K =  0-15  |  K = 32-47  |  K = 64-79 | K = 96-111  |  v[0]     ||    Reg 0-2 [0:95]    | K =  0-15  |  K = 32-47  |  v[0]  |
    // Reg 3-5 [0:95]    | K = 16-31  |  K = 48-63  |  K = 80-95 | K = 112-127 |  v[0]     ||    Reg 3-5 [0:95]    | K = 16-31  |  K = 48-63  |  v[0]  |

    // clang-format on

    static constexpr int32_t WAVE_SIZE = 64;

    // FP8 chunk_size = 16, num_chunks = 2, packed_size = 1
    // FP4 chunk_size = 32, num_chunks = 1, packed_size = 2
    // FP6 chunk_size = 32, num_chunks = 1, packed_size = 32

    constexpr index_t num_chunks = is_packed_type_v<AType> ? 1 : 2;

    // Here we want to load from rows of A in chunks of 16 elements each.
    constexpr uint32_t chunk_size = is_packed_type_v<AType> ? 32 : 16;

    // each chunk is separated by offset
    static constexpr uint32_t chunk_offset = chunk_size * WAVE_SIZE / BLOCK_M;

    // To start the loading process, let's visualize in 2D coords.
    // Each thread will load 32 elements.
    // We need to know where they start, and where the next elements are.
    // FP8/6/4 Row {0-31}  |  {0-15}
    // FP8     Col {0, 16} |  {0, 16, 32, 48}
    // FP6/4   Col {0, 32} |  {0, 32, 64, 96}
    auto startCoord2D = std::make_pair(threadIdx.x % BLOCK_M, (threadIdx.x / BLOCK_M) * chunk_size);

    auto majorStepCoord2D = std::make_pair(0, chunk_offset); // read a chunk from a row

    // Flatten to 1D row_major offsets.
    auto row_major = [](auto const& coord, auto ld) { return coord.first * ld + coord.second; };

    using ARawT = typename scalar_type<AFragT>::type;
    using AScalarChunkT =
        typename vector_type<ARawT, scalar_type<AFragT>::vector_size / num_chunks>::type;

    union
    {
        AFragT frag;
        AScalarChunkT chunks[num_chunks];
    } fragA{};

    const AScalarChunkT* fragPtr;

    // BLOCK_K is a stride in A matrix
    auto startOffset  = row_major(startCoord2D, BLOCK_K) / packed_size_v<AType>;
    auto kMajorOffset = row_major(majorStepCoord2D, BLOCK_K) / packed_size_v<AType>;

    for(index_t chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++)
    {
        fragPtr                 = reinterpret_cast<AScalarChunkT const*>(input_ptr + startOffset +
                                                         chunk_idx * kMajorOffset);
        fragA.chunks[chunk_idx] = *fragPtr;
    }

    return fragA.frag;
}

// Define a load function for scaled A blocks:
// Size: (BLOCK_M x BLOCK_K)
// ASSUMPTION:
// - The scale inputs distributed across 64 lanes.
template <typename AType,
          typename AFragT,
          typename ScaleType,
          typename ScaleFragT,
          int32_t BLOCK_M,
          int32_t BLOCK_K,
          int32_t BLOCK_X>
__device__ AFragT load_mx_A_row_major(AType const* input_ptr,
                                      ScaleType const* scale_ptr,
                                      ScaleFragT& fragX)
{
    // clang-format off
    // Register Mapping for 16x128:                                                                              ||    Register Mapping for 32x64:
    // Size              |   BLOCK_M  |   BLOCK_M   |          |   BLOCK_M  |   BLOCK_M   |          |           ||    Size              |   BLOCK_M  |   BLOCK_M   |        |          |
    // M                 | 0  ...  15 |  0  ...  15 |          | 0  ...  15 |  0  ...  15 |          | Vector    ||    M                 | 0  ...  31 |  0  ...  31 | Vector |          |
    // Thread Id         | 0  ...  15 | 16  ...  31 |  Scale   | 32  ... 47 | 48  ...  63 |  Scale   | Element   ||    Thread Id         | 0  ...  31 | 32  ...  63 | Element|  Scale   |
    // Register Element   ------------ ------------- ----------|------------ ------------- ----------|-----------||    Register Element  |------------|-------------|--------|----------|
    // Reg 0 [0:7]       |     K0     |     K16     |  x(M,0)  |     K32    |     K48     |  x(M,1)  |  v[0]     ||    Reg 0 [0:7]       |     K0     |     K16     |  v[0]  |  x(M,0)  |
    // Reg 0 [8:15]      |     K1     |     K17     |  x(M,0)  |     K33    |     K49     |  x(M,1)  |  v[1]     ||    Reg 0 [8:15]      |     K1     |     K17     |  v[1]  |  x(M,0)  |
    // Reg 0 [16:23]     |     K2     |     K18     |  x(M,0)  |     K34    |     K50     |  x(M,1)  |  v[2]     ||    Reg 0 [16:23]     |     K2     |     K18     |  v[2]  |  x(M,0)  |
    // Reg 0 [24:31]     |     K3     |     K19     |  x(M,0)  |     K35    |     K51     |  x(M,1)  |  v[3]     ||    Reg 0 [24:31]     |     K3     |     K19     |  v[3]  |  x(M,0)  |
    // Reg 1 [0:7]       |     K4     |     K20     |  x(M,0)  |     K36    |     K52     |  x(M,1)  |  v[4]     ||    Reg 1 [0:7]       |     K4     |     K20     |  v[4]  |  x(M,0)  |
    // Reg 1 [8:15]      |     K5     |     K21     |  x(M,0)  |     K37    |     K53     |  x(M,1)  |  v[5]     ||    Reg 1 [8:15]      |     K5     |     K21     |  v[5]  |  x(M,0)  |
    // Reg 1 [16:23]     |     K6     |     K22     |  x(M,0)  |     K38    |     K54     |  x(M,1)  |  v[6]     ||    Reg 1 [16:23]     |     K6     |     K22     |  v[6]  |  x(M,0)  |
    // Reg 1 [24:31]     |     K7     |     K23     |  x(M,0)  |     K39    |     K55     |  x(M,1)  |  v[7]     ||    Reg 1 [24:31]     |     K7     |     K23     |  v[7]  |  x(M,0)  |
    // Reg 2 [0:7]       |     K8     |     K24     |  x(M,0)  |     K40    |     K56     |  x(M,1)  |  v[8]     ||    Reg 2 [0:7]       |     K8     |     K24     |  v[8]  |  x(M,0)  |
    // Reg 2 [8:15]      |     K9     |     K25     |  x(M,0)  |     K41    |     K57     |  x(M,1)  |  v[9]     ||    Reg 2 [8:15]      |     K9     |     K25     |  v[9]  |  x(M,0)  |
    // Reg 2 [16:23]     |     K10    |     K26     |  x(M,0)  |     K42    |     K58     |  x(M,1)  |  v[10]    ||    Reg 2 [16:23]     |     K10    |     K26     |  v[10] |  x(M,0)  |
    // Reg 2 [24:31]     |     K11    |     K27     |  x(M,0)  |     K43    |     K59     |  x(M,1)  |  v[11]    ||    Reg 2 [24:31]     |     K11    |     K27     |  v[11] |  x(M,0)  |
    // Reg 3 [0:7]       |     K12    |     K28     |  x(M,0)  |     K44    |     K60     |  x(M,1)  |  v[12]    ||    Reg 3 [0:7]       |     K12    |     K28     |  v[12] |  x(M,0)  |
    // Reg 3 [8:15]      |     K13    |     K29     |  x(M,0)  |     K45    |     K61     |  x(M,1)  |  v[13]    ||    Reg 3 [8:15]      |     K13    |     K29     |  v[13] |  x(M,0)  |
    // Reg 3 [16:23]     |     K14    |     K30     |  x(M,0)  |     K46    |     K62     |  x(M,1)  |  v[14]    ||    Reg 3 [16:23]     |     K14    |     K30     |  v[14] |  x(M,0)  |
    // Reg 3 [24:31]     |     K15    |     K31     |  x(M,0)  |     K47    |     K63     |  x(M,1)  |  v[15]    ||    Reg 3 [24:31]     |     K15    |     K31     |  v[15] |  x(M,0)  |
    // Reg 4 [0:7]       |     K64    |     K80     |  x(M,2)  |     K96    |     K112    |  x(M,3)  |  v[16]    ||    Reg 4 [0:7]       |     K32    |     K48     |  v[16] |  x(M,1)  |
    // Reg 4 [8:15]      |     K65    |     K81     |  x(M,2)  |     K97    |     K113    |  x(M,3)  |  v[17]    ||    Reg 4 [8:15]      |     K33    |     K49     |  v[17] |  x(M,1)  |
    // Reg 4 [16:23]     |     K66    |     K82     |  x(M,2)  |     K98    |     K114    |  x(M,3)  |  v[18]    ||    Reg 4 [16:23]     |     K34    |     K50     |  v[18] |  x(M,1)  |
    // Reg 4 [24:31]     |     K67    |     K83     |  x(M,2)  |     K99    |     K115    |  x(M,3)  |  v[19]    ||    Reg 4 [24:31]     |     K35    |     K51     |  v[19] |  x(M,1)  |
    // Reg 5 [0:7]       |     K68    |     K84     |  x(M,2)  |     K100   |     K116    |  x(M,3)  |  v[20]    ||    Reg 5 [0:7]       |     K36    |     K52     |  v[20] |  x(M,1)  |
    // Reg 5 [8:15]      |     K69    |     K85     |  x(M,2)  |     K101   |     K117    |  x(M,3)  |  v[21]    ||    Reg 5 [8:15]      |     K37    |     K53     |  v[21] |  x(M,1)  |
    // Reg 5 [16:23]     |     K70    |     K86     |  x(M,2)  |     K102   |     K118    |  x(M,3)  |  v[22]    ||    Reg 5 [16:23]     |     K38    |     K54     |  v[22] |  x(M,1)  |
    // Reg 5 [24:31]     |     K71    |     K87     |  x(M,2)  |     K103   |     K119    |  x(M,3)  |  v[23]    ||    Reg 5 [24:31]     |     K39    |     K55     |  v[23] |  x(M,1)  |
    // Reg 6 [0:7]       |     K72    |     K88     |  x(M,2)  |     K104   |     K120    |  x(M,3)  |  v[24]    ||    Reg 6 [0:7]       |     K40    |     K56     |  v[24] |  x(M,1)  |
    // Reg 6 [8:15]      |     K73    |     K89     |  x(M,2)  |     K105   |     K121    |  x(M,3)  |  v[25]    ||    Reg 6 [8:15]      |     K41    |     K57     |  v[25] |  x(M,1)  |
    // Reg 6 [16:23]     |     K74    |     K90     |  x(M,2)  |     K106   |     K122    |  x(M,3)  |  v[26]    ||    Reg 6 [16:23]     |     K42    |     K58     |  v[26] |  x(M,1)  |
    // Reg 6 [24:31]     |     K75    |     K91     |  x(M,2)  |     K107   |     K123    |  x(M,3)  |  v[27]    ||    Reg 6 [24:31]     |     K43    |     K59     |  v[27] |  x(M,1)  |
    // Reg 7 [0:7]       |     K76    |     K92     |  x(M,2)  |     K108   |     K124    |  x(M,3)  |  v[28]    ||    Reg 7 [0:7]       |     K44    |     K60     |  v[28] |  x(M,1)  |
    // Reg 7 [8:15]      |     K77    |     K93     |  x(M,2)  |     K109   |     K125    |  x(M,3)  |  v[29]    ||    Reg 7 [8:15]      |     K45    |     K61     |  v[29] |  x(M,1)  |
    // Reg 7 [16:23]     |     K78    |     K94     |  x(M,2)  |     K110   |     K126    |  x(M,3)  |  v[30]    ||    Reg 7 [16:23]     |     K46    |     K62     |  v[30] |  x(M,1)  |
    // Reg 7 [24:31]     |     K79    |     K95     |  x(M,2)  |     K111   |     K127    |  x(M,3)  |  v[31]    ||    Reg 7 [24:31]     |     K47    |     K63     |  v[31] |  x(M,1)  |

    // Register Mapping for 16x128 for FP4:                                                                                            ||    Register Mapping for 32x64 for FP4:
    // Size              |   BLOCK_M  |          |   BLOCK_M   |          |   BLOCK_M  |          |   BLOCK_M   |          |           ||    Size              |   BLOCK_M  |          |   BLOCK_M   |          |        |
    // M                 | 0  ...  15 |          |  0  ...  15 |          | 0  ...  15 |          |  0  ...  15 |          | Vector    ||    M                 | 0  ...  31 |          |  0  ...  31 |          | Vector |
    // Thread Id         | 0  ...  15 |  Scale   | 16  ...  31 |  Scale   | 32  ... 47 |  Scale   | 48  ...  63 |  Scale   | Element   ||    Thread Id         | 0  ...  31 |  Scale   | 32  ...  63 |  Scale   | Element|
    // Register Element  |------------ ----------|------------- ----------|------------ ----------|------------- ----------|-----------||    Register Element  |------------|----------|-------------|----------|--------|
    // Reg 0 [0:7]       |     K0K1   |  x(M,0)  |     K32K33  |  x(M,1)  |     K64K65 |  x(M,2)  |    K96K97   |  x(M,3)  |  v[0]     ||    Reg 0 [0:7]       |     K0K1   |  x(M,0)  |     K32K33  |  x(M,1)  |  v[0]  |
    // Reg 0 [8:15]      |     K2K3   |  x(M,0)  |     K34K35  |  x(M,1)  |     K66K67 |  x(M,2)  |    K98K99   |  x(M,3)  |  v[1]     ||    Reg 0 [8:15]      |     K2K3   |  x(M,0)  |     K34K35  |  x(M,1)  |  v[1]  |
    // Reg 0 [16:23]     |     K4K5   |  x(M,0)  |     K36K37  |  x(M,1)  |     K68K69 |  x(M,2)  |    K100K101 |  x(M,3)  |  v[2]     ||    Reg 0 [16:23]     |     K4K5   |  x(M,0)  |     K36K37  |  x(M,1)  |  v[2]  |
    // Reg 0 [24:31]     |     K6K7   |  x(M,0)  |     K38K39  |  x(M,1)  |     K70K71 |  x(M,2)  |    K102K103 |  x(M,3)  |  v[3]     ||    Reg 0 [24:31]     |     K6K7   |  x(M,0)  |     K38K39  |  x(M,1)  |  v[3]  |
    // Reg 1 [0:7]       |     K8K9   |  x(M,0)  |     K40K41  |  x(M,1)  |     K72K73 |  x(M,2)  |    K104K105 |  x(M,3)  |  v[4]     ||    Reg 1 [0:7]       |     K8K9   |  x(M,0)  |     K40K41  |  x(M,1)  |  v[4]  |
    // Reg 1 [8:15]      |     K10K11 |  x(M,0)  |     K42K43  |  x(M,1)  |     K74K75 |  x(M,2)  |    K106K107 |  x(M,3)  |  v[5]     ||    Reg 1 [8:15]      |     K10K11 |  x(M,0)  |     K42K43  |  x(M,1)  |  v[5]  |
    // Reg 1 [16:23]     |     K12K13 |  x(M,0)  |     K44K45  |  x(M,1)  |     K76K77 |  x(M,2)  |    K108K109 |  x(M,3)  |  v[6]     ||    Reg 1 [16:23]     |     K12K13 |  x(M,0)  |     K44K45  |  x(M,1)  |  v[6]  |
    // Reg 1 [24:31]     |     K14K15 |  x(M,0)  |     K46K47  |  x(M,1)  |     K78K79 |  x(M,2)  |    K110K111 |  x(M,3)  |  v[7]     ||    Reg 1 [24:31]     |     K14K15 |  x(M,0)  |     K46K47  |  x(M,1)  |  v[7]  |
    // Reg 2 [0:7]       |     K16K17 |  x(M,0)  |     K48K49  |  x(M,1)  |     K80K81 |  x(M,2)  |    K112K113 |  x(M,3)  |  v[8]     ||    Reg 2 [0:7]       |     K16K17 |  x(M,0)  |     K48K49  |  x(M,1)  |  v[8]  |
    // Reg 2 [8:15]      |     K18K19 |  x(M,0)  |     K50K51  |  x(M,1)  |     K82K83 |  x(M,2)  |    K114K115 |  x(M,3)  |  v[9]     ||    Reg 2 [8:15]      |     K18K19 |  x(M,0)  |     K50K51  |  x(M,1)  |  v[9]  |
    // Reg 2 [16:23]     |     K20K21 |  x(M,0)  |     K52K53  |  x(M,1)  |     K84K85 |  x(M,2)  |    K116K117 |  x(M,3)  |  v[10]    ||    Reg 2 [16:23]     |     K20K21 |  x(M,0)  |     K52K53  |  x(M,1)  |  v[10] |
    // Reg 2 [24:31]     |     K22K23 |  x(M,0)  |     K54K55  |  x(M,1)  |     K86K87 |  x(M,2)  |    K118K119 |  x(M,3)  |  v[11]    ||    Reg 2 [24:31]     |     K22K23 |  x(M,0)  |     K54K55  |  x(M,1)  |  v[11] |
    // Reg 3 [0:7]       |     K24K25 |  x(M,0)  |     K56K57  |  x(M,1)  |     K88K89 |  x(M,2)  |    K120K121 |  x(M,3)  |  v[12]    ||    Reg 3 [0:7]       |     K24K25 |  x(M,0)  |     K56K57  |  x(M,1)  |  v[12] |
    // Reg 3 [8:15]      |     K26K27 |  x(M,0)  |     K58K59  |  x(M,1)  |     K90K91 |  x(M,2)  |    K122K123 |  x(M,3)  |  v[13]    ||    Reg 3 [8:15]      |     K26K27 |  x(M,0)  |     K58K59  |  x(M,1)  |  v[13] |
    // Reg 3 [16:23]     |     K28K29 |  x(M,0)  |     K60K61  |  x(M,1)  |     K92K93 |  x(M,2)  |    K124K125 |  x(M,3)  |  v[14]    ||    Reg 3 [16:23]     |     K28K29 |  x(M,0)  |     K60K61  |  x(M,1)  |  v[14] |
    // Reg 3 [24:31]     |     K30K31 |  x(M,0)  |     K62K63  |  x(M,1)  |     K94K95 |  x(M,2)  |    K126K127 |  x(M,3)  |  v[15]    ||    Reg 3 [24:31]     |     K30K31 |  x(M,0)  |     K62K63  |  x(M,1)  |  v[15] |
    // clang-format on

    // To start the loading process, let's visualize in 2D coords.
    // Each thread will load 1 element
    // We need to know where they start
    auto startCoord2D = std::make_pair(threadIdx.x % BLOCK_M,    // Row
                                       (threadIdx.x / BLOCK_M)); // Col

    // Flatten to 1D row_major offsets.
    auto row_major = [](auto const& coord, auto ld) { return coord.first * ld + coord.second; };

    // BLOCK_K / BLOCK_X is a stride in xA matrix
    auto startOffset = row_major(startCoord2D, BLOCK_K / BLOCK_X);

    fragX = scale_ptr[startOffset];

    return load_A_row_major<AType, AFragT, BLOCK_M, BLOCK_K>(input_ptr);
}

// Define a load function for input B blocks:
// Size: (BLOCK_K x BLOCK_N)
// - Data is in col major format
// - Cols are loaded in contiguous chunks that map to corresponding microscales
// - Each col is loaded in chunks of size 16 and each thread loads 32 elements
template <typename BType, typename BFragT, int32_t BLOCK_K, int32_t BLOCK_N>
__device__ BFragT load_B_col_major(BType const* input_ptr)
{
    // clang-format off
    // Register Mapping for 128x16 for FP8:                                                ||    Register Mapping for 64x32 for FP8:
    // Size              |   BLOCK_N  |   BLOCK_N   |   BLOCK_N  |   BLOCK_N   |           ||    Size              |   BLOCK_N  |   BLOCK_N   |        |
    // N                 | 0  ...  15 |  0  ...  15 | 0  ...  15 |  0  ...  15 | Vector    ||    N                 | 0  ...  31 |  0  ...  31 | Vector |
    // Thread Id         | 0  ...  15 | 16  ...  31 | 32  ... 47 | 48  ...  63 | Element   ||    Thread Id         | 0  ...  31 | 32  ...  63 | Element|
    // Register Element  |------------|-------------|------------|-------------|-----------||    Register Element  |------------|-------------|--------|
    // Reg 0 [0:7]       |     K0     |     K16     |     K32    |     K48     |  v[0]     ||    Reg 0 [0:7]       |     K0     |     K16     |  v[0]  |
    // Reg 0 [8:15]      |     K1     |     K17     |     K33    |     K49     |  v[1]     ||    Reg 0 [8:15]      |     K1     |     K17     |  v[1]  |
    // Reg 0 [16:23]     |     K2     |     K18     |     K34    |     K50     |  v[2]     ||    Reg 0 [16:23]     |     K2     |     K18     |  v[2]  |
    // Reg 0 [24:31]     |     K3     |     K19     |     K35    |     K51     |  v[3]     ||    Reg 0 [24:31]     |     K3     |     K19     |  v[3]  |
    // Reg 1 [0:7]       |     K4     |     K20     |     K36    |     K52     |  v[4]     ||    Reg 1 [0:7]       |     K4     |     K20     |  v[4]  |
    // Reg 1 [8:15]      |     K5     |     K21     |     K37    |     K53     |  v[5]     ||    Reg 1 [8:15]      |     K5     |     K21     |  v[5]  |
    // Reg 1 [16:23]     |     K6     |     K22     |     K38    |     K54     |  v[6]     ||    Reg 1 [16:23]     |     K6     |     K22     |  v[6]  |
    // Reg 1 [24:31]     |     K7     |     K23     |     K39    |     K55     |  v[7]     ||    Reg 1 [24:31]     |     K7     |     K23     |  v[7]  |
    // Reg 2 [0:7]       |     K8     |     K24     |     K40    |     K56     |  v[8]     ||    Reg 2 [0:7]       |     K8     |     K24     |  v[8]  |
    // Reg 2 [8:15]      |     K9     |     K25     |     K41    |     K57     |  v[9]     ||    Reg 2 [8:15]      |     K9     |     K25     |  v[9]  |
    // Reg 2 [16:23]     |     K10    |     K26     |     K42    |     K58     |  v[10]    ||    Reg 2 [16:23]     |     K10    |     K26     |  v[10] |
    // Reg 2 [24:31]     |     K11    |     K27     |     K43    |     K59     |  v[11]    ||    Reg 2 [24:31]     |     K11    |     K27     |  v[11] |
    // Reg 3 [0:7]       |     K12    |     K28     |     K44    |     K60     |  v[12]    ||    Reg 3 [0:7]       |     K12    |     K28     |  v[12] |
    // Reg 3 [8:15]      |     K13    |     K29     |     K45    |     K61     |  v[13]    ||    Reg 3 [8:15]      |     K13    |     K29     |  v[13] |
    // Reg 3 [16:23]     |     K14    |     K30     |     K46    |     K62     |  v[14]    ||    Reg 3 [16:23]     |     K14    |     K30     |  v[14] |
    // Reg 3 [24:31]     |     K15    |     K31     |     K47    |     K63     |  v[15]    ||    Reg 3 [24:31]     |     K15    |     K31     |  v[15] |
    // Reg 4 [0:7]       |     K64    |     K80     |     K96    |     K112    |  v[16]    ||    Reg 4 [0:7]       |     K32    |     K48     |  v[16] |
    // Reg 4 [8:15]      |     K65    |     K81     |     K97    |     K113    |  v[17]    ||    Reg 4 [8:15]      |     K33    |     K49     |  v[17] |
    // Reg 4 [16:23]     |     K66    |     K82     |     K98    |     K114    |  v[18]    ||    Reg 4 [16:23]     |     K34    |     K50     |  v[18] |
    // Reg 4 [24:31]     |     K67    |     K83     |     K99    |     K115    |  v[19]    ||    Reg 4 [24:31]     |     K35    |     K51     |  v[19] |
    // Reg 5 [0:7]       |     K68    |     K84     |     K100   |     K116    |  v[20]    ||    Reg 5 [0:7]       |     K36    |     K52     |  v[20] |
    // Reg 5 [8:15]      |     K69    |     K85     |     K101   |     K117    |  v[21]    ||    Reg 5 [8:15]      |     K37    |     K53     |  v[21] |
    // Reg 5 [16:23]     |     K70    |     K86     |     K102   |     K118    |  v[22]    ||    Reg 5 [16:23]     |     K38    |     K54     |  v[22] |
    // Reg 5 [24:31]     |     K71    |     K87     |     K103   |     K119    |  v[23]    ||    Reg 5 [24:31]     |     K39    |     K55     |  v[23] |
    // Reg 6 [0:7]       |     K72    |     K88     |     K104   |     K120    |  v[24]    ||    Reg 6 [0:7]       |     K40    |     K56     |  v[24] |
    // Reg 6 [8:15]      |     K73    |     K89     |     K105   |     K121    |  v[25]    ||    Reg 6 [8:15]      |     K41    |     K57     |  v[25] |
    // Reg 6 [16:23]     |     K74    |     K90     |     K106   |     K122    |  v[26]    ||    Reg 6 [16:23]     |     K42    |     K58     |  v[26] |
    // Reg 6 [24:31]     |     K75    |     K91     |     K107   |     K123    |  v[27]    ||    Reg 6 [24:31]     |     K43    |     K59     |  v[27] |
    // Reg 7 [0:7]       |     K76    |     K92     |     K108   |     K124    |  v[28]    ||    Reg 7 [0:7]       |     K44    |     K60     |  v[28] |
    // Reg 7 [8:15]      |     K77    |     K93     |     K109   |     K125    |  v[29]    ||    Reg 7 [8:15]      |     K45    |     K61     |  v[29] |
    // Reg 7 [16:23]     |     K78    |     K94     |     K110   |     K126    |  v[30]    ||    Reg 7 [16:23]     |     K46    |     K62     |  v[30] |
    // Reg 7 [24:31]     |     K79    |     K95     |     K111   |     K127    |  v[31]    ||    Reg 7 [24:31]     |     K47    |     K63     |  v[31] |

    // Register Mapping for 128x16 for FP4:                                                ||    Register Mapping for 64x32 for FP4:
    // Size              |   BLOCK_N  |   BLOCK_N   |   BLOCK_N  |   BLOCK_N   |           ||    Size              |   BLOCK_N  |   BLOCK_N   |        |
    // N                 | 0  ...  15 |  0  ...  15 | 0  ...  15 |  0  ...  15 | Vector    ||    N                 | 0  ...  31 |  0  ...  31 | Vector |
    // Thread Id         | 0  ...  15 | 16  ...  31 | 32  ... 47 | 48  ...  63 | Element   ||    Thread Id         | 0  ...  31 | 32  ...  63 | Element|
    // Register Element  |------------|-------------|------------|-------------|-----------||    Register Element  |------------|-------------|--------|
    // Reg 0 [0:7]       |     K0K1   |     K32K33  |     K64K65 |    K96K97   |  v[0]     ||    Reg 0 [0:7]       |     K0K1   |     K32K33  |  v[0]  |
    // Reg 0 [8:15]      |     K2K3   |     K34K35  |     K66K67 |    K98K99   |  v[1]     ||    Reg 0 [8:15]      |     K2K3   |     K34K35  |  v[1]  |
    // Reg 0 [16:23]     |     K4K5   |     K36K37  |     K68K69 |    K100K101 |  v[2]     ||    Reg 0 [16:23]     |     K4K5   |     K36K37  |  v[2]  |
    // Reg 0 [24:31]     |     K6K7   |     K38K39  |     K70K71 |    K102K103 |  v[3]     ||    Reg 0 [24:31]     |     K6K7   |     K38K39  |  v[3]  |
    // Reg 1 [0:7]       |     K8K9   |     K40K41  |     K72K73 |    K104K105 |  v[4]     ||    Reg 1 [0:7]       |     K8K9   |     K40K41  |  v[4]  |
    // Reg 1 [8:15]      |     K10K11 |     K42K43  |     K74K75 |    K106K107 |  v[5]     ||    Reg 1 [8:15]      |     K10K11 |     K42K43  |  v[5]  |
    // Reg 1 [16:23]     |     K12K13 |     K44K45  |     K76K77 |    K108K109 |  v[6]     ||    Reg 1 [16:23]     |     K12K13 |     K44K45  |  v[6]  |
    // Reg 1 [24:31]     |     K14K15 |     K46K47  |     K78K79 |    K110K111 |  v[7]     ||    Reg 1 [24:31]     |     K14K15 |     K46K47  |  v[7]  |
    // Reg 2 [0:7]       |     K16K17 |     K48K49  |     K80K81 |    K112K113 |  v[8]     ||    Reg 2 [0:7]       |     K16K17 |     K48K49  |  v[8]  |
    // Reg 2 [8:15]      |     K18K19 |     K50K51  |     K82K83 |    K114K115 |  v[9]     ||    Reg 2 [8:15]      |     K18K19 |     K50K51  |  v[9]  |
    // Reg 2 [16:23]     |     K20K21 |     K52K53  |     K84K85 |    K116K117 |  v[10]    ||    Reg 2 [16:23]     |     K20K21 |     K52K53  |  v[10] |
    // Reg 2 [24:31]     |     K22K23 |     K54K55  |     K86K87 |    K118K119 |  v[11]    ||    Reg 2 [24:31]     |     K22K23 |     K54K55  |  v[11] |
    // Reg 3 [0:7]       |     K24K25 |     K56K57  |     K88K89 |    K120K121 |  v[12]    ||    Reg 3 [0:7]       |     K24K25 |     K56K57  |  v[12] |
    // Reg 3 [8:15]      |     K26K27 |     K58K59  |     K90K91 |    K122K123 |  v[13]    ||    Reg 3 [8:15]      |     K26K27 |     K58K59  |  v[13] |
    // Reg 3 [16:23]     |     K28K29 |     K60K61  |     K92K93 |    K124K125 |  v[14]    ||    Reg 3 [16:23]     |     K28K29 |     K60K61  |  v[14] |
    // Reg 3 [24:31]     |     K30K31 |     K62K63  |     K94K95 |    K126K127 |  v[15]    ||    Reg 3 [24:31]     |     K30K31 |     K62K63  |  v[15] |

    // Register Mapping for 16x128 for FP6:                                                ||    Register Mapping for 32x64 for FP6:
    // Size              |   BLOCK_N  |   BLOCK_N   |   BLOCK_N  |   BLOCK_N   |           ||    Size              |   BLOCK_N  |   BLOCK_N   |        |
    // N                 | 0  ...  15 |  0  ...  15 | 0  ...  15 |  0  ...  15 | Vector    ||    N                 | 0  ...  31 |  0  ...  31 | Vector |
    // Thread Id         | 0  ...  15 | 16  ...  31 | 32  ... 47 | 48  ...  63 | Element   ||    Thread Id         | 0  ...  31 | 32  ...  63 | Element|
    // Register Element  |------------|-------------|------------|-------------|-----------||    Register Element  |------------|-------------|--------|
    // Reg 0-2 [0:95]    | K =  0-15  |  K = 32-47  |  K = 64-79 | K = 96-111  |  v[0]     ||    Reg 0-2 [0:95]    | K =  0-15  |  K = 32-47  |  v[0]  |
    // Reg 3-5 [0:95]    | K = 16-31  |  K = 48-63  |  K = 80-95 | K = 112-127 |  v[0]     ||    Reg 3-5 [0:95]    | K = 16-31  |  K = 48-63  |  v[0]  |

    // clang-format on

    static constexpr int32_t WAVE_SIZE = 64;

    // FP8 chunk_size = 16, num_chunks = 2, packed_size = 1
    // FP4 chunk_size = 32, num_chunks = 1, packed_size = 2
    // FP6 chunk_size = 32, num_chunks = 1, packed_size = 32

    constexpr index_t num_chunks = is_packed_type_v<BType> ? 1 : 2;

    // Here we want to load from cols of B in chunks of 16 elements each.
    constexpr uint32_t chunk_size = is_packed_type_v<BType> ? 32 : 16;

    // each chunk is separated by an offset
    static constexpr uint32_t chunk_offset = chunk_size * WAVE_SIZE / BLOCK_N; // 32 or 64

    // To start the loading process, let's visualize in 2D coords.
    // Each thread will load 32 elements.
    // We need to know where they start, and where the next elements are.
    // FP8/6/4 Col {0-31}  |  {0-15}
    // FP8     Row {0, 16} |  {0, 16, 32, 48}
    // FP6/4   Row {0, 32} |  {0, 32, 64, 96}
    auto startCoord2D = std::make_pair((threadIdx.x / BLOCK_N) * chunk_size, threadIdx.x % BLOCK_N);

    // Flatten to 1D col_major offsets.
    auto col_major = [](auto const& coord, auto ld) { return coord.first + coord.second * ld; };

    auto majorStepCoord2D = std::make_pair(chunk_offset, 0); // read a chunk from a col

    using BRawT = typename scalar_type<BFragT>::type;
    using BScalarChunkT =
        typename vector_type<BRawT, scalar_type<BFragT>::vector_size / num_chunks>::type;

    union
    {
        BFragT frag;
        BScalarChunkT chunks[num_chunks];
    } fragB{};

    const BScalarChunkT* fragPtr;

    // BLOCK_K is a stride in B matrix
    auto startOffset  = col_major(startCoord2D, BLOCK_K) / packed_size_v<BType>;
    auto kMajorOffset = col_major(majorStepCoord2D, BLOCK_K) / packed_size_v<BType>;

    for(index_t chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++)
    {
        fragPtr                 = reinterpret_cast<BScalarChunkT const*>(input_ptr + startOffset +
                                                         chunk_idx * kMajorOffset);
        fragB.chunks[chunk_idx] = *fragPtr;
    }

    return fragB.frag;
}

// Define a load function for scaled B blocks:
// Size: (BLOCK_K x BLOCK_N)
// ASSUMPTION:
// - The scale inputs distributed across 64 lanes.
template <typename BType,
          typename BFragT,
          typename ScaleType,
          typename ScaleFragT,
          int32_t BLOCK_K,
          int32_t BLOCK_N,
          int32_t BLOCK_X>
__device__ BFragT load_mx_B_col_major(BType const* input_ptr,
                                      ScaleType const* scale_ptr,
                                      ScaleFragT& fragX)

{
    // clang-format off
    // Register Mapping for 128x16:                                                                              ||    Register Mapping for 64x32:
    // Size              |   BLOCK_N  |   BLOCK_N   |          |   BLOCK_N  |   BLOCK_N   |          |           ||    Size              |   BLOCK_N  |   BLOCK_N   |        |          |
    // N                 | 0  ...  15 |  0  ...  15 |          | 0  ...  15 |  0  ...  15 |          | Vector    ||    N                 | 0  ...  31 |  0  ...  31 | Vector |          |
    // Thread Id         | 0  ...  15 | 16  ...  31 |  Scale   | 32  ... 47 | 48  ...  63 |  Scale   | Element   ||    Thread Id         | 0  ...  31 | 32  ...  63 | Element|  Scale   |
    // Register Element   ------------ ------------- ----------|------------ ------------- ----------|-----------||    Register Element  |------------|-------------|--------|----------|
    // Reg 0 [0:7]       |     K0     |     K16     |  x(0,N)  |     K32    |     K48     |  x(1,N)  |  v[0]     ||    Reg 0 [0:7]       |     K0     |     K16     |  v[0]  |  x(0,N)  |
    // Reg 0 [8:15]      |     K1     |     K17     |  x(0,N)  |     K33    |     K49     |  x(1,N)  |  v[1]     ||    Reg 0 [8:15]      |     K1     |     K17     |  v[1]  |  x(0,N)  |
    // Reg 0 [16:23]     |     K2     |     K18     |  x(0,N)  |     K34    |     K50     |  x(1,N)  |  v[2]     ||    Reg 0 [16:23]     |     K2     |     K18     |  v[2]  |  x(0,N)  |
    // Reg 0 [24:31]     |     K3     |     K19     |  x(0,N)  |     K35    |     K51     |  x(1,N)  |  v[3]     ||    Reg 0 [24:31]     |     K3     |     K19     |  v[3]  |  x(0,N)  |
    // Reg 1 [0:7]       |     K4     |     K20     |  x(0,N)  |     K36    |     K52     |  x(1,N)  |  v[4]     ||    Reg 1 [0:7]       |     K4     |     K20     |  v[4]  |  x(0,N)  |
    // Reg 1 [8:15]      |     K5     |     K21     |  x(0,N)  |     K37    |     K53     |  x(1,N)  |  v[5]     ||    Reg 1 [8:15]      |     K5     |     K21     |  v[5]  |  x(0,N)  |
    // Reg 1 [16:23]     |     K6     |     K22     |  x(0,N)  |     K38    |     K54     |  x(1,N)  |  v[6]     ||    Reg 1 [16:23]     |     K6     |     K22     |  v[6]  |  x(0,N)  |
    // Reg 1 [24:31]     |     K7     |     K23     |  x(0,N)  |     K39    |     K55     |  x(1,N)  |  v[7]     ||    Reg 1 [24:31]     |     K7     |     K23     |  v[7]  |  x(0,N)  |
    // Reg 2 [0:7]       |     K8     |     K24     |  x(0,N)  |     K40    |     K56     |  x(1,N)  |  v[8]     ||    Reg 2 [0:7]       |     K8     |     K24     |  v[8]  |  x(0,N)  |
    // Reg 2 [8:15]      |     K9     |     K25     |  x(0,N)  |     K41    |     K57     |  x(1,N)  |  v[9]     ||    Reg 2 [8:15]      |     K9     |     K25     |  v[9]  |  x(0,N)  |
    // Reg 2 [16:23]     |     K10    |     K26     |  x(0,N)  |     K42    |     K58     |  x(1,N)  |  v[10]    ||    Reg 2 [16:23]     |     K10    |     K26     |  v[10] |  x(0,N)  |
    // Reg 2 [24:31]     |     K11    |     K27     |  x(0,N)  |     K43    |     K59     |  x(1,N)  |  v[11]    ||    Reg 2 [24:31]     |     K11    |     K27     |  v[11] |  x(0,N)  |
    // Reg 3 [0:7]       |     K12    |     K28     |  x(0,N)  |     K44    |     K60     |  x(1,N)  |  v[12]    ||    Reg 3 [0:7]       |     K12    |     K28     |  v[12] |  x(0,N)  |
    // Reg 3 [8:15]      |     K13    |     K29     |  x(0,N)  |     K45    |     K61     |  x(1,N)  |  v[13]    ||    Reg 3 [8:15]      |     K13    |     K29     |  v[13] |  x(0,N)  |
    // Reg 3 [16:23]     |     K14    |     K30     |  x(0,N)  |     K46    |     K62     |  x(1,N)  |  v[14]    ||    Reg 3 [16:23]     |     K14    |     K30     |  v[14] |  x(0,N)  |
    // Reg 3 [24:31]     |     K15    |     K31     |  x(0,N)  |     K47    |     K63     |  x(1,N)  |  v[15]    ||    Reg 3 [24:31]     |     K15    |     K31     |  v[15] |  x(0,N)  |
    // Reg 4 [0:7]       |     K64    |     K80     |  x(2,N)  |     K96    |     K112    |  x(3,N)  |  v[16]    ||    Reg 4 [0:7]       |     K32    |     K48     |  v[16] |  x(1,N)  |
    // Reg 4 [8:15]      |     K65    |     K81     |  x(2,N)  |     K97    |     K113    |  x(3,N)  |  v[17]    ||    Reg 4 [8:15]      |     K33    |     K49     |  v[17] |  x(1,N)  |
    // Reg 4 [16:23]     |     K66    |     K82     |  x(2,N)  |     K98    |     K114    |  x(3,N)  |  v[18]    ||    Reg 4 [16:23]     |     K34    |     K50     |  v[18] |  x(1,N)  |
    // Reg 4 [24:31]     |     K67    |     K83     |  x(2,N)  |     K99    |     K115    |  x(3,N)  |  v[19]    ||    Reg 4 [24:31]     |     K35    |     K51     |  v[19] |  x(1,N)  |
    // Reg 5 [0:7]       |     K68    |     K84     |  x(2,N)  |     K100   |     K116    |  x(3,N)  |  v[20]    ||    Reg 5 [0:7]       |     K36    |     K52     |  v[20] |  x(1,N)  |
    // Reg 5 [8:15]      |     K69    |     K85     |  x(2,N)  |     K101   |     K117    |  x(3,N)  |  v[21]    ||    Reg 5 [8:15]      |     K37    |     K53     |  v[21] |  x(1,N)  |
    // Reg 5 [16:23]     |     K70    |     K86     |  x(2,N)  |     K102   |     K118    |  x(3,N)  |  v[22]    ||    Reg 5 [16:23]     |     K38    |     K54     |  v[22] |  x(1,N)  |
    // Reg 5 [24:31]     |     K71    |     K87     |  x(2,N)  |     K103   |     K119    |  x(3,N)  |  v[23]    ||    Reg 5 [24:31]     |     K39    |     K55     |  v[23] |  x(1,N)  |
    // Reg 6 [0:7]       |     K72    |     K88     |  x(2,N)  |     K104   |     K120    |  x(3,N)  |  v[24]    ||    Reg 6 [0:7]       |     K40    |     K56     |  v[24] |  x(1,N)  |
    // Reg 6 [8:15]      |     K73    |     K89     |  x(2,N)  |     K105   |     K121    |  x(3,N)  |  v[25]    ||    Reg 6 [8:15]      |     K41    |     K57     |  v[25] |  x(1,N)  |
    // Reg 6 [16:23]     |     K74    |     K90     |  x(2,N)  |     K106   |     K122    |  x(3,N)  |  v[26]    ||    Reg 6 [16:23]     |     K42    |     K58     |  v[26] |  x(1,N)  |
    // Reg 6 [24:31]     |     K75    |     K91     |  x(2,N)  |     K107   |     K123    |  x(3,N)  |  v[27]    ||    Reg 6 [24:31]     |     K43    |     K59     |  v[27] |  x(1,N)  |
    // Reg 7 [0:7]       |     K76    |     K92     |  x(2,N)  |     K108   |     K124    |  x(3,N)  |  v[28]    ||    Reg 7 [0:7]       |     K44    |     K60     |  v[28] |  x(1,N)  |
    // Reg 7 [8:15]      |     K77    |     K93     |  x(2,N)  |     K109   |     K125    |  x(3,N)  |  v[29]    ||    Reg 7 [8:15]      |     K45    |     K61     |  v[29] |  x(1,N)  |
    // Reg 7 [16:23]     |     K78    |     K94     |  x(2,N)  |     K110   |     K126    |  x(3,N)  |  v[30]    ||    Reg 7 [16:23]     |     K46    |     K62     |  v[30] |  x(1,N)  |
    // Reg 7 [24:31]     |     K79    |     K95     |  x(2,N)  |     K111   |     K127    |  x(3,N)  |  v[31]    ||    Reg 7 [24:31]     |     K47    |     K63     |  v[31] |  x(1,N)  |

    // Register Mapping for 128x16 for FP4:                                                ||    Register Mapping for 64x32 for FP4:
    // Size              |   BLOCK_N  |   BLOCK_N   |   BLOCK_N  |   BLOCK_N   |           ||    Size              |   BLOCK_N  |   BLOCK_N   |        |
    // N                 | 0  ...  15 |  0  ...  15 | 0  ...  15 |  0  ...  15 | Vector    ||    N                 | 0  ...  31 |  0  ...  31 | Vector |
    // Thread Id         | 0  ...  15 | 16  ...  31 | 32  ... 47 | 48  ...  63 | Element   ||    Thread Id         | 0  ...  31 | 32  ...  63 | Element|
    // Register Element  |------------|-------------|------------|-------------|-----------||    Register Element  |------------|-------------|--------|
    // Reg 0 [0:7]       |     K0K1   |     K32K33  |     K64K65 |    K96K97   |  v[0]     ||    Reg 0 [0:7]       |     K0K1   |     K32K33  |  v[0]  |
    // Reg 0 [8:15]      |     K2K3   |     K34K35  |     K66K67 |    K98K99   |  v[1]     ||    Reg 0 [8:15]      |     K2K3   |     K34K35  |  v[1]  |
    // Reg 0 [16:23]     |     K4K5   |     K36K37  |     K68K69 |    K100K101 |  v[2]     ||    Reg 0 [16:23]     |     K4K5   |     K36K37  |  v[2]  |
    // Reg 0 [24:31]     |     K6K7   |     K38K39  |     K70K71 |    K102K103 |  v[3]     ||    Reg 0 [24:31]     |     K6K7   |     K38K39  |  v[3]  |
    // Reg 1 [0:7]       |     K8K9   |     K40K41  |     K72K73 |    K104K105 |  v[4]     ||    Reg 1 [0:7]       |     K8K9   |     K40K41  |  v[4]  |
    // Reg 1 [8:15]      |     K10K11 |     K42K43  |     K74K75 |    K106K107 |  v[5]     ||    Reg 1 [8:15]      |     K10K11 |     K42K43  |  v[5]  |
    // Reg 1 [16:23]     |     K12K13 |     K44K45  |     K76K77 |    K108K109 |  v[6]     ||    Reg 1 [16:23]     |     K12K13 |     K44K45  |  v[6]  |
    // Reg 1 [24:31]     |     K14K15 |     K46K47  |     K78K79 |    K110K111 |  v[7]     ||    Reg 1 [24:31]     |     K14K15 |     K46K47  |  v[7]  |
    // Reg 2 [0:7]       |     K16K17 |     K48K49  |     K80K81 |    K112K113 |  v[8]     ||    Reg 2 [0:7]       |     K16K17 |     K48K49  |  v[8]  |
    // Reg 2 [8:15]      |     K18K19 |     K50K51  |     K82K83 |    K114K115 |  v[9]     ||    Reg 2 [8:15]      |     K18K19 |     K50K51  |  v[9]  |
    // Reg 2 [16:23]     |     K20K21 |     K52K53  |     K84K85 |    K116K117 |  v[10]    ||    Reg 2 [16:23]     |     K20K21 |     K52K53  |  v[10] |
    // Reg 2 [24:31]     |     K22K23 |     K54K55  |     K86K87 |    K118K119 |  v[11]    ||    Reg 2 [24:31]     |     K22K23 |     K54K55  |  v[11] |
    // Reg 3 [0:7]       |     K24K25 |     K56K57  |     K88K89 |    K120K121 |  v[12]    ||    Reg 3 [0:7]       |     K24K25 |     K56K57  |  v[12] |
    // Reg 3 [8:15]      |     K26K27 |     K58K59  |     K90K91 |    K122K123 |  v[13]    ||    Reg 3 [8:15]      |     K26K27 |     K58K59  |  v[13] |
    // Reg 3 [16:23]     |     K28K29 |     K60K61  |     K92K93 |    K124K125 |  v[14]    ||    Reg 3 [16:23]     |     K28K29 |     K60K61  |  v[14] |
    // Reg 3 [24:31]     |     K30K31 |     K62K63  |     K94K95 |    K126K127 |  v[15]    ||    Reg 3 [24:31]     |     K30K31 |     K62K63  |  v[15] |

    // Register Mapping for 128x16 for FP4:                                                                                            ||    Register Mapping for 64x32 for FP4:
    // Size              |   BLOCK_N  |          |   BLOCK_N   |          |   BLOCK_N  |          |   BLOCK_N   |          |           ||    Size              |   BLOCK_N  |          |   BLOCK_N   |          |        |
    // N                 | 0  ...  15 |          |  0  ...  15 |          | 0  ...  15 |          |  0  ...  15 |          | Vector    ||    N                 | 0  ...  31 |          |  0  ...  31 |          | Vector |
    // Thread Id         | 0  ...  15 |  Scale   | 16  ...  31 |  Scale   | 32  ... 47 |  Scale   | 48  ...  63 |  Scale   | Element   ||    Thread Id         | 0  ...  31 |  Scale   | 32  ...  63 |  Scale   | Element|
    // Register Element  |------------ ----------|------------- ----------|------------ ----------|------------- ----------|-----------||    Register Element  |------------|----------|-------------|----------|--------|
    // Reg 0 [0:7]       |     K0K1   |  x(0,N)  |     K32K33  |  x(M,1)  |     K64K65 |  x(M,2)  |    K96K97   |  x(M,3)  |  v[0]     ||    Reg 0 [0:7]       |     K0K1   |  x(M,0)  |     K32K33  |  x(M,1)  |  v[0]  |
    // Reg 0 [8:15]      |     K2K3   |  x(0,N)  |     K34K35  |  x(M,1)  |     K66K67 |  x(M,2)  |    K98K99   |  x(M,3)  |  v[1]     ||    Reg 0 [8:15]      |     K2K3   |  x(M,0)  |     K34K35  |  x(M,1)  |  v[1]  |
    // Reg 0 [16:23]     |     K4K5   |  x(0,N)  |     K36K37  |  x(M,1)  |     K68K69 |  x(M,2)  |    K100K101 |  x(M,3)  |  v[2]     ||    Reg 0 [16:23]     |     K4K5   |  x(M,0)  |     K36K37  |  x(M,1)  |  v[2]  |
    // Reg 0 [24:31]     |     K6K7   |  x(0,N)  |     K38K39  |  x(M,1)  |     K70K71 |  x(M,2)  |    K102K103 |  x(M,3)  |  v[3]     ||    Reg 0 [24:31]     |     K6K7   |  x(M,0)  |     K38K39  |  x(M,1)  |  v[3]  |
    // Reg 1 [0:7]       |     K8K9   |  x(0,N)  |     K40K41  |  x(M,1)  |     K72K73 |  x(M,2)  |    K104K105 |  x(M,3)  |  v[4]     ||    Reg 1 [0:7]       |     K8K9   |  x(M,0)  |     K40K41  |  x(M,1)  |  v[4]  |
    // Reg 1 [8:15]      |     K10K11 |  x(0,N)  |     K42K43  |  x(M,1)  |     K74K75 |  x(M,2)  |    K106K107 |  x(M,3)  |  v[5]     ||    Reg 1 [8:15]      |     K10K11 |  x(M,0)  |     K42K43  |  x(M,1)  |  v[5]  |
    // Reg 1 [16:23]     |     K12K13 |  x(0,N)  |     K44K45  |  x(M,1)  |     K76K77 |  x(M,2)  |    K108K109 |  x(M,3)  |  v[6]     ||    Reg 1 [16:23]     |     K12K13 |  x(M,0)  |     K44K45  |  x(M,1)  |  v[6]  |
    // Reg 1 [24:31]     |     K14K15 |  x(0,N)  |     K46K47  |  x(M,1)  |     K78K79 |  x(M,2)  |    K110K111 |  x(M,3)  |  v[7]     ||    Reg 1 [24:31]     |     K14K15 |  x(M,0)  |     K46K47  |  x(M,1)  |  v[7]  |
    // Reg 2 [0:7]       |     K16K17 |  x(0,N)  |     K48K49  |  x(M,1)  |     K80K81 |  x(M,2)  |    K112K113 |  x(M,3)  |  v[8]     ||    Reg 2 [0:7]       |     K16K17 |  x(M,0)  |     K48K49  |  x(M,1)  |  v[8]  |
    // Reg 2 [8:15]      |     K18K19 |  x(0,N)  |     K50K51  |  x(M,1)  |     K82K83 |  x(M,2)  |    K114K115 |  x(M,3)  |  v[9]     ||    Reg 2 [8:15]      |     K18K19 |  x(M,0)  |     K50K51  |  x(M,1)  |  v[9]  |
    // Reg 2 [16:23]     |     K20K21 |  x(0,N)  |     K52K53  |  x(M,1)  |     K84K85 |  x(M,2)  |    K116K117 |  x(M,3)  |  v[10]    ||    Reg 2 [16:23]     |     K20K21 |  x(M,0)  |     K52K53  |  x(M,1)  |  v[10] |
    // Reg 2 [24:31]     |     K22K23 |  x(0,N)  |     K54K55  |  x(M,1)  |     K86K87 |  x(M,2)  |    K118K119 |  x(M,3)  |  v[11]    ||    Reg 2 [24:31]     |     K22K23 |  x(M,0)  |     K54K55  |  x(M,1)  |  v[11] |
    // Reg 3 [0:7]       |     K24K25 |  x(0,N)  |     K56K57  |  x(M,1)  |     K88K89 |  x(M,2)  |    K120K121 |  x(M,3)  |  v[12]    ||    Reg 3 [0:7]       |     K24K25 |  x(M,0)  |     K56K57  |  x(M,1)  |  v[12] |
    // Reg 3 [8:15]      |     K26K27 |  x(0,N)  |     K58K59  |  x(M,1)  |     K90K91 |  x(M,2)  |    K122K123 |  x(M,3)  |  v[13]    ||    Reg 3 [8:15]      |     K26K27 |  x(M,0)  |     K58K59  |  x(M,1)  |  v[13] |
    // Reg 3 [16:23]     |     K28K29 |  x(0,N)  |     K60K61  |  x(M,1)  |     K92K93 |  x(M,2)  |    K124K125 |  x(M,3)  |  v[14]    ||    Reg 3 [16:23]     |     K28K29 |  x(M,0)  |     K60K61  |  x(M,1)  |  v[14] |
    // Reg 3 [24:31]     |     K30K31 |  x(0,N)  |     K62K63  |  x(M,1)  |     K94K95 |  x(M,2)  |    K126K127 |  x(M,3)  |  v[15]    ||    Reg 3 [24:31]     |     K30K31 |  x(M,0)  |     K62K63  |  x(M,1)  |  v[15] |
    // clang-format on

    // To start the loading process, let's visualize in 2D coords.
    // Each thread will load 1 element
    // We need to know where to start
    auto startCoord2D = std::make_pair((threadIdx.x / BLOCK_N), // Row
                                       threadIdx.x % BLOCK_N);  // Col

    // Flatten to 1D col_major offsets.
    auto col_major = [](auto const& coord, auto ld) { return coord.first + coord.second * ld; };

    auto startOffset = col_major(startCoord2D, BLOCK_K / BLOCK_X);

    // obtain 8-bit exponent
    fragX = scale_ptr[startOffset];

    return load_B_col_major<BType, BFragT, BLOCK_K, BLOCK_N>(input_ptr);
}

// Define a store function for C
// Size: (BLOCK_M x BLOCK_N)
// ASSUMPTION:
// - We want contiguous BLOCK_N sized row neighbors in register.
// - Data is in col_major format
// This means:
// - From C we will load BLOCK_M rows of size BLOCK_N to satisfy our input data
template <typename CType, typename CFragT, int32_t BLOCK_M, int32_t BLOCK_N>
struct store_C_col_major;

// Here we want to store a 16x16 block of data.
//
// Size              |   BLOCK_N  |   BLOCK_N   |   BLOCK_N   |   BLOCK_N   |
// N                 | 0  ...  15 |  0  ...  15 | 0  ...  15  |  0  ...  15 |
// Thread Id         | 0  ...  15 | 16  ...  31 | 32  ... 47  | 48  ...  63 | Vector
// Register Element   ------------ ------------- ------------ -------------- Element
// Reg0              |     M0     |     M4      |     M8      |     M12     | v[0]
// Reg1              |     M1     |     M5      |     M9      |     M13     | v[1]
// Reg2              |     M2     |     M6      |     M10     |     M14     | v[2]
// Reg3              |     M3     |     M7      |     M11     |     M15     | v[3]
template <typename CType, typename CFragT>
struct store_C_col_major<CType, CFragT, 16, 16>
{
    __device__ void operator()(CType* output, CFragT cFrag)
    {
        static constexpr uint32_t VW  = vectorSize(cFrag); // 4
        static constexpr uint32_t Dim = 16;

        // Each thread will load 4 elements.
        // We need to know where they start, and where the next elements are.
        auto startCoord2D = std::make_pair((threadIdx.x / Dim) * VW, // Row
                                           threadIdx.x % Dim);       // Col

        // Flatten to 1D col_major offsets.
        auto col_major = [](auto const& coord, auto ld) { return coord.first + coord.second * ld; };

        auto startOffset = col_major(startCoord2D, 16);

        auto* fragPtr = reinterpret_cast<CFragT*>(output + startOffset);
        *fragPtr      = cFrag;
    }
};

// Here we want to store a 32x32 block of data.
// Register Mapping:

// Size              |   BLOCK_N  |   BLOCK_N   |
// N                 | 0  ...  31 |  0  ...  31 |
// Thread Id         | 0  ...  31 | 32  ...  63 | Vector
// Register Element   ------------ -------------  Element
// Reg0              |     M0     |     M4      | v[0]
// Reg1              |     M1     |     M5      | v[1]
// Reg2              |     M2     |     M6      | v[2]
// Reg3              |     M3     |     M7      | v[3]
//                    ____________ _____________
// Reg4              |     M8     |     M12     | v[4]
// Reg5              |     M9     |     M13     | v[5]
// Reg6              |     M10    |     M14     | v[6]
// Reg7              |     M11    |     M15     | v[7]
//                    ____________ _____________
// Reg8              |     M16    |     M20     | v[8]
// Reg9              |     M17    |     M21     | v[9]
// Reg10             |     M18    |     M22     | v[10]
// Reg11             |     M19    |     M23     | v[11]
//                    ____________ _____________
// Reg12             |     M24    |     M28     | v[12]
// Reg13             |     M25    |     M29     | v[13]
// Reg14             |     M26    |     M30     | v[14]
// Reg15             |     M27    |     M31     | v[15]

template <typename CType, typename CFragT>
struct store_C_col_major<CType, CFragT, 32, 32>
{
    __device__ void operator()(CType* output, CFragT cFrag)
    {
        static constexpr uint32_t WAVE_SIZE      = 64;
        static constexpr uint32_t VW             = 4;
        static constexpr uint32_t Dim            = 32;
        static constexpr uint32_t M_PER_VW_CHUNK = VW * WAVE_SIZE / 32; // 8

        auto startCoord2D = std::make_pair((threadIdx.x / Dim) * VW, // Row
                                           threadIdx.x % Dim);       // Col

        // Major step between 'chunks'
        auto majorStepCoord2D = std::make_pair(M_PER_VW_CHUNK, 0);

        // Flatten to 1D col_major offsets.
        auto col_major = [](auto const& coord, auto ld) { return coord.first + coord.second * ld; };

        auto startOffset  = col_major(startCoord2D, 32);
        auto kMajorOffset = col_major(majorStepCoord2D, 32); // 8

        // we can vector store 4 contiguous elements at a time.
        using CRawT        = typename scalar_type<CFragT>::type;
        using CScalarFragT = typename vector_type<CRawT, VW>::type;
        union
        {
            CFragT frag;
            CScalarFragT chunks[vectorSize(CFragT{}) / VW];
        } fragC{cFrag}; // Initialize with input fragment

        CScalarFragT* fragPtr;
        for(uint32_t idx = 0; idx < vectorSize(CFragT{}) / VW; ++idx)
        {
            fragPtr  = reinterpret_cast<CScalarFragT*>(output + startOffset + idx * kMajorOffset);
            *fragPtr = fragC.chunks[idx];
        }
    }
};

// Define a store function for C
// Size: (BLOCK_M x BLOCK_N)
// ASSUMPTION:
// - We want contiguous BLOCK_N sized row neighbors in register.
// - Data is in row major format
template <typename CType, typename CFragT, int32_t BLOCK_M, int32_t BLOCK_N>
struct store_C_row_major;

// Here we want to store a 16x16 block of data.
//
// Size              |   BLOCK_N  |   BLOCK_N   |   BLOCK_N   |   BLOCK_N   |
// N                 | 0  ...  15 |  0  ...  15 | 0  ...  15  |  0  ...  15 |
// Thread Id         | 0  ...  15 | 16  ...  31 | 32  ... 47  | 48  ...  63 | Vector
// Register Element   ------------ ------------- ------------ -------------- Element
// Reg0              |     M0     |     M4      |     M8      |     M12     | v[0]
// Reg1              |     M1     |     M5      |     M9      |     M13     | v[1]
// Reg2              |     M2     |     M6      |     M10     |     M14     | v[2]
// Reg3              |     M3     |     M7      |     M11     |     M15     | v[3]
template <typename CType, typename CFragT>
struct store_C_row_major<CType, CFragT, 16, 16>
{
    __device__ void operator()(CType* output, CFragT cFrag)
    {
        static constexpr uint32_t VW  = vectorSize(cFrag); // 4
        static constexpr uint32_t Dim = 16;

        // Each thread will load 4 elements.
        // We need to know where they start, and where the next elements are.
        auto startCoord2D = std::make_pair((threadIdx.x / Dim) * VW, // Row
                                           threadIdx.x % Dim);       // Col
        auto stepCoord2D  = std::make_pair(1u, 0u);

        // Flatten to 1D row_major offsets.
        auto row_major = [](auto const& coord, auto ld) { return coord.first * ld + coord.second; };

        auto startOffset = row_major(startCoord2D, 16);
        auto kOffset     = row_major(stepCoord2D, 16);

        auto* fragPtr = reinterpret_cast<CFragT*>(output + startOffset);
        *fragPtr      = cFrag;

        // If you notice carefully, kOffset != 1.
        // This means the following is vector is updated with 4 non-contiguous offsets,
        // which the compiler will separate into 4 different global_store_dword instructions.
        output[startOffset]               = cFrag[0]; // v[0] = Reg 0
        output[startOffset + kOffset]     = cFrag[1]; // v[1] = Reg 1
        output[startOffset + 2 * kOffset] = cFrag[2]; // v[2] = Reg 2
        output[startOffset + 3 * kOffset] = cFrag[3]; // v[3] = Reg 3
    }
};

// Here we want to store a 32x32 block of data.
// Register Mapping:

// Size              |   BLOCK_N  |   BLOCK_N   |
// N                 | 0  ...  31 |  0  ...  31 |
// Thread Id         | 0  ...  31 | 32  ...  63 | Vector
// Register Element   ------------ -------------  Element
// Reg0              |     M0     |     M4      | v[0]
// Reg1              |     M1     |     M5      | v[1]
// Reg2              |     M2     |     M6      | v[2]
// Reg3              |     M3     |     M7      | v[3]
//                    ____________ _____________
// Reg4              |     M8     |     M12     | v[4]
// Reg5              |     M9     |     M13     | v[5]
// Reg6              |     M10    |     M14     | v[6]
// Reg7              |     M11    |     M15     | v[7]
//                    ____________ _____________
// Reg8              |     M16    |     M20     | v[8]
// Reg9              |     M17    |     M21     | v[9]
// Reg10             |     M18    |     M22     | v[10]
// Reg11             |     M19    |     M23     | v[11]
//                    ____________ _____________
// Reg12             |     M24    |     M28     | v[12]
// Reg13             |     M25    |     M29     | v[13]
// Reg14             |     M26    |     M30     | v[14]
// Reg15             |     M27    |     M31     | v[15]

template <typename CType, typename CFragT>
struct store_C_row_major<CType, CFragT, 32, 32>
{
    __device__ void operator()(CType* output, CFragT cFrag)
    {
        static constexpr uint32_t WAVE_SIZE      = 64;
        static constexpr uint32_t VW             = 4;                   // This VW is per 'chunk'
        static constexpr uint32_t Dim            = 32;                  // BLOCK_N
        static constexpr uint32_t M_PER_VW_CHUNK = VW * WAVE_SIZE / 32; // 8

        auto startCoord2D = std::make_pair((threadIdx.x / Dim) * VW, // Row
                                           threadIdx.x % Dim);       // Col

        // Minor step for each 'chunk'
        auto minorStepCoord2D = std::make_pair(1u, 0u);

        // Major step between 'chunks'
        auto majorStepCoord2D = std::make_pair(M_PER_VW_CHUNK, 0);

        // Flatten to 1D row_major offsets.
        auto row_major = [](auto const& coord, auto ld) { return coord.first * ld + coord.second; };

        auto startOffset  = row_major(startCoord2D, 32);
        auto kMinorOffset = row_major(minorStepCoord2D, 32);
        auto kMajorOffset = row_major(majorStepCoord2D, 32);

        output[startOffset]                                       = cFrag[0];  // v[0] = Reg 0
        output[startOffset + kMinorOffset]                        = cFrag[1];  // v[1] = Reg 1
        output[startOffset + 2 * kMinorOffset]                    = cFrag[2];  // v[2] = Reg 2
        output[startOffset + 3 * kMinorOffset]                    = cFrag[3];  // v[3] = Reg 3
        output[startOffset + kMajorOffset]                        = cFrag[4];  // v[4] = Reg 4
        output[startOffset + kMajorOffset + kMinorOffset]         = cFrag[5];  // v[5] = Reg 5
        output[startOffset + kMajorOffset + 2 * kMinorOffset]     = cFrag[6];  // v[6] = Reg 6
        output[startOffset + kMajorOffset + 3 * kMinorOffset]     = cFrag[7];  // v[7] = Reg 7
        output[startOffset + 2 * kMajorOffset]                    = cFrag[8];  // v[8] = Reg 8
        output[startOffset + 2 * kMajorOffset + kMinorOffset]     = cFrag[9];  // v[9] = Reg 9
        output[startOffset + 2 * kMajorOffset + 2 * kMinorOffset] = cFrag[10]; // v[10] = Reg 10
        output[startOffset + 2 * kMajorOffset + 3 * kMinorOffset] = cFrag[11]; // v[11] = Reg 11
        output[startOffset + 3 * kMajorOffset]                    = cFrag[12]; // v[12] = Reg 12
        output[startOffset + 3 * kMajorOffset + kMinorOffset]     = cFrag[13]; // v[13] = Reg 13
        output[startOffset + 3 * kMajorOffset + 2 * kMinorOffset] = cFrag[14]; // v[14] = Reg 14
        output[startOffset + 3 * kMajorOffset + 3 * kMinorOffset] = cFrag[15]; // v[15] = Reg 15
    }
};

template <typename AType,
          typename BType,
          typename CType,
          typename AccType,
          int32_t BLOCK_M,
          int32_t BLOCK_N,
          int32_t BLOCK_K,
          typename ALayout,
          typename BLayout,
          typename CLayout>
__global__ void matmul(const packed_type_t<AType>* a, const packed_type_t<BType>* b, CType* c)
{
    using PackedAType            = packed_type_t<AType>;
    constexpr auto packed_size_a = packed_size_v<PackedAType>;
    using PackedBType            = packed_type_t<BType>;
    constexpr auto packed_size_b = packed_size_v<PackedBType>;

    constexpr int WAVE_SIZE = 64;
    assert(threadIdx.x < WAVE_SIZE);
    assert(blockDim.x == 1 && blockDim.y == 1 && blockDim.z == 1);

    using AFragT =
        typename vector_type<PackedAType, BLOCK_M * BLOCK_K / WAVE_SIZE / packed_size_a>::type;
    using BFragT =
        typename vector_type<PackedBType, BLOCK_K * BLOCK_N / WAVE_SIZE / packed_size_b>::type;

    using CFragT        = typename vector_type<CType, BLOCK_M * BLOCK_N / WAVE_SIZE>::type;
    using AccumFragT    = vector_type<AccType, BLOCK_M * BLOCK_N / WAVE_SIZE>;
    using RawAccumFragT = typename vector_type<AccType, BLOCK_M * BLOCK_N / WAVE_SIZE>::type;

    // Create frags
    auto fragA   = AFragT{};
    auto fragB   = BFragT{};
    auto fragC   = CFragT{};
    auto fragAcc = AccumFragT{0};

    // Load the inputs.
    if constexpr(is_same_v<ALayout, tensor_layout::gemm::RowMajor>)
    {
        fragA = load_A_row_major<PackedAType, AFragT, BLOCK_M, BLOCK_K>(a);
    }
    else
    {
        fragA = load_A_col_major<PackedAType, AFragT, BLOCK_M, BLOCK_K>(a);
    }

    if constexpr(is_same_v<BLayout, tensor_layout::gemm::RowMajor>)
    {
        printf("This layout is not implemented\n");
    }
    else
    {
        fragB = load_B_col_major<PackedBType, BFragT, BLOCK_K, BLOCK_N>(b);
    }

    // Matrix multiply-accumulate using MFMA units
    // Accumulation intermediate = BLOCK_M x BLOCK_N
    using mfma = mfma_type_selector<BLOCK_M, BLOCK_N>;
    mfma::template run<>(fragA, fragB, fragAcc);

    for(int i = 0; i < vectorSize(fragC); ++i)
    {
        fragC[i] = type_convert<CType>(fragAcc.template AsType<RawAccumFragT>()[Number<0>{}][i]);
    }

    if constexpr(is_same_v<CLayout, tensor_layout::gemm::RowMajor>)
    {
        store_C_row_major<CType, CFragT, BLOCK_M, BLOCK_N>{}(c, fragC);
    }
    else
    {
        store_C_col_major<CType, CFragT, BLOCK_M, BLOCK_N>{}(c, fragC);
    }
}

template <typename AType,
          typename BType,
          typename ScaleType,
          typename CType,
          typename AccType,
          int32_t BLOCK_M,
          int32_t BLOCK_N,
          int32_t BLOCK_K,
          int32_t BLOCK_X,
          typename ALayout,
          typename BLayout,
          typename CLayout>
__global__ void matmul(const packed_type_t<AType>* a,
                       const ScaleType* xa,
                       const packed_type_t<BType>* b,
                       const ScaleType* xb,
                       CType* c)
{
    using PackedAType            = packed_type_t<AType>;
    constexpr auto packed_size_a = packed_size_v<PackedAType>;
    using PackedBType            = packed_type_t<BType>;
    constexpr auto packed_size_b = packed_size_v<PackedBType>;

    constexpr int WAVE_SIZE = 64;
    assert(threadIdx.x < WAVE_SIZE);
    assert(blockDim.x == 1 && blockDim.y == 1 && blockDim.z == 1);

    using AFragT =
        typename vector_type<PackedAType, BLOCK_M * BLOCK_K / WAVE_SIZE / packed_size_a>::type;
    using BFragT =
        typename vector_type<PackedBType, BLOCK_K * BLOCK_N / WAVE_SIZE / packed_size_b>::type;

    using CFragT        = typename vector_type<CType, BLOCK_M * BLOCK_N / WAVE_SIZE>::type;
    using AccumFragT    = vector_type<AccType, BLOCK_M * BLOCK_N / WAVE_SIZE>;
    using RawAccumFragT = typename vector_type<AccType, BLOCK_M * BLOCK_N / WAVE_SIZE>::type;
    using AScaleFragT   = typename vector_type<ScaleType, 1>::type;
    using BScaleFragT   = typename vector_type<ScaleType, 1>::type;

    // Create frags
    auto fragA   = AFragT{};
    auto fragB   = BFragT{};
    auto fragC   = CFragT{};
    auto fragAcc = AccumFragT{0};
    auto fragXa  = AScaleFragT{};
    auto fragXb  = BScaleFragT{};

    // Load the inputs.
    if constexpr(is_same_v<ALayout, tensor_layout::gemm::RowMajor>)
    {
        fragA = load_mx_A_row_major<PackedAType,
                                    AFragT,
                                    ScaleType,
                                    AScaleFragT,
                                    BLOCK_M,
                                    BLOCK_K,
                                    BLOCK_X>(a, xa, fragXa);
    }
    else
    {
        printf("This layout is not implemented\n");
    }

    if constexpr(is_same_v<BLayout, tensor_layout::gemm::RowMajor>)
    {
        printf("This layout is not implemented\n");
    }
    else
    {
        fragB = load_mx_B_col_major<PackedBType,
                                    BFragT,
                                    ScaleType,
                                    BScaleFragT,
                                    BLOCK_K,
                                    BLOCK_N,
                                    BLOCK_X>(b, xb, fragXb);
    }

    // Scaled Matrix multiply-accumulate using MFMA units
    // Accumulation intermediate = BLOCK_M x BLOCK_N
    using mfma = mfma_scale_type_selector<BLOCK_M, BLOCK_N>;
    mfma::template run<>(fragA,
                         fragXa.template AsType<ScaleType>(),
                         fragB,
                         fragXb.template AsType<ScaleType>(),
                         fragAcc);

    for(int i = 0; i < vectorSize(fragC); ++i)
    {
        fragC[i] = type_convert<CType>(fragAcc.template AsType<RawAccumFragT>()[Number<0>{}][i]);
    }

    if constexpr(is_same_v<CLayout, tensor_layout::gemm::RowMajor>)
    {
        store_C_row_major<CType, CFragT, BLOCK_M, BLOCK_N>{}(c, fragC);
    }
    else
    {
        store_C_col_major<CType, CFragT, BLOCK_M, BLOCK_N>{}(c, fragC);
    }
}

/**
 * @brief Structure to hold dimension parameters for GEMM tensors.
 *
 * M Number of rows in matrix A and matrix C.
 * N Number of columns in matrix B and matrix C.
 * K Number of columns in matrix A and number of rows in matrix B.
 * StrideA Stride (leading dimension) of matrix A.
 * StrideB Stride (leading dimension) of matrix B.
 * StrideC Stride (leading dimension) of matrix C.
 */
struct GemmParams
{
    ck::index_t M = 16;
    ck::index_t N = 16;
    ck::index_t K = 128;

    ck::index_t StrideA = -1;
    ck::index_t StrideB = -1;
    ck::index_t StrideC = -1;
};

namespace mxmfma_test {
template <typename ADataType, typename BDataType, typename ScaleType, typename CDataType>
void RunHostGEMM(const Tensor<ADataType>& A,
                 const Tensor<ScaleType>& a_scales,
                 const Tensor<BDataType>& B,
                 const Tensor<ScaleType>& b_scales,
                 Tensor<CDataType>& C)
{
    using PassThrough = ck::tensor_operation::element_wise::PassThrough;

    using ReferenceGemmInstance = ck::tensor_operation::host::ReferenceMXGemm<ADataType,
                                                                              BDataType,
                                                                              CDataType,
                                                                              float,
                                                                              ScaleType,
                                                                              PassThrough,
                                                                              PassThrough,
                                                                              PassThrough,
                                                                              float,
                                                                              float>;
    auto ref_gemm               = ReferenceGemmInstance{};
    auto ref_invoker            = ref_gemm.MakeInvoker();

    auto ref_argument = ref_gemm.MakeArgument(
        A, a_scales, B, b_scales, C, PassThrough{}, PassThrough{}, PassThrough{});

    ref_invoker.Run(ref_argument);
}

template <typename KernelType,
          typename ADataType,
          typename BDataType,
          typename ScaleType,
          typename CDataType>
bool RunDeviceGEMM(KernelType kernel,
                   const Tensor<ADataType>& A,
                   const Tensor<ScaleType>& a_scales,
                   const Tensor<BDataType>& B,
                   const Tensor<ScaleType>& b_scales,
                   Tensor<CDataType>& C)
{
    DeviceMem a_m_k_device_buf(sizeof(ADataType) * A.mDesc.GetElementSpaceSize());
    DeviceMem a_scales_device_buf(sizeof(ScaleType) * a_scales.mDesc.GetElementSpaceSize());
    DeviceMem b_n_k_device_buf(sizeof(BDataType) * B.mDesc.GetElementSpaceSize());
    DeviceMem b_scales_device_buf(sizeof(ScaleType) * b_scales.mDesc.GetElementSpaceSize());
    DeviceMem c_m_n_device_buf(sizeof(CDataType) * C.mDesc.GetElementSpaceSize());

    a_m_k_device_buf.ToDevice(A.mData.data());
    a_scales_device_buf.ToDevice(a_scales.mData.data());
    b_n_k_device_buf.ToDevice(B.mData.data());
    b_scales_device_buf.ToDevice(b_scales.mData.data());
    kernel<<<1, 64>>>(static_cast<const ADataType*>(a_m_k_device_buf.GetDeviceBuffer()),
                      static_cast<const ScaleType*>(a_scales_device_buf.GetDeviceBuffer()),
                      static_cast<const BDataType*>(b_n_k_device_buf.GetDeviceBuffer()),
                      static_cast<const ScaleType*>(b_scales_device_buf.GetDeviceBuffer()),
                      static_cast<CDataType*>(c_m_n_device_buf.GetDeviceBuffer()));
    c_m_n_device_buf.FromDevice(C.mData.data());

    return true;
}

template <typename DeviceMFMA,
          typename ADataType,
          typename BDataType,
          typename ScaleType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          index_t BLOCK_M,
          index_t BLOCK_N,
          index_t BLOCK_K,
          index_t BLOCK_X>
struct TestMXMFMA
{
    using PackedAType                   = packed_type_t<ADataType>;
    static constexpr auto packed_size_a = packed_size_v<PackedAType>;
    using PackedBType                   = packed_type_t<BDataType>;
    static constexpr auto packed_size_b = packed_size_v<PackedBType>;

    auto PrepareGemmTensors(const GemmParams& params, index_t init)
    {
        auto f_host_tensor_descriptor =
            [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
                if(std::is_same<decltype(layout), ck::tensor_layout::gemm::RowMajor>::value)
                {
                    return HostTensorDescriptor(std::vector<std::size_t>({row, col}),
                                                std::vector<std::size_t>({stride, 1}));
                }
                else
                {
                    return HostTensorDescriptor(std::vector<std::size_t>({row, col}),
                                                std::vector<std::size_t>({1, stride}));
                }
            };

        Tensor<PackedAType> a_m_k(
            f_host_tensor_descriptor(params.M, params.K, params.StrideA, ALayout{}));
        Tensor<ScaleType> a_scales(
            f_host_tensor_descriptor(params.M, params.K / BLOCK_X, params.K / BLOCK_X, ALayout{}));
        Tensor<PackedBType> b_n_k(
            f_host_tensor_descriptor(params.K, params.N, params.StrideB, BLayout{}));
        Tensor<ScaleType> b_scales(
            f_host_tensor_descriptor(params.K / BLOCK_X, params.N, params.K / BLOCK_X, BLayout{}));
        Tensor<CDataType> c_m_n_host_result(
            f_host_tensor_descriptor(params.M, params.N, params.StrideC, CLayout{}));
        Tensor<CDataType> c_m_n_device_result(
            f_host_tensor_descriptor(params.M, params.N, params.StrideC, CLayout{}));

        switch(init)
        {
        case 0:
            a_m_k.GenerateTensorValue(GeneratorTensor_1<PackedAType>{1.0f});
            a_scales.GenerateTensorValue(GeneratorTensor_1<ScaleType>{0.5f});
            // NOTE: not all numbers are representable in FP8, BF8, etc.
            // 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 16 18 20 20 20 22 24 24 24 26 28 28 28 30 32
            b_n_k.GenerateTensorValue(GeneratorTensor_Sequential<PackedBType, 1>{});
            b_scales.GenerateTensorValue(GeneratorTensor_1<ScaleType>{1.0f});
            break;
        case 1:
            // results in C = {K}
            a_m_k.GenerateTensorValue(GeneratorTensor_1<PackedAType>{1.0f});
            a_scales.GenerateTensorValue(GeneratorTensor_1<ScaleType>{512.0f});
            b_n_k.GenerateTensorValue(GeneratorTensor_1<PackedBType>{1.0f});
            b_scales.GenerateTensorValue(GeneratorTensor_1<ScaleType>{1.0f / 512});
            break;
        case 2:
            // expect small round off errors
            a_m_k.GenerateTensorValue(GeneratorTensor_3<PackedAType>{-2.0, 2.0});
            a_scales.GenerateTensorValue(
                GeneratorTensor_2<ScaleType>{126, 129}); // scales: {0.5, 1, 2}
            b_n_k.GenerateTensorValue(GeneratorTensor_3<PackedBType>{-2.0, 2.0});
            b_scales.GenerateTensorValue(GeneratorTensor_2<ScaleType>{126, 129});
            break;
        case 3:
            // expect small round off errors
            a_m_k.GenerateTensorValue(GeneratorTensor_4<PackedAType>(0, 1, time(nullptr)));
            a_scales.GenerateTensorValue(
                GeneratorTensor_2<ScaleType>{126, 129}); // scales: {0.5, 1, 2}
            b_n_k.GenerateTensorValue(GeneratorTensor_4<PackedBType>(0, 1, time(nullptr) / 2));
            b_scales.GenerateTensorValue(
                GeneratorTensor_2<ScaleType>{126, 129}); //  scales: {0.5, 1, 2}
            break;

        default:
            // all initial values are representable in FP8, BF8
            a_m_k.GenerateTensorValue(GeneratorTensor_2<PackedAType>{-6, 7}); // Z[-6,6]
            a_scales.GenerateTensorValue(
                GeneratorTensor_2<ScaleType>{122, 129}); // scales: [1/32,..., 2]
            b_n_k.GenerateTensorValue(GeneratorTensor_2<PackedBType>{-6, 7}); // Z[-6,6]
            b_scales.GenerateTensorValue(
                GeneratorTensor_2<ScaleType>{122, 129}); //  scales: [1/32,..., 2]

            break;
        }

        return std::make_tuple(
            a_m_k, a_scales, b_n_k, b_scales, c_m_n_host_result, c_m_n_device_result);
    }

    auto operator()(const DeviceMFMA& mfma_kernel, index_t init)
    {
        // Arrange
        GemmParams params;
        params.M = BLOCK_M;
        params.N = BLOCK_N;
        params.K = BLOCK_K;

        auto f_get_default_stride = [](std::size_t row,
                                       std::size_t col,
                                       ck::index_t stride,
                                       auto layout) {
            if(stride == -1)
            {
                // give a chance if stride is -1, return a default packed stride
                if constexpr(std::is_same_v<decltype(layout), ck::tensor_layout::gemm::RowMajor>)
                {
                    return static_cast<std::size_t>(col);
                }
                else
                {
                    return static_cast<std::size_t>(row);
                }
            }
            else
                return static_cast<std::size_t>(stride);
        };

        params.StrideA = f_get_default_stride(BLOCK_M, BLOCK_K, params.StrideA, ALayout{});
        params.StrideB = f_get_default_stride(BLOCK_K, BLOCK_N, params.StrideB, BLayout{});
        params.StrideC = f_get_default_stride(BLOCK_M, BLOCK_N, params.StrideC, CLayout{});

        auto host_tensors = PrepareGemmTensors(params, init);

        const Tensor<PackedAType>& a      = std::get<0>(host_tensors);
        const Tensor<ScaleType>& a_scales = std::get<1>(host_tensors);
        const Tensor<PackedBType>& b      = std::get<2>(host_tensors);
        const Tensor<ScaleType>& b_scales = std::get<3>(host_tensors);
        Tensor<CDataType>& c_host         = std::get<4>(host_tensors);
        Tensor<CDataType>& c_device       = std::get<5>(host_tensors);

        RunHostGEMM(a, a_scales, b, b_scales, c_host);

        RunDeviceGEMM(mfma_kernel, a, a_scales, b, b_scales, c_device);

        bool res = false;
        if constexpr(std::is_same<CDataType, float>::value ||
                     std::is_same<CDataType, half_t>::value)
        {
            res = ck::utils::check_err(c_device.mData, c_host.mData);
        }
        else
        {
            std::cout << "UNSUPPORTED CDataType" << std::endl;
        }

        return res;
    }
};

} // namespace mxmfma_test

namespace mfma_test {
template <typename GemmInstance,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation>
void RunHostGEMM(const Tensor<ADataType>& A,
                 const Tensor<BDataType>& B,
                 Tensor<CDataType>& C,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 CElementwiseOperation c_element_op)
{
    auto ref_gemm     = GemmInstance{};
    auto ref_invoker  = ref_gemm.MakeInvoker();
    auto ref_argument = ref_gemm.MakeArgument(A, B, C, a_element_op, b_element_op, c_element_op);

    ref_invoker.Run(ref_argument);
}

template <typename KernelType, typename ADataType, typename BDataType, typename CDataType>
bool RunDeviceGEMM(KernelType kernel,
                   const Tensor<ADataType>& A,
                   const Tensor<BDataType>& B,
                   Tensor<CDataType>& C)
{
    DeviceMem a_m_k_device_buf(sizeof(ADataType) * A.mDesc.GetElementSpaceSize());
    DeviceMem b_n_k_device_buf(sizeof(BDataType) * B.mDesc.GetElementSpaceSize());
    DeviceMem c_m_n_device_buf(sizeof(CDataType) * C.mDesc.GetElementSpaceSize());

    a_m_k_device_buf.ToDevice(A.mData.data());
    b_n_k_device_buf.ToDevice(B.mData.data());
    kernel<<<1, 64>>>(static_cast<const ADataType*>(a_m_k_device_buf.GetDeviceBuffer()),
                      static_cast<const BDataType*>(b_n_k_device_buf.GetDeviceBuffer()),
                      static_cast<CDataType*>(c_m_n_device_buf.GetDeviceBuffer()));
    c_m_n_device_buf.FromDevice(C.mData.data());

    return true;
}

template <typename DeviceMFMA,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename GPUAccDataType,
          typename CPUAccDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          index_t BLOCK_M,
          index_t BLOCK_N,
          index_t BLOCK_K>
struct TestMFMA
{
    using PackedAType                   = packed_type_t<ADataType>;
    static constexpr auto packed_size_a = packed_size_v<PackedAType>;
    using PackedBType                   = packed_type_t<BDataType>;
    static constexpr auto packed_size_b = packed_size_v<PackedBType>;

    auto PrepareGemmTensors(const GemmParams& params, index_t init)
    {
        auto f_host_tensor_descriptor =
            [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
                if(std::is_same<decltype(layout), ck::tensor_layout::gemm::RowMajor>::value)
                {
                    return HostTensorDescriptor(std::vector<std::size_t>({row, col}),
                                                std::vector<std::size_t>({stride, 1}));
                }
                else
                {
                    return HostTensorDescriptor(std::vector<std::size_t>({row, col}),
                                                std::vector<std::size_t>({1, stride}));
                }
            };

        Tensor<PackedAType> a_m_k(
            f_host_tensor_descriptor(params.M, params.K, params.StrideA, ALayout{}));
        Tensor<PackedBType> b_n_k(
            f_host_tensor_descriptor(params.K, params.N, params.StrideB, BLayout{}));
        Tensor<CDataType> c_m_n_host_result(
            f_host_tensor_descriptor(params.M, params.N, params.StrideC, CLayout{}));
        Tensor<CDataType> c_m_n_device_result(
            f_host_tensor_descriptor(params.M, params.N, params.StrideC, CLayout{}));

        switch(init)
        {
        case 0:
            a_m_k.GenerateTensorValue(GeneratorTensor_1<PackedAType>{0.625f});
            // NOTE: not all numbers are representable in FP8, BF8, etc.
            b_n_k.GenerateTensorValue(GeneratorTensor_Sequential<PackedBType, 1>{});
            break;
        case 1:
            // results in C = {K}
            a_m_k.GenerateTensorValue(GeneratorTensor_1<PackedAType>{1.0f});
            b_n_k.GenerateTensorValue(GeneratorTensor_1<PackedBType>{1.0f});
            break;
        case 2:
            // expect small round off errors that lead to FP8MFMA32x32x64 failures
            a_m_k.GenerateTensorValue(GeneratorTensor_3<PackedAType>{-5, 5});
            b_n_k.GenerateTensorValue(GeneratorTensor_3<PackedBType>{-5, 5});
            break;
        case 3:
            // expect small round off errors that lead to FP8MFMA32x32x64 failures
            a_m_k.GenerateTensorValue(GeneratorTensor_4<PackedAType>(-1, 3));
            b_n_k.GenerateTensorValue(GeneratorTensor_4<PackedBType>(1, 3));
            break;

        default:
            // all initial values are representable in FP8/6, BF8/6 FP4 is missing 5
            a_m_k.GenerateTensorValue(GeneratorTensor_2<PackedAType>{-6, 7}); // Z[-6,6]
            b_n_k.GenerateTensorValue(GeneratorTensor_2<PackedBType>{-6, 7});

            break;
        }

        return std::make_tuple(a_m_k, b_n_k, c_m_n_host_result, c_m_n_device_result);
    }

    auto operator()(const DeviceMFMA& mfma_kernel, index_t init)
    {
        // Arrange
        GemmParams params;
        params.M = BLOCK_M;
        params.N = BLOCK_N;
        params.K = BLOCK_K;

        auto f_get_default_stride = [](std::size_t row,
                                       std::size_t col,
                                       ck::index_t stride,
                                       auto layout) {
            if(stride == -1)
            {
                // give a chance if stride is -1, return a default packed stride
                if constexpr(std::is_same_v<decltype(layout), ck::tensor_layout::gemm::RowMajor>)
                {
                    return static_cast<std::size_t>(col);
                }
                else
                {
                    return static_cast<std::size_t>(row);
                }
            }
            else
                return static_cast<std::size_t>(stride);
        };

        params.StrideA = f_get_default_stride(BLOCK_M, BLOCK_K, params.StrideA, ALayout{});
        params.StrideB = f_get_default_stride(BLOCK_K, BLOCK_N, params.StrideB, BLayout{});
        params.StrideC = f_get_default_stride(BLOCK_M, BLOCK_N, params.StrideC, CLayout{});

        auto host_tensors = PrepareGemmTensors(params, init);

        const Tensor<PackedAType>& a = std::get<0>(host_tensors);
        const Tensor<PackedBType>& b = std::get<1>(host_tensors);
        Tensor<CDataType>& c_host    = std::get<2>(host_tensors);
        Tensor<CDataType>& c_device  = std::get<3>(host_tensors);

        using PassThrough = ck::tensor_operation::element_wise::PassThrough;

        auto a_element_op = PassThrough{};
        auto b_element_op = PassThrough{};
        auto c_element_op = PassThrough{};

        using ReferenceGemmInstance = ck::tensor_operation::host::ReferenceGemm<PackedAType,
                                                                                PackedBType,
                                                                                CDataType,
                                                                                CPUAccDataType,
                                                                                PassThrough,
                                                                                PassThrough,
                                                                                PassThrough,
                                                                                CPUAccDataType>;

        RunHostGEMM<ReferenceGemmInstance>(a, b, c_host, a_element_op, b_element_op, c_element_op);

        RunDeviceGEMM(mfma_kernel, a, b, c_device);

        bool res = false;
        if constexpr(std::is_same<CDataType, float>::value ||
                     std::is_same<CDataType, half_t>::value)
        {
            res = ck::utils::check_err(c_device.mData, c_host.mData);
        }
        else
        {
            std::cout << "UNSUPPORTED CDataType" << std::endl;
        }

        return res;
    }
};

} // namespace mfma_test
} // namespace ck
