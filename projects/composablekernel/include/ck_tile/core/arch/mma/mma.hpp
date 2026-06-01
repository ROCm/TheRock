// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once
#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/numeric/vector_type.hpp"

#include "amdgcn_mma.hpp"
#include "mma_selector.hpp"
#include "mma_traits.hpp"
#include "mma_transforms.hpp"

#include "mfma/mfma.hpp"
#include "wmma/wmma.hpp"

namespace ck_tile::core::arch::mma {

/*! @enum MmaAccumPolicy
 * @brief Accumulation order for Mma decomposition
 */
enum struct MmaAccumPolicy
{
    // Decomposition and accumulation in row-major block order
    ROW_MAJOR,
    // Decomposition and accumulation in col-major block order
    COL_MAJOR
};

/**
 * @class Mma
 * @brief Driver for the wave-tile Mma operation. Given a backend block-wise MmaOp implementation
 * (e.g., mfma or wmma), this class performs block-wise decomposition to matrix-multiply input
 * fragments of (A: FragM x FragK) x (B: FragK x FragN) and accumulates results into output fragment
 * (C: FragM x FragN).
 * @tparam ADataType Data type of input fragment A
 * @tparam BDataType Data type of input fragment B
 * @tparam CDataType Data type of input/output fragment C (accumulator)
 * @tparam FragM Mma fragment M dimension
 * @tparam FragN Mma fragment K dimension
 * @tparam FragK Mma fragment M dimension
 * @tparam AccumPolicy The block order of the accumulation registers (row major or col major block
 * order)
 * @tparam CompilerTarget The compiler target
 * @tparam MmaOp The backend wrapper class that will perform block-wise mma op (e.g., mfma or
 * wmma)
 * @tparam MmaTransforms The set of transforms to be applied to input/output fragments
 * @par This is an example of an Mma decomposition driver class that can be used in a wave-tile
 * context. Given a fragment size, we can decompose the fragment into smaller block-wise mma ops
 * that are natively supported by the hardware (e.g., mfma or wmma). The class also supports
 * applying transforms to the input/output fragments as needed (e.g., layout conversions, data type
 * conversions, etc.). We may also specify the accumulation order (row-major or col-major) for the
 * output fragment. This is a powerful example of how to build a flexible and reusable mma driver
 * that can adapt to different hardware capabilities and requirements.
 */
template <typename ADataType,
          typename BDataType,
          typename CDataType,
          uint32_t FragM,
          uint32_t FragN,
          uint32_t FragK,
          MmaAccumPolicy AccumPolicy = MmaAccumPolicy::ROW_MAJOR,
          typename CompilerTarget =
              decltype(get_compiler_target()), // TODO: c++20 amdgcn_target_arch_id GfxTargetId =
                                               // get_compiler_target(),
          typename MmaOp =
              typename MmaDefaultSelector<ADataType, // TODO: c++20 MmaOpI MmaOp = typename
                                                     // MmaDefaultSelector<ADataType,
                                          BDataType,
                                          CDataType,
                                          FragM,
                                          FragN,
                                          FragK,
                                          CompilerTarget>::SelectedOp,
          typename MmaTransforms = // TODO: c++20 MmaTransformsI MmaTransforms =
          typename MmaTransformsDefaultSelector<MmaOp, CompilerTarget>::SelectedTransforms>
struct WaveWiseMma
{

    using BlockWiseMmaOp       = MmaOp;
    using BlockWiseMmaOpTraits = MmaOpTraits<BlockWiseMmaOp>;

    // Block dimensions
    constexpr static uint32_t BlockM = BlockWiseMmaOpTraits::BlockM;
    constexpr static uint32_t BlockN = BlockWiseMmaOpTraits::BlockN;
    constexpr static uint32_t BlockK = BlockWiseMmaOpTraits::BlockK;

    // Block counts for decomposition
    constexpr static uint32_t BlocksM = FragM / BlockM;
    constexpr static uint32_t BlocksN = FragN / BlockN;
    constexpr static uint32_t BlocksK = FragK / BlockK;
    constexpr static uint32_t BlocksC = BlocksM * BlocksN;

    // Vector types for packed registers in each block
    using AVecType = typename BlockWiseMmaOpTraits::AVecType;
    using BVecType = typename BlockWiseMmaOpTraits::BVecType;
    using CVecType = typename BlockWiseMmaOpTraits::CVecType;

    // Buffer types for fragments
    using ABufferType = AVecType[BlocksM][BlocksK];
    using BBufferType = BVecType[BlocksN][BlocksK];
    using CBufferType = CVecType[BlocksM][BlocksN];

    // Transforms
    using ATransform = typename MmaTransforms::ATransform;
    using BTransform = typename MmaTransforms::BTransform;
    using CTransform = typename MmaTransforms::CTransform;
    using DTransform = typename MmaTransforms::DTransform;

    // Sanity checks
    static_assert(FragM >= BlockM, "FragM must be larger than BlockM");
    static_assert(FragN >= BlockN, "FragN must be larger than BlockN");
    static_assert(FragK >= BlockK, "FragK must be larger than BlockK");
    static_assert(FragM % BlockM == 0u, "FragM must be a multiple of BlockM");
    static_assert(FragN % BlockN == 0u, "FragN must be a multiple of BlockN");
    static_assert(FragK % BlockK == 0u, "FragK must be a multiple of BlockK");

    private:
    template <typename DstT, typename SrcT>
    CK_TILE_DEVICE static auto formatBuffer(SrcT const& inputBuffer)
    {
        // TODO: Implement formatting logic as needed.
        // This is intended to convert input fragments to the native vector types
        // required by the BlockWiseMma operation for iteration
        static_assert(sizeof(DstT) == sizeof(SrcT), "Size mismatch in formatBuffer");
        return reinterpret_cast<DstT const&>(inputBuffer);
    }

    template <typename DstT, typename SrcT>
    CK_TILE_DEVICE static auto formatBuffer(SrcT& inputBuffer)
    {
        // TODO: Implement formatting logic as needed.
        // This is intended to convert input fragments to the native vector types
        // required by the BlockWiseMma operation for iteration
        static_assert(sizeof(DstT) == sizeof(SrcT), "Size mismatch in formatBuffer");
        return reinterpret_cast<DstT&>(inputBuffer);
    }

    /*! @brief Execute Mma in row-major accumulation order.
     * @tparam VecTA The input fragment A vector type
     * @tparam VecTB The input fragment B vector type
     * @tparam VecTC The input/output fragment C vector type
     */
    template <typename VecTA, typename VecTB, typename VecTC>
    CK_TILE_DEVICE static decltype(auto) exec_col_major(VecTA&& a, VecTB&& b, VecTC&& accum)
    {
        // We implement an example wave-tile pipeline here.
        // First, we apply the necessary transforms to the input fragments,
        // then we convert the result into buffers of native vector formats
        // that we can easily index. Native vector formats are necessary inputs
        // to the given MmaOp exec function.
        auto a_frag = formatBuffer<ABufferType>(ATransform::exec(a));
        auto b_frag = formatBuffer<BBufferType>(BTransform::exec(b));
        auto c_frag = formatBuffer<CBufferType>(CTransform::exec(accum));

        // "Col-major" accumulation over the M-dimension blocks first.
        // Pseudo code here, but we would basically iterate over the blocks in col-major order
        for(uint32_t bn = 0u; bn < BlocksN; ++bn)
        {
            for(uint32_t bm = 0u; bm < BlocksM; ++bm)
            {
                for(uint32_t bk = 0u; bk < BlocksK; ++bk)
                {
                    c_frag[bm][bn] =
                        BlockWiseMmaOp::exec(a_frag[bm][bk], b_frag[bn][bk], c_frag[bm][bn]);
                }
            }
        }

        // Convert native vector results back to the output fragment format
        // and then return after we apply the final output transform.
        return DTransform::exec(formatBuffer<std::decay_t<VecTC>>(c_frag));
    }

    /*! @brief Execute Mma in row-major accumulation order.
     * @tparam VecTA The input fragment A vector type
     * @tparam VecTB The input fragment B vector type
     * @tparam VecTC The input/output fragment C vector type
     */
    template <typename VecTA, typename VecTB, typename VecTC>
    CK_TILE_DEVICE static decltype(auto) exec_row_major(VecTA&& a, VecTB&& b, VecTC&& accum)
    {
        // We implement an example wave-tile pipeline here.
        // First, we apply the necessary transforms to the input fragments,
        // then we convert the result into buffers of native vector formats
        // that we can easily index. Native vector formats are necessary inputs
        // to the given MmaOp exec function.
        auto a_frag = formatBuffer<ABufferType>(ATransform::exec(a));
        auto b_frag = formatBuffer<BBufferType>(BTransform::exec(b));
        auto c_frag = formatBuffer<CBufferType>(CTransform::exec(accum));

        // "Row-major" accumulation over the N-dimension blocks first.
        // Pseudo code here, but we would basically iterate over the blocks in row-major order.
        // We also have to ensure that the incoming vector fragments are converted to native vector
        // types before passing to the BlockWiseMma exec function.
        for(uint32_t bm = 0u; bm < BlocksM; ++bm)
        {
            for(uint32_t bn = 0u; bn < BlocksN; ++bn)
            {
                for(uint32_t bk = 0u; bk < BlocksK; ++bk)
                {
                    c_frag[bm][bn] =
                        BlockWiseMmaOp::exec(a_frag[bm][bk], b_frag[bn][bk], c_frag[bm][bn]);
                }
            }
        }

        // Convert native vector results back to the output fragment format
        // and then return after we apply the final output transform.
        return DTransform::exec(formatBuffer<std::decay_t<VecTC>>(c_frag));
    }

    public:
    /*! @brief Forward to Mma operation with specified accumulation order.
     * @tparam VecTA The input fragment A vector type
     * @tparam VecTB The input fragment B vector type
     * @tparam VecTC The input/output fragment C vector type
     */
    template <typename VecTA, typename VecTB, typename VecTC>
    CK_TILE_DEVICE static decltype(auto) exec(VecTA&& a, VecTB&& b, VecTC&& accum)
    {
        if constexpr(AccumPolicy == MmaAccumPolicy::ROW_MAJOR)
        {
            return exec_row_major(
                std::forward<VecTA>(a), std::forward<VecTB>(b), std::forward<VecTC>(accum));
        }
        else // if constexpr(AccumPolicy == MmaAccumPolicy::COL_MAJOR)
        {
            return exec_col_major(
                std::forward<VecTA>(a), std::forward<VecTB>(b), std::forward<VecTC>(accum));
        }
    }
};

} // namespace ck_tile::core::arch::mma
