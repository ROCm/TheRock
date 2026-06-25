// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"
#include "ck_tile/host/concat.hpp"

namespace ck_tile {

template <typename AsDataType_,
          typename BsDataType_,
          typename EDataType_,
          typename BlockGemmShape_,
          typename Traits_,
          typename ComputeDataType_ = AsDataType_,
          typename AElementWise_    = ck_tile::element_wise::PassThrough,
          typename BElementWise_    = ck_tile::element_wise::PassThrough,
          bool FixedVectorSize_     = false,
          index_t VectorSizeA_      = 1,
          index_t VectorSizeB_      = 1>
struct GemmPipelineProblemBase
{
    using Traits = remove_cvref_t<Traits_>;

    using AsDataType = remove_cvref_t<AsDataType_>;
    using BsDataType = remove_cvref_t<BsDataType_>;
    using CDataType  = remove_cvref_t<EDataType_>; // actually AccDataType

    static constexpr bool FixedVectorSize = FixedVectorSize_;

    using BlockGemmShape = remove_cvref_t<BlockGemmShape_>;

    using AElementWise = remove_cvref_t<AElementWise_>;
    using BElementWise = remove_cvref_t<BElementWise_>;

    using AsLayout = remove_cvref_t<typename Traits::AsLayout>;
    using BsLayout = remove_cvref_t<typename Traits::BsLayout>;
    using CLayout  = remove_cvref_t<typename Traits::CLayout>;

    static constexpr bool ComputeDataTypeIsTuple = is_detected<is_tuple, ComputeDataType_>::value;
    static constexpr bool ADataTypeIsTuple       = is_detected<is_tuple, AsDataType>::value;
    static constexpr bool BDataTypeIsTuple       = is_detected<is_tuple, BsDataType>::value;

    static constexpr bool ALayoutIsTuple = is_detected<is_tuple, AsLayout>::value;
    static constexpr bool BLayoutIsTuple = is_detected<is_tuple, BsLayout>::value;

    using ComputeDataTypeTuple = std::conditional_t<ComputeDataTypeIsTuple,
                                                    remove_cvref_t<ComputeDataType_>,
                                                    remove_cvref_t<tuple<ComputeDataType_>>>;
    using AsLayoutTuple        = std::
        conditional_t<ALayoutIsTuple, remove_cvref_t<AsLayout>, remove_cvref_t<tuple<AsLayout>>>;
    using BsLayoutTuple = std::
        conditional_t<BLayoutIsTuple, remove_cvref_t<BsLayout>, remove_cvref_t<tuple<BsLayout>>>;

    using AsDataTypeTuple = std::conditional_t<ADataTypeIsTuple,
                                               remove_cvref_t<AsDataType>,
                                               remove_cvref_t<tuple<AsDataType>>>;

    using BsDataTypeTuple = std::conditional_t<BDataTypeIsTuple,
                                               remove_cvref_t<BsDataType>,
                                               remove_cvref_t<tuple<BsDataType>>>;

    using ComputeDataType = remove_cvref_t<std::tuple_element_t<number<0>{}, ComputeDataTypeTuple>>;
    using ADataType       = remove_cvref_t<std::tuple_element_t<number<0>{}, AsDataTypeTuple>>;
    using ALayout         = remove_cvref_t<std::tuple_element_t<number<0>{}, AsLayoutTuple>>;
    using BDataType       = remove_cvref_t<std::tuple_element_t<number<0>{}, BsDataTypeTuple>>;
    using BLayout         = remove_cvref_t<std::tuple_element_t<number<0>{}, BsLayoutTuple>>;

    static constexpr bool TransposeC            = Traits::TransposeC;
    static constexpr index_t NumWaveGroups      = Traits::NumWaveGroups;
    static constexpr bool UseStructuredSparsity = Traits::UseStructuredSparsity;

    static constexpr index_t kBlockSize = BlockGemmShape::NumWarps * get_warp_size();

    static constexpr bool kPadM = Traits::kPadM;
    static constexpr bool kPadN = Traits::kPadN;
    static constexpr bool kPadK = Traits::kPadK;

    static constexpr bool DoubleSmemBuffer  = Traits::DoubleSmemBuffer;
    static constexpr auto Scheduler         = GemmPipelineScheduler::Default;
    static constexpr index_t VectorLoadSize = Traits::_VectorSize;

    // In the base situation, the Preshuffle setting should be false.
    static constexpr bool Preshuffle = false;

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        // clang-format off
        return concat('_', "gemm_problem", 
                      concat('x', VectorLoadSize, kBlockSize),
                      concat('x', kPadM, kPadN, kPadK),
                      Scheduler);
        // clang-format on
    }

    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentA()
    {
        constexpr index_t PackedSize =
            ck_tile::numeric_traits<remove_cvref_t<ADataType>>::PackedSize;
        if constexpr(std::is_same_v<ALayout, ck_tile::tensor_layout::gemm::ColumnMajor>)
        {
            constexpr index_t pixels_per_thread =
                BlockGemmShape::kM * BlockGemmShape::kK / kBlockSize;
            return pixels_per_thread < PackedSize * VectorLoadSize / sizeof(ADataType)
                       ? pixels_per_thread
                       : PackedSize * VectorLoadSize / sizeof(ADataType);
        }
        else
        {
            return VectorLoadSize / sizeof(ADataType);
        }
    }

    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentB()
    {
        constexpr index_t PackedSize =
            ck_tile::numeric_traits<remove_cvref_t<BDataType>>::PackedSize;
        if constexpr(std::is_same_v<BLayout, ck_tile::tensor_layout::gemm::RowMajor>)
        {
            constexpr index_t pixels_per_thread =
                BlockGemmShape::kN * BlockGemmShape::kK / kBlockSize;
            return pixels_per_thread < PackedSize * VectorLoadSize / sizeof(BDataType)
                       ? pixels_per_thread
                       : PackedSize * VectorLoadSize / sizeof(BDataType);
        }
        else
        {
            return PackedSize * VectorLoadSize / sizeof(BDataType);
        }
    }

    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentC()
    {
        if constexpr(std::is_same_v<CLayout, ck_tile::tensor_layout::gemm::ColumnMajor>)
        {
            constexpr index_t N1 = kBlockSize / get_warp_size();
            constexpr index_t N2 = std::min(BlockGemmShape::kN / N1, get_warp_size());
            constexpr index_t M0 = get_warp_size() / N2;
            constexpr index_t M1 = BlockGemmShape::kM / M0;

            return std::min(M1, static_cast<index_t>(VectorLoadSize / sizeof(CDataType)));
        }
        else
        {
            constexpr index_t M1 = kBlockSize / get_warp_size();
            constexpr index_t M2 = std::min(BlockGemmShape::kM / M1, get_warp_size());
            constexpr index_t N0 = get_warp_size() / M2;
            constexpr index_t N1 = BlockGemmShape::kN / N0;

            return std::min(N1, static_cast<index_t>(VectorLoadSize / sizeof(CDataType)));
        }
    }

    static constexpr index_t VectorSizeA = []() {
        if constexpr(FixedVectorSize)
        {
            return VectorSizeA_;
        }
        else if constexpr(std::is_same_v<AsLayout, tensor_layout::gemm::RowMajor>)
        {
            return kPadK ? 1 : GetAlignmentA();
        }
        else
        {
            return kPadM ? 1 : GetAlignmentA();
        }
    }();

    static constexpr index_t VectorSizeB = []() {
        if constexpr(FixedVectorSize)
        {
            return VectorSizeB_;
        }
        else if constexpr(std::is_same_v<BsLayout, tensor_layout::gemm::ColumnMajor>)
        {
            return kPadN ? 1 : GetAlignmentB();
        }
        else
        {
            return kPadK ? 1 : GetAlignmentB();
        }
    }();
    static constexpr index_t VectorSizeC = []() {
        if constexpr(std::is_same_v<CLayout, tensor_layout::gemm::RowMajor>)
        {
            return kPadN ? 1 : GetAlignmentC();
        }
        else
        {
            return kPadM ? 1 : GetAlignmentC();
        }
    }();
};

template <typename AsDataType_,
          typename BsDataType_,
          typename EDataType_,
          typename BlockGemmShape_,
          typename Traits_,
          typename AElementWise_    = ck_tile::element_wise::PassThrough,
          typename BElementWise_    = ck_tile::element_wise::PassThrough,
          typename ComputeDataType_ = AsDataType_,
          bool FixedVectorSize_     = false,
          index_t VectorSizeA_      = 1,
          index_t VectorSizeB_      = 1>
using GemmPipelineProblem = GemmPipelineProblemBase<AsDataType_,
                                                    BsDataType_,
                                                    EDataType_,
                                                    BlockGemmShape_,
                                                    Traits_,
                                                    ComputeDataType_,
                                                    AElementWise_,
                                                    BElementWise_,
                                                    FixedVectorSize_,
                                                    VectorSizeA_,
                                                    VectorSizeB_>;

template <typename AsDataType_,
          typename BsDataType_,
          typename EDataType_,
          typename BlockGemmShape_,
          typename Traits_,
          GemmPipelineScheduler Scheduler_ = GemmPipelineScheduler::Intrawave,
          typename AElementWise_           = ck_tile::element_wise::PassThrough,
          typename BElementWise_           = ck_tile::element_wise::PassThrough,
          typename ComputeDataType_        = AsDataType_,
          bool FixedVectorSize_            = false,
          index_t VectorSizeA_             = 1,
          index_t VectorSizeB_             = 1>
struct UniversalGemmPipelineProblem
{
    using Traits = remove_cvref_t<Traits_>;

    using AsDataType   = remove_cvref_t<AsDataType_>;
    using BsDataType   = remove_cvref_t<BsDataType_>;
    using CDataType    = remove_cvref_t<EDataType_>; // actually AccDataType
    using AElementWise = remove_cvref_t<AElementWise_>;
    using BElementWise = remove_cvref_t<BElementWise_>;

    static constexpr bool FixedVectorSize = FixedVectorSize_;

    using BlockGemmShape = remove_cvref_t<BlockGemmShape_>;

    using AsLayout = remove_cvref_t<typename Traits::AsLayout>;
    using BsLayout = remove_cvref_t<typename Traits::BsLayout>;
    using CLayout  = remove_cvref_t<typename Traits::CLayout>;

    static constexpr bool ComputeDataTypeIsTuple = is_detected<is_tuple, ComputeDataType_>::value;
    static constexpr bool ADataTypeIsTuple       = is_detected<is_tuple, AsDataType>::value;
    static constexpr bool BDataTypeIsTuple       = is_detected<is_tuple, BsDataType>::value;

    static constexpr bool ALayoutIsTuple = is_detected<is_tuple, AsLayout>::value;
    static constexpr bool BLayoutIsTuple = is_detected<is_tuple, BsLayout>::value;

    using ComputeDataTypeTuple = std::conditional_t<ComputeDataTypeIsTuple,
                                                    remove_cvref_t<ComputeDataType_>,
                                                    remove_cvref_t<tuple<ComputeDataType_>>>;
    using AsLayoutTuple        = std::
        conditional_t<ALayoutIsTuple, remove_cvref_t<AsLayout>, remove_cvref_t<tuple<AsLayout>>>;
    using BsLayoutTuple = std::
        conditional_t<BLayoutIsTuple, remove_cvref_t<BsLayout>, remove_cvref_t<tuple<BsLayout>>>;

    using AsDataTypeTuple = std::conditional_t<ADataTypeIsTuple,
                                               remove_cvref_t<AsDataType>,
                                               remove_cvref_t<tuple<AsDataType>>>;

    using BsDataTypeTuple = std::conditional_t<BDataTypeIsTuple,
                                               remove_cvref_t<BsDataType>,
                                               remove_cvref_t<tuple<BsDataType>>>;

    using ComputeDataType = remove_cvref_t<std::tuple_element_t<number<0>{}, ComputeDataTypeTuple>>;
    using ADataType       = remove_cvref_t<std::tuple_element_t<number<0>{}, AsDataTypeTuple>>;
    using ALayout         = remove_cvref_t<std::tuple_element_t<number<0>{}, AsLayoutTuple>>;
    using BDataType       = remove_cvref_t<std::tuple_element_t<number<0>{}, BsDataTypeTuple>>;
    using BLayout         = remove_cvref_t<std::tuple_element_t<number<0>{}, BsLayoutTuple>>;

    static constexpr bool TransposeC            = Traits::TransposeC;
    static constexpr index_t NumWaveGroups      = Traits::NumWaveGroups;
    static constexpr bool UseStructuredSparsity = Traits::UseStructuredSparsity;

    static constexpr index_t kBlockSize = BlockGemmShape::NumWarps * get_warp_size();

    static constexpr bool kPadM = Traits::kPadM;
    static constexpr bool kPadN = Traits::kPadN;
    static constexpr bool kPadK = Traits::kPadK;

    static constexpr bool DoubleSmemBuffer = Traits::DoubleSmemBuffer;
    static constexpr auto Scheduler        = Scheduler_;
    static constexpr bool Preshuffle       = Traits::Preshuffle;

    static constexpr index_t VectorSizeA = VectorSizeA_;
    static constexpr index_t VectorSizeB = VectorSizeB_;

    static constexpr index_t VectorLoadSize = Traits::_VectorSize;
    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        // clang-format off
        return concat('_', "gemm_problem", 
                      concat('x', kBlockSize),
                      concat('x', kPadM, kPadN, kPadK),
                      Scheduler,
                      "NumWaveGroups",
                      NumWaveGroups,
                      "DoubleSmemBuffer",
                      DoubleSmemBuffer
                    );
        // clang-format on
    }
};

template <typename ADataType_,
          typename BDataType_,
          typename CDataType_,
          typename BlockGemmShape_,
          typename Traits_,
          GemmPipelineScheduler Scheduler_      = GemmPipelineScheduler::Intrawave,
          bool HasHotLoop_                      = true,
          TailNumber TailNum_                   = TailNumber::Full,
          amd_buffer_coherence_enum BMemNTType_ = amd_buffer_coherence_enum::coherence_default,
          bool BPreShufflePermute_              = false,
          typename ComputeDataType_             = ADataType_>
struct FlatmmPipelineProblem
{
    using Traits = remove_cvref_t<Traits_>;

    using ADataType       = remove_cvref_t<ADataType_>;
    using BDataType       = remove_cvref_t<BDataType_>;
    using CDataType       = remove_cvref_t<CDataType_>;
    using ComputeDataType = remove_cvref_t<ComputeDataType_>;

    using BlockGemmShape = remove_cvref_t<BlockGemmShape_>;

    using ALayout = remove_cvref_t<typename Traits::AsLayout>;
    using BLayout = remove_cvref_t<typename Traits::BsLayout>;
    using CLayout = remove_cvref_t<typename Traits::CLayout>;

    static constexpr bool TransposeC            = Traits::TransposeC;
    static constexpr index_t NumWaveGroups      = Traits::NumWaveGroups;
    static constexpr bool UseStructuredSparsity = Traits::UseStructuredSparsity;

    static constexpr index_t kBlockSize = BlockGemmShape::NumWarps * get_warp_size();

    static constexpr bool kPadM = Traits::kPadM;
    static constexpr bool kPadN = Traits::kPadN;
    static constexpr bool kPadK = Traits::kPadK;

    static constexpr bool DoubleSmemBuffer = Traits::DoubleSmemBuffer;

    static constexpr auto Scheduler         = GemmPipelineScheduler::Default;
    static constexpr index_t VectorLoadSize = Traits::_VectorSize;

    static constexpr auto HasHotLoop = HasHotLoop_;
    static constexpr auto TailNum    = TailNum_;

    static constexpr auto BMemNTType         = BMemNTType_;
    static constexpr bool BPreShufflePermute = BPreShufflePermute_;

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        // clang-format off
        return concat('_', "gemm_problem", 
                      concat('x', VectorLoadSize, kBlockSize),
                      concat('x', kPadM, kPadN, kPadK),
                      Scheduler);
        // clang-format on
    }

    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentA()
    {
        constexpr index_t PackedSize =
            ck_tile::numeric_traits<remove_cvref_t<ADataType>>::PackedSize;
        if constexpr(std::is_same_v<ALayout, ck_tile::tensor_layout::gemm::ColumnMajor>)
        {
            constexpr index_t pixels_per_thread =
                BlockGemmShape::kM * BlockGemmShape::kK / kBlockSize;
            return pixels_per_thread < PackedSize * VectorLoadSize / sizeof(ADataType)
                       ? pixels_per_thread
                       : PackedSize * VectorLoadSize / sizeof(ADataType);
        }
        else
        {
            return VectorLoadSize / sizeof(ADataType);
        }
    }

    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentB()
    {
        constexpr index_t PackedSize =
            ck_tile::numeric_traits<remove_cvref_t<BDataType>>::PackedSize;
        if constexpr(std::is_same_v<BLayout, ck_tile::tensor_layout::gemm::RowMajor>)
        {
            constexpr index_t pixels_per_thread =
                BlockGemmShape::kN * BlockGemmShape::kK / kBlockSize;
            return pixels_per_thread < PackedSize * VectorLoadSize / sizeof(BDataType)
                       ? pixels_per_thread
                       : PackedSize * VectorLoadSize / sizeof(BDataType);
        }
        else
        {
            return PackedSize * VectorLoadSize / sizeof(BDataType);
        }
    }

    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentC()
    {
        if constexpr(std::is_same_v<CLayout, ck_tile::tensor_layout::gemm::ColumnMajor>)
        {
            constexpr index_t N1 = kBlockSize / get_warp_size();
            constexpr index_t N2 = std::min(BlockGemmShape::kN / N1, get_warp_size());
            constexpr index_t M0 = get_warp_size() / N2;
            constexpr index_t M1 = BlockGemmShape::kM / M0;

            return std::min(M1, static_cast<index_t>(VectorLoadSize / sizeof(CDataType)));
        }
        else
        {
            constexpr index_t M1 = kBlockSize / get_warp_size();
            constexpr index_t M2 = std::min(BlockGemmShape::kM / M1, get_warp_size());
            constexpr index_t N0 = get_warp_size() / M2;
            constexpr index_t N1 = BlockGemmShape::kN / N0;

            return std::min(N1, static_cast<index_t>(VectorLoadSize / sizeof(CDataType)));
        }
    }

    static constexpr index_t VectorSizeA = []() {
        if constexpr(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>)
        {
            return kPadK ? 1 : GetAlignmentA();
        }
        else
        {
            return kPadM ? 1 : GetAlignmentA();
        }
    }();

    static constexpr index_t VectorSizeB = []() {
        if constexpr(std::is_same_v<BLayout, tensor_layout::gemm::ColumnMajor>)
        {
            return kPadN ? 1 : GetAlignmentB();
        }
        else
        {
            return kPadK ? 1 : GetAlignmentB();
        }
    }();
    static constexpr index_t VectorSizeC = []() {
        if constexpr(std::is_same_v<CLayout, tensor_layout::gemm::RowMajor>)
        {
            return kPadN ? 1 : GetAlignmentC();
        }
        else
        {
            return kPadM ? 1 : GetAlignmentC();
        }
    }();
};

} // namespace ck_tile
