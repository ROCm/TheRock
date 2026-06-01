// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/host/concat.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/host/stream_utils.hpp"
#include "ck_tile/core/utility/env.hpp"
#include "ck_tile/core/utility/type_traits.hpp"

namespace ck_tile {

/// @brief The Universal GEMM kernel host arguments.
///
/// @par Overview
///      This structure is passed to @ref UniversalGemmKernel "UniversalGemmKernel" when creating
///      kernel arguments object. It contain all necessary information required to build proper
///      kernel argument and launch kernel on GPU. This structure defines the GEMM problem
///      configuration by stating all required information like M,N,K sizes and respective strides.
///      NumATensor describes the number of A tensors. The minimum number of tensors is 1(required).
///      NumBTensor describes the number of B tensors. The minimum number of tensors is 1(required).
///      NumDTensor describes the number of D tensors. The minimum number of tensors is 0(not
///      required).
template <index_t NumATensor = 1, index_t NumBTensor = 1, index_t NumDTensor = 0>
struct UniversalGemmHostArgs
{
    CK_TILE_HOST UniversalGemmHostArgs(const std::array<const void*, NumATensor>& as_ptr_,
                                       const std::array<const void*, NumBTensor>& bs_ptr_,
                                       const std::array<const void*, NumDTensor>& ds_ptr_,
                                       void* e_ptr_,
                                       index_t k_batch_,
                                       index_t M_,
                                       index_t N_,
                                       index_t K_,
                                       const std::array<index_t, NumATensor>& stride_As_,
                                       const std::array<index_t, NumBTensor>& stride_Bs_,
                                       const std::array<index_t, NumDTensor>& stride_Ds_,
                                       index_t stride_E_)
        : as_ptr(as_ptr_),
          bs_ptr(bs_ptr_),
          ds_ptr(ds_ptr_),
          e_ptr(e_ptr_),
          M(M_),
          N(N_),
          K(K_),
          stride_As(stride_As_),
          stride_Bs(stride_Bs_),
          stride_Ds(stride_Ds_),
          stride_E(stride_E_),
          k_batch(k_batch_)
    {
    }

    const std::array<const void*, NumATensor> as_ptr;
    const std::array<const void*, NumBTensor> bs_ptr;
    const std::array<const void*, NumDTensor> ds_ptr;
    union
    {
        void* e_ptr;
        void* c_ptr;
    };
    index_t M;
    index_t N;
    index_t K;
    const std::array<index_t, NumATensor> stride_As;
    const std::array<index_t, NumBTensor> stride_Bs;
    const std::array<index_t, NumDTensor> stride_Ds;
    union
    {
        index_t stride_E;
        index_t stride_C;
    };

    index_t k_batch;
};

/// @brief The GEMM kernel device arguments.
template <index_t NumATensor = 1, index_t NumBTensor = 1, index_t NumDTensor = 0>
struct UniversalGemmKernelArgs
{
    /// @brief The As input tensor's pointer to device memory.
    const std::array<const void*, NumATensor> as_ptr;
    /// @brief The Bs input tensor's pointer to device memory.
    const std::array<const void*, NumBTensor> bs_ptr;
    /// @brief The Ds input tensor's pointer to device memory.
    const std::array<const void*, NumDTensor> ds_ptr;
    /// @brief The E output tensor's pointer to device memory.
    void* e_ptr;
    /// @brief GEMM's M dimension size.
    index_t M;
    /// @brief GEMM's N dimension size.
    index_t N;
    /// @brief GEMM's K dimension size.
    index_t K;
    /// @brief The distance between consecutive elements of non-contiguous dimension
    ///        (in memory) of As tensor.
    std::array<index_t, NumATensor> stride_As;
    /// @brief The distance between consecutive elements of non-contiguous dimension
    ///        (in memory) of Bs tensor.
    std::array<index_t, NumBTensor> stride_Bs;
    /// @brief The distance between consecutive elements of non-contiguous dimension
    ///        (in memory) of Ds tensor.
    std::array<index_t, NumDTensor> stride_Ds;
    /// @brief The distance between consecutive elements of non-contiguous dimension
    ///        (in memory) of E tensor.
    index_t stride_E;
    index_t k_batch;
};

/// @brief The Universal GEMM kernel template.
///
/// @paragraph Overview Overview
///            This class provides the generic matrix multiplication kernel template. By semantic
///            division of GEMM algorithm into following parts we achieve flexible, versatile
///            and robust kernel implementation.
///
///            @li @b Prolog - The start of GEMM kernel implementation in @ref operator()
///                function call operator" which determines the work scope of each workgroup.
///            @li @b GemmPipeline - The core part @a "heart" of matrix multiplication algorithm.
///                This is the place where each workgroup is loading data from global memory and
///                carrying out dot products.
///            @li @b Epilogue - The @a "final" part of matrix multiplication implementation
///                 responsible for storing results to global memory. This is also the place where
///                 any additional operator fusion may take place.
///
///            Additionally both @ref GemmPipeline_ "GemmPipeline" and @ref EpiloguePipeline_
///            "EpiloguePipeline" are parameterized with so called @a Policy which determines all
///            internal details of those functional parts. You can think of it like both gemm and
///            epilogue pipelines provides the control-flow logic controlled by policies. Moreover
///            the policy is responsible for definition of all necessary data layouts and thread's
///            work distribution.
///
/// @tparam TilePartitioner_    The type of class providing mapping of workgroup index into the
///                             output data tile to be calculated. It determines the workgroup to
///                             data relationship (or in other words - which data would be
///                             processed and calculated by which workgroup).
/// @tparam GemmPipeline_       The type of class which provides the core part of matrix
///                             multiplication. This class should provide implementation of data
///                             loading from global memory and performing block-wise matrix
///                             multiplication. You can think of it as a work done by single
///                             workgroup point of view.
/// @tparam EpiloguePipeline_   The type of class providing the final part of matrix
///                             multiplication implementation. It is responsible for storing
///                             results calculated by @ref GemmPipeline_ "GemmPipeline" to
///                             the output E tensor in global memory.
template <typename TilePartitioner_, typename GemmPipeline_, typename EpiloguePipeline_>
struct UniversalGemmKernel
{
    using TilePartitioner  = remove_cvref_t<TilePartitioner_>;
    using GemmPipeline     = remove_cvref_t<GemmPipeline_>;
    using EpiloguePipeline = remove_cvref_t<EpiloguePipeline_>;

    static constexpr bool ADataTypeIsTuple =
        is_detected<is_tuple, typename GemmPipeline::AsDataType>::value;
    static constexpr bool BDataTypeIsTuple =
        is_detected<is_tuple, typename GemmPipeline::BsDataType>::value;
    static constexpr bool DDataTypeIsTuple =
        is_detected<is_tuple, typename EpiloguePipeline::DsDataType>::value;
    static constexpr bool ALayoutIsTuple =
        is_detected<is_tuple, typename GemmPipeline::AsLayout>::value;
    static constexpr bool BLayoutIsTuple =
        is_detected<is_tuple, typename GemmPipeline::BsLayout>::value;
    static constexpr bool DLayoutIsTuple =
        is_detected<is_tuple, typename EpiloguePipeline::DsLayout>::value;

    using AsLayout = std::conditional_t<ALayoutIsTuple,
                                        remove_cvref_t<typename GemmPipeline::AsLayout>,
                                        remove_cvref_t<tuple<typename GemmPipeline::ALayout>>>;
    using BsLayout = std::conditional_t<BLayoutIsTuple,
                                        remove_cvref_t<typename GemmPipeline::BsLayout>,
                                        remove_cvref_t<tuple<typename GemmPipeline::BLayout>>>;

    using DsLayout = std::conditional_t<DLayoutIsTuple,
                                        remove_cvref_t<typename EpiloguePipeline::DsLayout>,
                                        remove_cvref_t<tuple<typename EpiloguePipeline::DsLayout>>>;

    using AsDataType = std::conditional_t<ADataTypeIsTuple,
                                          remove_cvref_t<typename GemmPipeline::AsDataType>,
                                          remove_cvref_t<tuple<typename GemmPipeline::ADataType>>>;

    using BsDataType = std::conditional_t<BDataTypeIsTuple,
                                          remove_cvref_t<typename GemmPipeline::BsDataType>,
                                          remove_cvref_t<tuple<typename GemmPipeline::BDataType>>>;

    using DsDataType =
        std::conditional_t<DDataTypeIsTuple,
                           remove_cvref_t<typename EpiloguePipeline::DsDataType>,
                           remove_cvref_t<tuple<typename EpiloguePipeline::DsDataType>>>;

    using CLayout   = remove_cvref_t<typename GemmPipeline::CLayout>;
    using EDataType = remove_cvref_t<typename EpiloguePipeline::ODataType>;

    using AElementWise = remove_cvref_t<typename GemmPipeline::AElementWise>;
    using BElementWise = remove_cvref_t<typename GemmPipeline::BElementWise>;

    static constexpr index_t kBlockSize = GemmPipeline::BlockSize;

    // Get the persistent kernel if the pipeline has it available
    struct has_persistent_kernel
    {
        template <typename T>
        using has_persistent_type = decltype(T::UsePersistentKernel);

        static constexpr bool value = []() {
            if constexpr(is_detected<has_persistent_type, GemmPipeline>{})
                return GemmPipeline::UsePersistentKernel;
            else
                return false;
        }();
    };
    static constexpr bool PersistentKernel = has_persistent_kernel::value;

    // Check if TilePartitioner has GetOutputOffset method with kargs and k_id
    struct has_tile_partitioner_output_offset_impl
    {
        template <typename T, typename KernelArgs>
        using has_get_output_offset_t =
            decltype(T::GetOutputOffset(std::declval<KernelArgs>(), std::declval<index_t>()));

        static constexpr bool value = []() {
            if constexpr(is_detected<has_get_output_offset_t, TilePartitioner>{})
                return true;
            else
                return false;
        }();
    };
    static constexpr bool has_tile_partitioner_output_offset =
        has_tile_partitioner_output_offset_impl::value;

    static constexpr auto I0 = number<0>();
    static constexpr auto I1 = number<1>();
    static constexpr auto I2 = number<2>();
    static constexpr auto I3 = number<3>{};

    static constexpr index_t NumATensor = AsDataType::size();
    static constexpr index_t NumBTensor = BsDataType::size();
    static constexpr index_t NumDTensor = DsDataType::size();

    using ADataType = remove_cvref_t<std::tuple_element_t<I0, AsDataType>>;
    using BDataType = remove_cvref_t<std::tuple_element_t<I0, BsDataType>>;

    static_assert(AsLayout::size() == AsDataType::size(),
                  "The size of AsLayout and AsDataType should be the same");

    static_assert(BsLayout::size() == BsDataType::size(),
                  "The size of BsLayout and BsDataType should be the same");

    static_assert(DsLayout::size() == DsDataType::size(),
                  "The size of DsLayout and DsDataType should be the same");

    static_assert(!GemmPipeline::BlockGemmShape::PermuteA, "Not implemented!");

    using KernelArgs =
        UniversalGemmKernelArgs<AsLayout::size(), BsLayout::size(), DsLayout::size()>;

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        // clang-format off
        return concat('_', "gemm", gemm_prec_str<ADataType, BDataType>(), GemmPipeline::GetName());
        // clang-format on
    }

    CK_TILE_HOST static constexpr auto GridSize(index_t M, index_t N, index_t KBatch)
    {
        return dim3(TilePartitioner::GridSize(M, N), 1, KBatch);
    }

    /**
     * @brief Get the maximum occupancy grid size for the persistent kernel on the current device.
     * @return The maximum occupancy grid size.
     * @note This function queries the maximum occupancy of the kernel using
     *       `hipOccupancyMaxActiveBlocksPerMultiprocessor`.
     */
    CK_TILE_HOST static auto MaxOccupancyGridSize(const stream_config& s) -> dim3
    {
        using Kernel      = UniversalGemmKernel<TilePartitioner, GemmPipeline, EpiloguePipeline>;
        const auto kernel = kentry<1, Kernel, KernelArgs>;
        int occupancy;
        ck_tile::hip_check_error(
            hipOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, kernel, BlockSize().x, 0));

        const int grid_size = get_available_compute_units(s) * occupancy;
        return dim3(grid_size, 1, 1);
    }

    CK_TILE_HOST static auto BlockSize()
    {
        if(ck_tile::is_wave32())
        {
            return dim3(kBlockSize / 2);
        }
        else
        {
            return dim3(kBlockSize);
        }
    }

    CK_TILE_HOST static constexpr KernelArgs
    MakeKernelArgs(const UniversalGemmHostArgs<NumATensor, NumBTensor, NumDTensor>& hostArgs)
    {
        return KernelArgs{hostArgs.as_ptr,
                          hostArgs.bs_ptr,
                          hostArgs.ds_ptr,
                          hostArgs.e_ptr,
                          hostArgs.M,
                          hostArgs.N,
                          hostArgs.K,
                          hostArgs.stride_As,
                          hostArgs.stride_Bs,
                          hostArgs.stride_Ds,
                          hostArgs.stride_E,
                          hostArgs.k_batch};
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return max(GemmPipeline::GetSmemSize(), EpiloguePipeline::GetSmemSize());
    }

    struct SplitKBatchOffset
    {
        // This structure distributes work evenly among splitkk workgroups
        // It's based on a principle that if there is enough work to fill all workgroups,
        // then we can distribute the (K / K1) parts among k_batch workgroups in such a way
        // that each workgroup will be doing ceil((K / K1) / splitk) or ceil((K / K1) / splitk) - 1
        // and leave the potential tail for last(splitk - 1) indexed workgroup.
        __device__ SplitKBatchOffset(const KernelArgs& kargs, const index_t k_id = blockIdx.z)
        {
            constexpr auto K1     = TilePartitioner::BlockGemmShape::WarpTile::at(number<2>{});
            const index_t num_all = amd_wave_read_first_lane(
                kargs.K / K1); // num of all loops not including potential tail
            index_t num_full = amd_wave_read_first_lane(num_all % kargs.k_batch);
            num_full         = num_full == 0 ? kargs.k_batch : num_full;

            const index_t num_full_iters =
                amd_wave_read_first_lane(std::max(integer_divide_ceil(num_all, kargs.k_batch), 1));
            const index_t full_k_read    = num_full_iters * K1;
            const index_t partial_k_read = (num_full_iters - 1) * K1;

            static_for<0, NumATensor, 1>{}([&](auto index) {
                using AiLayout = remove_cvref_t<std::tuple_element_t<index.value, AsLayout>>;
                if constexpr(std::is_same_v<tensor_layout::gemm::RowMajor, AiLayout>)
                {
                    as_k_split_offset[index] =
                        amd_wave_read_first_lane(std::min(k_id, num_full) * full_k_read +
                                                 std::max(k_id - num_full, 0) * partial_k_read);
                }
                else if constexpr(std::is_same_v<tensor_layout::gemm::ColumnMajor, AiLayout>)
                {
                    as_k_split_offset[index] =
                        amd_wave_read_first_lane((std::min(k_id, num_full) * full_k_read +
                                                  std::max(k_id - num_full, 0) * partial_k_read) *
                                                 kargs.stride_As[index]);
                }
            });

            static_for<0, NumBTensor, 1>{}([&](auto index) {
                using BiLayout = remove_cvref_t<std::tuple_element_t<index.value, BsLayout>>;
                if constexpr(std::is_same_v<tensor_layout::gemm::RowMajor, BiLayout>)
                {
                    bs_k_split_offset[index] =
                        amd_wave_read_first_lane((std::min(k_id, num_full) * full_k_read +
                                                  std::max(k_id - num_full, 0) * partial_k_read) *
                                                 kargs.stride_Bs[index]);
                }
                else if constexpr(std::is_same_v<tensor_layout::gemm::ColumnMajor, BiLayout>)
                {
                    bs_k_split_offset[index] =
                        amd_wave_read_first_lane(std::min(k_id, num_full) * full_k_read +
                                                 std::max(k_id - num_full, 0) * partial_k_read);
                }
            });

            if(k_id == kargs.k_batch - 1)
            {
                splitted_k = kargs.K - std::min(k_id, num_full) * full_k_read -
                             std::max(k_id - num_full, 0) * partial_k_read;
            }
            else if(k_id < num_full)
            {
                splitted_k = full_k_read;
            }
            else
            {
                splitted_k = partial_k_read;
            }
        }

        std::array<index_t, NumATensor> as_k_split_offset;
        std::array<index_t, NumBTensor> bs_k_split_offset;
        index_t splitted_k;
    };

    CK_TILE_HOST static bool IsSupportedArgument(const KernelArgs& kargs)
    {
        if constexpr(EpiloguePipeline::GetVectorSizeC() % 2 != 0 &&
                     is_any_of<EDataType, fp16_t, bf16_t>::value)
        {
            if(kargs.k_batch != 1)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR("Conditions not met for Kbatch >1 !");
                }
                return false;
            }
        }

        if(kargs.K < GemmPipeline::BlockGemmShape::WarpTile::at(number<2>{}) * kargs.k_batch)
        {
            if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
            {
                CK_TILE_ERROR("KBatch is too large, part of GPU wouldn't be utilized!");
            }
            return false;
        }

        const auto vectorSizeA = is_wave32() ? GemmPipeline::template GetVectorSizeA<true>()
                                             : GemmPipeline::template GetVectorSizeA<false>();
        bool AsTensorIsValid   = {true};
        static_for<0, NumATensor, 1>{}([&](auto index) {
            using AiLayout = remove_cvref_t<std::tuple_element_t<index.value, AsLayout>>;
            if constexpr(std::is_same_v<AiLayout, tensor_layout::gemm::RowMajor>)
            {
                if(kargs.K % (TilePartitioner::KPerBlock * kargs.k_batch) != 0 &&
                   GemmPipeline::kPadK == false)
                {
                    if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                    {
                        CK_TILE_ERROR(
                            "Can't support K that is not a multiple of k_batch * KPerBlock "
                            "without padding!");
                    }
                    AsTensorIsValid = false;
                }
                if(kargs.K % vectorSizeA != 0)
                {
                    const auto remainder = kargs.K % vectorSizeA;
                    constexpr ck_tile::index_t APackedSize =
                        ck_tile::numeric_traits<ADataType>::PackedSize;
                    const auto remainder_in_bytes = remainder * sizeof(ADataType) / APackedSize;
                    // oob can support to dword level
                    if(remainder_in_bytes % 4 == 0)
                    {
                        AsTensorIsValid = true;
                    }
                    else
                    {
                        if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                        {
                            CK_TILE_ERROR("K is not a multiple of vector load size for A tensor!");
                        }
                        AsTensorIsValid = false;
                    }
                }
            }
            else
            {
                if(kargs.M % TilePartitioner::MPerBlock != 0 && GemmPipeline::kPadM == false)
                {
                    if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                    {
                        CK_TILE_ERROR(
                            "Can't support M that is not a multiple of MPerBlock without padding!");
                    }
                    AsTensorIsValid = false;
                }
                if(kargs.M % vectorSizeA != 0)
                {
                    const auto remainder = kargs.M % vectorSizeA;
                    constexpr ck_tile::index_t APackedSize =
                        ck_tile::numeric_traits<ADataType>::PackedSize;
                    const auto remainder_in_bytes = remainder * sizeof(ADataType) / APackedSize;
                    // oob can support to dword level
                    if(remainder_in_bytes % 4 == 0)
                    {

                        AsTensorIsValid = true;
                    }
                    else
                    {
                        if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                        {
                            CK_TILE_ERROR("M is not a multiple of vector load size for A tensor!");
                        }
                        AsTensorIsValid = false;
                    }
                }
            }
        });

        bool BsTensorIsValid   = {true};
        const auto vectorSizeB = is_wave32() ? GemmPipeline::template GetVectorSizeB<true>()
                                             : GemmPipeline::template GetVectorSizeB<false>();
        static_for<0, NumBTensor, 1>{}([&](auto index) {
            using BiLayout = remove_cvref_t<std::tuple_element_t<index.value, BsLayout>>;
            if constexpr(std::is_same_v<BiLayout, tensor_layout::gemm::RowMajor>)
            {
                if(kargs.N % TilePartitioner::NPerBlock != 0 && GemmPipeline::kPadN == false)
                {
                    if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                    {
                        CK_TILE_ERROR(
                            "Can't support N that is not a multiple of NPerBlock without padding!");
                    }
                    BsTensorIsValid = false;
                }
                if(kargs.N % vectorSizeB != 0)
                {
                    const auto remainder = kargs.N % vectorSizeB;
                    constexpr ck_tile::index_t BPackedSize =
                        ck_tile::numeric_traits<BDataType>::PackedSize;
                    const auto remainder_in_bytes = remainder * sizeof(BDataType) / BPackedSize;
                    // oob can support to dword level
                    if(remainder_in_bytes % 4 == 0)
                    {
                        BsTensorIsValid = true;
                    }
                    else
                    {
                        if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                        {
                            CK_TILE_ERROR("N is not a multiple of vector load size for B tensor!");
                        }
                        BsTensorIsValid = false;
                    }
                }
                else
                {
                    if(kargs.K % (TilePartitioner::KPerBlock * kargs.k_batch) != 0 &&
                       GemmPipeline::kPadK == false)
                    {
                        if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                        {
                            CK_TILE_ERROR(
                                "Can't support K that is not a multiple of k_batch * KPerBlock "
                                "without padding!");
                        }
                        BsTensorIsValid = false;
                    }
                    if(kargs.K % vectorSizeB != 0)
                    {
                        const auto remainder = kargs.K % vectorSizeB;
                        constexpr ck_tile::index_t BPackedSize =
                            ck_tile::numeric_traits<BDataType>::PackedSize;
                        const auto remainder_in_bytes = remainder * sizeof(BDataType) / BPackedSize;
                        // oob can support to dword level
                        if(remainder_in_bytes % 4 == 0)
                        {
                            BsTensorIsValid = true;
                        }
                        else
                        {
                            if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                            {
                                CK_TILE_ERROR(
                                    "K is not a multiple of vector load size for B tensor!");
                            }
                            BsTensorIsValid = false;
                        }
                    }
                }
            }
        });

        bool DTensorIsValid = {true};
        static_for<0, NumDTensor, 1>{}([&](auto index) {
            using DiLayout = remove_cvref_t<std::tuple_element_t<index.value, DsLayout>>;
            if(std::is_same_v<DiLayout, CLayout> == false)
            {
                DTensorIsValid = false;
            }
            if constexpr(std::is_same_v<DiLayout, tensor_layout::gemm::RowMajor>)
            {
                if(kargs.N % TilePartitioner::NPerBlock != 0 && GemmPipeline::kPadN == false)
                {
                    if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                    {
                        CK_TILE_ERROR("Can't support N for tensor D that is not a multiple of "
                                      "NPerBlock without padding!");
                    }
                    DTensorIsValid = false;
                }
                if(kargs.N % EpiloguePipeline::GetVectorSizeD(index) != 0)
                {
                    if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                    {
                        CK_TILE_ERROR("N is not a multiple of vector load size for D tensor!");
                    }
                    DTensorIsValid = false;
                }
            }
            else
            {
                if(kargs.M % TilePartitioner::MPerBlock != 0 && GemmPipeline::kPadM == false)
                {
                    if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                    {
                        CK_TILE_ERROR("Can't support M for tensor D that is not a multiple of "
                                      "MPerBlock without padding!");
                    }
                    DTensorIsValid = false;
                }
                if(kargs.M % EpiloguePipeline::GetVectorSizeD(index) != 0)
                {
                    if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                    {
                        CK_TILE_ERROR("M is not a multiple of vector load size for D tensor!");
                    }
                    DTensorIsValid = false;
                }
            }
        });

        if constexpr(std::is_same_v<CLayout, tensor_layout::gemm::RowMajor>)
        {
            if(kargs.N % TilePartitioner::NPerBlock != 0 && GemmPipeline::kPadN == false)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR(
                        "Can't support N that is not a multiple of NPerBlock without padding!");
                }
                return false;
            }
            if(kargs.N % EpiloguePipeline::GetVectorSizeC() != 0)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR("N is not a multiple of vector load size for C tensor!");
                }
                return false;
            }
        }
        else
        {
            if(kargs.M % TilePartitioner::MPerBlock != 0 && GemmPipeline::kPadM == false)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR(
                        "Can't support M that is not a multiple of MPerBlock without padding!");
                }
                return false;
            }
            if(kargs.M % EpiloguePipeline::GetVectorSizeC() != 0)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR("M is not a multiple of vector load size for C tensor!");
                }
                return false;
            }
        }
        return AsTensorIsValid && BsTensorIsValid && DTensorIsValid;
    }

    CK_TILE_DEVICE static auto
    MakeABlockWindows(const std::array<const ADataType*, NumATensor>& as_ptr,
                      const KernelArgs& kargs,
                      const index_t k_size,
                      const index_t i_m)
    {
        // Step 1: Create tensor views for A tensors (from MakeGemmTensorViews)
        const auto& as_tensor_view = generate_tuple(
            [&](auto i) {
                using AiLayout   = remove_cvref_t<std::tuple_element_t<i.value, AsLayout>>;
                using AiDataType = remove_cvref_t<std::tuple_element_t<i.value, AsDataType>>;
                if constexpr(std::is_same_v<AiLayout, tensor_layout::gemm::RowMajor>)
                {
                    return make_naive_tensor_view<address_space_enum::global>(
                        static_cast<const AiDataType*>(as_ptr[i]),
                        make_tuple(kargs.M, k_size),
                        make_tuple(kargs.stride_As[i], 1),
                        number<GemmPipeline::GetVectorSizeA()>{},
                        number<1>{});
                }
                else
                {
                    return make_naive_tensor_view<address_space_enum::global>(
                        static_cast<const AiDataType*>(as_ptr[i]),
                        make_tuple(k_size, kargs.M),
                        make_tuple(kargs.stride_As[i], 1),
                        number<GemmPipeline::GetVectorSizeA()>{},
                        number<1>{});
                }
            },
            number<NumATensor>{});

        // Step 2: Create padded views (from MakeGemmPadViews)
        const auto& as_pad_view = generate_tuple(
            [&](auto i) {
                using AiLayout = remove_cvref_t<std::tuple_element_t<i.value, AsLayout>>;
                if constexpr(std::is_same_v<AiLayout, tensor_layout::gemm::RowMajor>)
                {
                    return pad_tensor_view(as_tensor_view[i],
                                           make_tuple(number<TilePartitioner::MPerBlock>{},
                                                      number<TilePartitioner::KPerBlock>{}),
                                           sequence<false, GemmPipeline::kPadK>{});
                }
                else
                {
                    return pad_tensor_view(as_tensor_view[i],
                                           make_tuple(number<TilePartitioner::KPerBlock>{},
                                                      number<TilePartitioner::MPerBlock>{}),
                                           sequence<false, GemmPipeline::kPadM>{});
                }
            },
            number<NumATensor>{});

        // Step 3: Create tile windows (from MakeGemmTileWindows)
        const auto& as_block_window = generate_tuple(
            [&](auto i) {
                using AiLayout = remove_cvref_t<std::tuple_element_t<i.value, AsLayout>>;
                if constexpr(std::is_same_v<AiLayout, tensor_layout::gemm::RowMajor>)
                {
                    return make_tile_window(as_pad_view[i],
                                            make_tuple(number<TilePartitioner::MPerBlock>{},
                                                       number<TilePartitioner::KPerBlock>{}),
                                            {i_m, 0});
                }
                else
                {
                    return make_tile_window(as_pad_view[i],
                                            make_tuple(number<TilePartitioner::KPerBlock>{},
                                                       number<TilePartitioner::MPerBlock>{}),
                                            {0, i_m});
                }
            },
            number<NumATensor>{});

        return as_block_window;
    }

    CK_TILE_DEVICE static auto
    MakeBBlockWindows(const std::array<const BDataType*, NumBTensor>& bs_ptr,
                      const KernelArgs& kargs,
                      const index_t k_size,
                      const index_t i_n)
    {
        // Step 1: Create tensor views for B tensors (from MakeGemmTensorViews)
        const auto& bs_tensor_view = generate_tuple(
            [&](auto i) {
                using BiLayout   = remove_cvref_t<std::tuple_element_t<i.value, BsLayout>>;
                using BiDataType = remove_cvref_t<std::tuple_element_t<i.value, BsDataType>>;
                if constexpr(std::is_same_v<BiLayout, tensor_layout::gemm::RowMajor>)
                {
                    if constexpr(GemmPipeline::BlockGemmShape::PermuteB)
                    {
                        constexpr index_t K1 = GemmPipeline::GetSmemPackB();
                        const index_t K0     = k_size / K1;
                        constexpr index_t VectorSizeB =
                            std::min(K1, GemmPipeline::GetVectorSizeB());
                        const auto b_k0_n_k1_desc =
                            make_naive_tensor_descriptor(make_tuple(K0, kargs.N, K1),
                                                         make_tuple(kargs.N * K1, K1, I1),
                                                         number<VectorSizeB>{},
                                                         number<1>{});
                        const auto b_n_k_desc = transform_tensor_descriptor(
                            b_k0_n_k1_desc,
                            make_tuple(make_merge_transform(make_tuple(K0, K1)),
                                       make_pass_through_transform(kargs.N)),
                            make_tuple(sequence<0, 2>{}, sequence<1>{}),
                            make_tuple(sequence<0>{}, sequence<1>{}));
                        return make_tensor_view<address_space_enum::global>(
                            static_cast<const BiDataType*>(bs_ptr[i]), b_n_k_desc);
                    }
                    else
                    {
                        return make_naive_tensor_view<address_space_enum::global>(
                            bs_ptr[i],
                            make_tuple(k_size, kargs.N),
                            make_tuple(kargs.stride_Bs[i], 1),
                            number<GemmPipeline::GetVectorSizeB()>{},
                            number<1>{});
                    }
                }
                else
                {
                    if constexpr(GemmPipeline::BlockGemmShape::PermuteB)
                    {
                        constexpr index_t K1 = GemmPipeline::GetSmemPackB();
                        const index_t K0     = k_size / K1;
                        constexpr index_t VectorSizeB =
                            std::min(K1, GemmPipeline::GetVectorSizeB());
                        const auto b_k0_n_k1_desc =
                            make_naive_tensor_descriptor(make_tuple(K0, kargs.N, K1),
                                                         make_tuple(kargs.N * K1, K1, I1),
                                                         number<VectorSizeB>{},
                                                         number<1>{});
                        const auto b_n_k_desc = transform_tensor_descriptor(
                            b_k0_n_k1_desc,
                            make_tuple(make_merge_transform(make_tuple(K0, K1)),
                                       make_pass_through_transform(kargs.N)),
                            make_tuple(sequence<0, 2>{}, sequence<1>{}),
                            make_tuple(sequence<1>{}, sequence<0>{}));
                        return make_tensor_view<address_space_enum::global>(
                            static_cast<const BiDataType*>(bs_ptr[i]), b_n_k_desc);
                    }
                    else
                    {
                        if constexpr(GemmPipeline::Preshuffle)
                        {
                            index_t kFlatK =
                                GemmPipeline::BlockGemmShape::flatKPerWarp *
                                (k_size / GemmPipeline::BlockGemmShape::WarpTile::at(number<2>{}));
                            index_t kFlatN = kargs.N * kargs.K / kFlatK;

                            return make_naive_tensor_view<address_space_enum::global>(
                                bs_ptr[i],
                                make_tuple(kFlatN, kFlatK),
                                make_tuple(kFlatK, 1),
                                number<GemmPipeline::GetVectorSizeB()>{},
                                number<1>{});
                        }
                        else
                        {
                            return make_naive_tensor_view<address_space_enum::global>(
                                bs_ptr[i],
                                make_tuple(kargs.N, k_size),
                                make_tuple(kargs.stride_Bs[i], 1),
                                number<GemmPipeline::GetVectorSizeB()>{},
                                number<1>{});
                        }
                    }
                }
            },
            number<NumBTensor>{});

        // Step 2: Create padded views (from MakeGemmPadViews)
        const auto& bs_pad_view = generate_tuple(
            [&](auto i) {
                using BiLayout = remove_cvref_t<std::tuple_element_t<i.value, BsLayout>>;
                if constexpr(std::is_same_v<BiLayout, tensor_layout::gemm::ColumnMajor>)
                {
                    return pad_tensor_view(bs_tensor_view[i],
                                           make_tuple(number<TilePartitioner::NPerBlock>{},
                                                      number<TilePartitioner::KPerBlock>{}),
                                           sequence<false, GemmPipeline::kPadK>{});
                }
                else
                {
                    return pad_tensor_view(bs_tensor_view[i],
                                           make_tuple(number<TilePartitioner::KPerBlock>{},
                                                      number<TilePartitioner::NPerBlock>{}),
                                           sequence<false, GemmPipeline::kPadN>{});
                }
            },
            number<NumBTensor>{});

        // Step 3: Create tile windows (from MakeGemmTileWindows)
        const auto& bs_block_window = generate_tuple(
            [&](auto i) {
                using BiLayout = remove_cvref_t<std::tuple_element_t<i.value, BsLayout>>;
                if constexpr(GemmPipeline::Preshuffle)
                {
                    return make_tile_window(
                        bs_pad_view[i],
                        make_tuple(number<GemmPipeline::BlockGemmShape::flatNPerWarp>{},
                                   number<GemmPipeline::BlockGemmShape::flatKPerWarp>{}),
                        {static_cast<int>(i_n / GemmPipeline::BlockGemmShape::WarpTile::at(I1)),
                         0});
                }
                else
                {
                    if constexpr(std::is_same_v<BiLayout, tensor_layout::gemm::ColumnMajor>)
                    {
                        return make_tile_window(bs_pad_view[i],
                                                make_tuple(number<TilePartitioner::NPerBlock>{},
                                                           number<TilePartitioner::KPerBlock>{}),
                                                {i_n, 0});
                    }
                    else
                    {
                        return make_tile_window(bs_pad_view[i],
                                                make_tuple(number<TilePartitioner::KPerBlock>{},
                                                           number<TilePartitioner::NPerBlock>{}),
                                                {0, i_n});
                    }
                }
            },
            number<NumBTensor>{});

        return bs_block_window;
    }

    CK_TILE_DEVICE static auto MakeDBlockWindows(const std::array<const void*, NumDTensor>& ds_ptr,
                                                 const KernelArgs& kargs,
                                                 const index_t i_m,
                                                 const index_t i_n)
    {
        // Step 1: Create tensor views for D tensors (from MakeGemmTensorViews)
        const auto& ds_tensor_view = generate_tuple(
            [&](auto i) {
                using DiLayout   = remove_cvref_t<std::tuple_element_t<i.value, DsLayout>>;
                using DDataType_ = remove_cvref_t<std::tuple_element_t<i.value, DsDataType>>;
                if constexpr(std::is_same_v<DiLayout, tensor_layout::gemm::RowMajor>)
                {
                    return make_naive_tensor_view<address_space_enum::global>(
                        static_cast<const DDataType_*>(ds_ptr[i]),
                        make_tuple(kargs.M, kargs.N),
                        make_tuple(kargs.stride_Ds[i], 1),
                        number<EpiloguePipeline::GetVectorSizeD(i)>{},
                        number<1>{});
                }
                else
                {
                    return make_naive_tensor_view<address_space_enum::global>(
                        static_cast<const DDataType_*>(ds_ptr[i]),
                        make_tuple(kargs.N, kargs.M),
                        make_tuple(kargs.stride_Ds[i], 1),
                        number<EpiloguePipeline::GetVectorSizeD(i)>{},
                        number<1>{});
                }
            },
            number<NumDTensor>{});

        // Step 2: Create padded views (from MakeGemmPadViews)
        const auto& ds_pad_view = generate_tuple(
            [&](auto i) {
                using DiLayout = remove_cvref_t<std::tuple_element_t<i.value, DsLayout>>;
                if constexpr(std::is_same_v<DiLayout, tensor_layout::gemm::RowMajor>)
                {
                    return pad_tensor_view(ds_tensor_view[i],
                                           make_tuple(number<TilePartitioner::MPerBlock>{},
                                                      number<TilePartitioner::NPerBlock>{}),
                                           sequence<false, GemmPipeline::kPadN>{});
                }
                else
                {
                    return pad_tensor_view(ds_tensor_view[i],
                                           make_tuple(number<TilePartitioner::NPerBlock>{},
                                                      number<TilePartitioner::MPerBlock>{}),
                                           sequence<false, GemmPipeline::kPadM>{});
                }
            },
            number<NumDTensor>{});

        // Step 3: Create tile windows (from MakeGemmTileWindows)
        const auto& ds_block_window = generate_tuple(
            [&](auto i) {
                using DiLayout = remove_cvref_t<std::tuple_element_t<i.value, DsLayout>>;
                if constexpr(std::is_same_v<DiLayout, tensor_layout::gemm::RowMajor>)
                {
                    return make_tile_window(ds_pad_view[i],
                                            make_tuple(number<TilePartitioner::MPerBlock>{},
                                                       number<TilePartitioner::NPerBlock>{}),
                                            {i_m, i_n});
                }
                else
                {
                    return make_tile_window(ds_pad_view[i],
                                            make_tuple(number<TilePartitioner::NPerBlock>{},
                                                       number<TilePartitioner::MPerBlock>{}),
                                            {i_n, i_m});
                }
            },
            number<NumDTensor>{});

        return ds_block_window;
    }

    template <memory_operation_enum DstInMemOp = memory_operation_enum::set>
    CK_TILE_DEVICE static auto MakeCBlockWindows(EDataType* e_ptr,
                                                 const KernelArgs& kargs,
                                                 const index_t i_m,
                                                 const index_t i_n)
    {
        // Step 1: Create tensor view for E/C tensor (from MakeGemmTensorViews)
        const auto& e_tensor_view = [&]() {
            if constexpr(std::is_same_v<CLayout, tensor_layout::gemm::RowMajor>)
            {
                return make_naive_tensor_view<address_space_enum::global, DstInMemOp>(
                    e_ptr,
                    make_tuple(kargs.M, kargs.N),
                    make_tuple(kargs.stride_E, 1),
                    number<EpiloguePipeline::GetVectorSizeC()>{},
                    number<1>{});
            }
            else
            {
                return make_naive_tensor_view<address_space_enum::global, DstInMemOp>(
                    e_ptr,
                    make_tuple(kargs.M, kargs.N),
                    make_tuple(1, kargs.stride_E),
                    number<1>{},
                    number<1>{});
            }
        }();

        // Step 2: Create padded view (from MakeGemmPadViews)
        const auto& e_pad_view = [&]() {
            if constexpr(std::is_same_v<CLayout, tensor_layout::gemm::RowMajor>)
            {
                return pad_tensor_view(e_tensor_view,
                                       make_tuple(number<TilePartitioner::MPerBlock>{},
                                                  number<TilePartitioner::NPerBlock>{}),
                                       sequence<false, GemmPipeline::kPadN>{});
            }
            else
            {
                return pad_tensor_view(e_tensor_view,
                                       make_tuple(number<TilePartitioner::MPerBlock>{},
                                                  number<TilePartitioner::NPerBlock>{}),
                                       sequence<GemmPipeline::kPadM, false>{});
            }
        }();

        // Step 3: Create tile window (from MakeGemmTileWindows)
        auto e_block_window = make_tile_window(
            e_pad_view,
            make_tuple(number<TilePartitioner::MPerBlock>{}, number<TilePartitioner::NPerBlock>{}),
            {i_m, i_n});

        return e_block_window;
    }

    /**
     * @brief Runs single GEMM problem cooperatively by whole workgroup.
     *
     * @param as_ptr input As pointer
     * @param bs_ptr input Bs pointer
     * @param ds_ptr input Ds pointer
     * @param e_ptr output E pointer
     * @param smem_ptr The start memory pointer of the shared memory block.
     * @param kargs GEMM kernel arguments
     * @param splitk_batch_offset splitk_batch_offset Utility structure used to calculate k batch.
     * @param block_idx_m The GEMM's output M dimension tile index processed by this workgroup.
     * @param block_idx_n The GEMM's output N dimension tile index processed by this workgroup.
     *
     */
    template <bool UseDefaultScheduler = true>
    CK_TILE_DEVICE static void RunGemm(const std::array<const ADataType*, NumATensor>& as_ptr,
                                       const std::array<const BDataType*, NumBTensor>& bs_ptr,
                                       const std::array<const void*, NumDTensor>& ds_ptr,
                                       EDataType* e_ptr,
                                       void* smem_ptr,
                                       const KernelArgs& kargs,
                                       const SplitKBatchOffset& splitk_batch_offset,
                                       const index_t block_idx_m,
                                       const index_t block_idx_n)
    {
        // Create block windows using specialized methods
        const auto& as_block_window =
            MakeABlockWindows(as_ptr, kargs, splitk_batch_offset.splitted_k, block_idx_m);
        const auto& bs_block_window =
            MakeBBlockWindows(bs_ptr, kargs, splitk_batch_offset.splitted_k, block_idx_n);
        const auto& ds_block_window = MakeDBlockWindows(ds_ptr, kargs, block_idx_m, block_idx_n);

        const index_t num_loop =
            amd_wave_read_first_lane(TilePartitioner::GetLoopNum(splitk_batch_offset.splitted_k));

        // Run GEMM cooperatively by whole workgroup.
        const auto& c_block_tile = GemmPipeline{}.template operator()(
            as_block_window, AElementWise{}, bs_block_window, BElementWise{}, num_loop, smem_ptr);

        const index_t k_batch = amd_wave_read_first_lane(kargs.k_batch);
        // Run Epilogue Pipeline
        if(k_batch == 1)
        {
            auto c_block_window = MakeCBlockWindows<memory_operation_enum::set>(
                e_ptr, kargs, block_idx_m, block_idx_n);
            EpiloguePipeline{}(c_block_window, c_block_tile, ds_block_window, smem_ptr);
        }
        else
        {
            auto c_block_window = MakeCBlockWindows<memory_operation_enum::atomic_add>(
                e_ptr, kargs, block_idx_m, block_idx_n);
            EpiloguePipeline{}(c_block_window, c_block_tile, ds_block_window, smem_ptr);
        }
    }

    CK_TILE_DEVICE static auto
    GetTileCoordinates(const KernelArgs& kargs) -> tuple<index_t, index_t>
    {
        index_t iM, iN;

        // Regular launch: use 1D block indexing
        const auto blockId          = amd_wave_read_first_lane(blockIdx.x);
        const auto [tile_m, tile_n] = TilePartitioner{kargs.M, kargs.N}.GetOutputTileIndex(blockId);
        iM                          = tile_m;
        iN                          = tile_n;

        const index_t i_m = amd_wave_read_first_lane(iM * TilePartitioner::MPerBlock);
        const index_t i_n = amd_wave_read_first_lane(iN * TilePartitioner::NPerBlock);

        return make_tuple(i_m, i_n);
    }

    // Helper functions
    CK_TILE_DEVICE static auto GetBlockId() -> index_t
    {
        // For 1D regular launch
        return amd_wave_read_first_lane(get_block_id());
    }

    CK_TILE_DEVICE static auto GetGridSize() -> index_t
    {
        // For 1D regular launch
        return amd_wave_read_first_lane(get_grid_size());
    }

    // Helper to get total number of tiles, handling both dim3 and index_t return types
    template <typename... Args>
    CK_TILE_HOST_DEVICE static auto GetNumTiles(Args&&... args) -> index_t
    {
        auto grid_size = TilePartitioner::GridSize(std::forward<Args>(args)...);

        using GridSizeType = decltype(grid_size);

        if constexpr(std::is_same_v<GridSizeType, dim3>)
        {
            // GridSize returns dim3: compute total tiles as x * y * z
            return amd_wave_read_first_lane(grid_size.x * grid_size.y * grid_size.z);
        }
        else
        {
            // GridSize returns scalar (index_t): use directly
            return amd_wave_read_first_lane(grid_size);
        }
    }

    // Non-persistent kernel entry point
    template <bool U = !PersistentKernel, typename = std::enable_if_t<U>>
    CK_TILE_DEVICE void operator()(KernelArgs kargs) const
    {
        const auto blockId  = amd_wave_read_first_lane(blockIdx.x);
        const auto [iM, iN] = TilePartitioner{kargs.M, kargs.N}.GetOutputTileIndex(blockId);
        const index_t i_m   = amd_wave_read_first_lane(iM * TilePartitioner::MPerBlock);
        const index_t i_n   = amd_wave_read_first_lane(iN * TilePartitioner::NPerBlock);

        const SplitKBatchOffset splitk_batch_offset(kargs);

        // options
        std::array<const ADataType*, NumATensor> as_ptr;
        static_for<0, NumATensor, 1>{}([&](auto i) {
            as_ptr[i] = static_cast<const ADataType*>(kargs.as_ptr[i]) +
                        splitk_batch_offset.as_k_split_offset[i];
        });

        std::array<const BDataType*, NumBTensor> bs_ptr;
        static_for<0, NumBTensor, 1>{}([&](auto i) {
            bs_ptr[i] = static_cast<const BDataType*>(kargs.bs_ptr[i]) +
                        splitk_batch_offset.bs_k_split_offset[i];
        });

        // Calculate output offset from tile partitioner and apply to output pointer
        EDataType* e_ptr = static_cast<EDataType*>(kargs.e_ptr);
        if constexpr(has_tile_partitioner_output_offset)
        {
            const index_t output_offset = TilePartitioner::GetOutputOffset(kargs, blockIdx.z);
            e_ptr += output_offset;
        }

        // allocate LDS
        __shared__ char smem_ptr[GetSmemSize()];

        constexpr auto scheduler_type =
            GemmPipeline::DoubleSmemBuffer || (GemmPipeline::NumWaveGroups == 1);
        RunGemm<scheduler_type>(
            as_ptr, bs_ptr, kargs.ds_ptr, e_ptr, smem_ptr, kargs, splitk_batch_offset, i_m, i_n);
    }

    // Persistent kernel entry point
    template <bool U = PersistentKernel, typename = std::enable_if_t<U>, typename = void>
    CK_TILE_DEVICE void operator()(KernelArgs kargs) const
    {
        const auto grid_size = amd_wave_read_first_lane(get_grid_size());
        const auto num_tiles =
            amd_wave_read_first_lane(TilePartitioner::GridSize(kargs.M, kargs.N));
        const auto num_work = amd_wave_read_first_lane(num_tiles * kargs.k_batch);
        auto block_id       = amd_wave_read_first_lane(get_block_id());

        while(block_id < num_work)
        {
            s_waitcnt_barrier();
            // Get the tile index for this block
            const auto tile_idx = amd_wave_read_first_lane(block_id % num_tiles);
            const auto [iM, iN] = TilePartitioner{kargs.M, kargs.N}.GetOutputTileIndex(tile_idx);
            const index_t i_m   = amd_wave_read_first_lane(iM * TilePartitioner::MPerBlock);
            const index_t i_n   = amd_wave_read_first_lane(iN * TilePartitioner::NPerBlock);

            // Get the SplitK offset for this block
            const auto k_batch = amd_wave_read_first_lane(block_id / num_tiles);
            const SplitKBatchOffset splitk_batch_offset(kargs, k_batch);

            std::array<const ADataType*, NumATensor> as_ptr;
            static_for<0, NumATensor, 1>{}([&](auto i) {
                as_ptr[i] = static_cast<const ADataType*>(kargs.as_ptr[i]) +
                            splitk_batch_offset.as_k_split_offset[i];
            });

            std::array<const BDataType*, NumBTensor> bs_ptr;
            static_for<0, NumBTensor, 1>{}([&](auto i) {
                bs_ptr[i] = static_cast<const BDataType*>(kargs.bs_ptr[i]) +
                            splitk_batch_offset.bs_k_split_offset[i];
            });

            // Calculate output offset from tile partitioner and apply to output pointer
            EDataType* e_ptr = static_cast<EDataType*>(kargs.e_ptr);
            if constexpr(has_tile_partitioner_output_offset)
            {
                const index_t output_offset = TilePartitioner::GetOutputOffset(kargs, k_batch);
                e_ptr += output_offset;
            }

            // allocate LDS
            __shared__ char smem_ptr[GetSmemSize()];
            // Run the GEMM

            RunGemm(as_ptr,
                    bs_ptr,
                    kargs.ds_ptr,
                    e_ptr,
                    smem_ptr,
                    kargs,
                    splitk_batch_offset,
                    i_m,
                    i_n);

            // Advance to the next work item
            block_id += grid_size;
            if(block_id >= num_work)
            {
                break;
            }
        }
    }
};
} // namespace ck_tile
