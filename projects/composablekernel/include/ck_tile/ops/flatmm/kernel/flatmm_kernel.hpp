// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"

namespace ck_tile {
struct FlatmmProblem
{
    CK_TILE_HOST FlatmmProblem() = default;
    CK_TILE_HOST FlatmmProblem(
        index_t M_, index_t N_, index_t K_, index_t stride_A_, index_t stride_B_, index_t stride_C_)
        : M(M_), N(N_), K(K_), stride_A(stride_A_), stride_B(stride_B_), stride_C(stride_C_)
    {
    }

    index_t M;
    index_t N;
    index_t K;
    index_t stride_A;
    index_t stride_B;
    index_t stride_C;
};

template <int SharedGranularityMN, int SharedGranularityK = 0, typename ScaleType_ = float>
struct FlatmmScalePointer
{
    using ScaleType                    = ScaleType_;
    static constexpr int GranularityMN = SharedGranularityMN;
    static constexpr int GranularityK  = SharedGranularityK;

    const ScaleType* ptr;

    CK_TILE_HOST_DEVICE FlatmmScalePointer() = default;
    CK_TILE_HOST_DEVICE FlatmmScalePointer(const ScaleType* ptr_) : ptr(ptr_) {}
    CK_TILE_HOST_DEVICE FlatmmScalePointer(const ScaleType* ptr_, [[maybe_unused]] index_t length_)
        : ptr(ptr_)
    {
    }

    CK_TILE_HOST_DEVICE FlatmmScalePointer operator+(index_t offset) const
    {
        FlatmmScalePointer ret;
        if constexpr(GranularityMN == 0)
        {
            ret.ptr = ptr + offset / GranularityK;
        }
        else
        {
            ret.ptr = ptr + offset / GranularityMN / GranularityK;
        }
        return ret;
    }

    CK_TILE_HOST_DEVICE ScaleType operator[](index_t i) const = delete;
};

template <int SharedGranularityMN, typename ScaleType_>
struct FlatmmScalePointer<SharedGranularityMN, 0, ScaleType_>
{
    using ScaleType                    = ScaleType_;
    static constexpr int GranularityMN = SharedGranularityMN;
    static constexpr int GranularityK  = 0;

    static_assert(GranularityMN != 0);

    const ScaleType* ptr;
    index_t length;

    CK_TILE_HOST_DEVICE FlatmmScalePointer() = default;
    CK_TILE_HOST_DEVICE FlatmmScalePointer(const ScaleType* ptr_) : ptr(ptr_), length(1) {}
    CK_TILE_HOST_DEVICE FlatmmScalePointer(const ScaleType* ptr_, index_t length_)
        : ptr(ptr_), length(length_)
    {
    }

    CK_TILE_HOST_DEVICE FlatmmScalePointer operator+(index_t offset) const
    {
        FlatmmScalePointer ret;
        if constexpr(GranularityMN == 1)
        {
            ret.ptr    = ptr + offset;
            ret.length = length - offset;
        }
        else
        {
            ret.ptr    = ptr + offset / GranularityMN;
            ret.length = length - offset / GranularityMN;
        }
        return ret;
    }

    CK_TILE_HOST_DEVICE ScaleType operator[](index_t i) const
    {
        // with additional oob check
        if constexpr(GranularityMN == 1)
            return i < length ? ptr[i] : 0;
        else
            return i / GranularityMN < length ? ptr[i / GranularityMN] : 0;
    }
};

// shared granularityMN = -1 means no scale
template <typename ScaleType_>
struct FlatmmScalePointer<-1, 0, ScaleType_>
{
    using ScaleType                    = ScaleType_;
    static constexpr int GranularityMN = -1;
    static constexpr int GranularityK  = 0;

    const ScaleType* ptr = nullptr;

    CK_TILE_HOST_DEVICE constexpr FlatmmScalePointer() = default;
    CK_TILE_HOST_DEVICE constexpr FlatmmScalePointer(const ScaleType*) {}
    CK_TILE_HOST_DEVICE constexpr FlatmmScalePointer(const ScaleType*, index_t) {}

    CK_TILE_HOST_DEVICE constexpr FlatmmScalePointer operator+(index_t) const
    {
        return FlatmmScalePointer{};
    }
    CK_TILE_HOST_DEVICE constexpr ScaleType operator[](index_t) const
    {
        return 1; // alway return 1, it doesn't change the result
    }
};

template <index_t NumDTensor = 0>
struct BaseFlatmmHostArgs
{
    CK_TILE_HOST BaseFlatmmHostArgs() = default;
    CK_TILE_HOST BaseFlatmmHostArgs(const void* a_ptr_,
                                    const void* b_ptr_,
                                    const std::array<const void*, NumDTensor>& ds_ptr_,
                                    void* e_ptr_,
                                    index_t k_batch_,
                                    index_t M_,
                                    index_t N_,
                                    index_t K_,
                                    index_t stride_A_,
                                    index_t stride_B_,
                                    const std::array<index_t, NumDTensor>& stride_Ds_,
                                    index_t stride_E_)
        : a_ptr(a_ptr_),
          b_ptr(b_ptr_),
          ds_ptr(ds_ptr_),
          e_ptr(e_ptr_),
          M(M_),
          N(N_),
          K(K_),
          stride_A(stride_A_),
          stride_B(stride_B_),
          stride_Ds(stride_Ds_),
          stride_E(stride_E_),
          k_batch(k_batch_)
    {
    }

    const void* a_ptr;
    const void* b_ptr;
    const std::array<const void*, NumDTensor> ds_ptr;
    union
    {
        void* e_ptr;
        void* c_ptr;
    };
    index_t M;
    index_t N;
    index_t K;
    index_t stride_A;
    index_t stride_B;
    const std::array<index_t, NumDTensor> stride_Ds;
    union
    {
        index_t stride_E;
        index_t stride_C;
    };

    index_t k_batch;
};
template <class ScaleM       = FlatmmScalePointer<-1>,
          class ScaleN       = FlatmmScalePointer<-1>,
          index_t NumDTensor = 0>
struct ScaleFlatmmHostArgs : public BaseFlatmmHostArgs<>
{
    CK_TILE_HOST ScaleFlatmmHostArgs() = default;
    CK_TILE_HOST ScaleFlatmmHostArgs(const void* a_ptr_,
                                     const void* b_shuffle_ptr_,
                                     const std::array<const void*, NumDTensor>& ds_ptr_,
                                     void* c_ptr_,
                                     index_t k_batch_,
                                     index_t M_,
                                     index_t N_,
                                     index_t K_,
                                     index_t stride_A_,
                                     index_t stride_B_,
                                     const std::array<index_t, NumDTensor>& stride_Ds_,
                                     index_t stride_C_,
                                     ScaleM scale_m_ = nullptr,
                                     ScaleN scale_n_ = nullptr)
        : BaseFlatmmHostArgs(a_ptr_,
                             b_shuffle_ptr_,
                             ds_ptr_,
                             c_ptr_,
                             k_batch_,
                             M_,
                             N_,
                             K_,
                             stride_A_,
                             stride_B_,
                             stride_Ds_,
                             stride_C_),
          scale_m(scale_m_),
          scale_n(scale_n_)
    {
    }
    ScaleM scale_m = nullptr;
    ScaleN scale_n = nullptr;
};

template <int NumberTensor = 0>
using FlatmmHostArgs =
    ScaleFlatmmHostArgs<FlatmmScalePointer<-1>, FlatmmScalePointer<-1>, NumberTensor>;

template <class ScaleM, class ScaleN, index_t NumDTensor = 0>
struct FlatmmKernelArgs
{
    const void* a_ptr;
    // const void* b_shuffle_ptr;
    const void* b_ptr;
    const std::array<const void*, NumDTensor> ds_ptr;
    void* e_ptr;
    index_t M;
    index_t N;
    index_t K;
    index_t stride_A;
    index_t stride_B;
    std::array<index_t, NumDTensor> stride_Ds;
    index_t stride_E;
    index_t k_batch;
    ScaleM scale_m_ptr = nullptr;
    ScaleN scale_n_ptr = nullptr;
};

template <typename TilePartitioner_, typename FlatmmPipeline_, typename EpiloguePipeline_>
struct FlatmmKernel
{
    using TilePartitioner = remove_cvref_t<TilePartitioner_>;
    using FlatmmPipeline  = remove_cvref_t<FlatmmPipeline_>;
    using BlockGemmShape =
        remove_cvref_t<typename FlatmmPipeline::BlockGemmShape>; // TileFlatmmShape
    using EpiloguePipeline              = remove_cvref_t<EpiloguePipeline_>;
    using ALayout                       = remove_cvref_t<typename FlatmmPipeline::ALayout>;
    using BLayout                       = remove_cvref_t<typename FlatmmPipeline::BLayout>;
    using ELayout                       = remove_cvref_t<typename FlatmmPipeline::CLayout>;
    using DsLayout                      = remove_cvref_t<typename EpiloguePipeline::DsLayout>;
    using DsDataType                    = remove_cvref_t<typename EpiloguePipeline::DsDataType>;
    static constexpr index_t kBlockSize = FlatmmPipeline::BlockSize;
    static constexpr bool UsePersistentKernel = FlatmmPipeline::UsePersistentKernel;

    using ADataType = remove_cvref_t<typename FlatmmPipeline::ADataType>;
    using BDataType = remove_cvref_t<typename FlatmmPipeline::BDataType>;
    // Below type is actually accumulation data type - the output of block GEMM.
    using EDataType = remove_cvref_t<typename EpiloguePipeline::ODataType>;

    static constexpr index_t NumDTensor = DsDataType::size();

    static constexpr auto I0 = number<0>();
    static constexpr auto I1 = number<1>();
    static constexpr auto I2 = number<2>();
    static constexpr auto I3 = number<3>();

    static_assert(DsLayout::size() == DsDataType::size(),
                  "The size of DsLayout and DsDataType should be the same");
    // using KernelArgs = FlatmmKernelArgs<DsLayout::size()>;

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        // clang-format off
        return concat('_', "gemm", gemm_prec_str<ADataType, BDataType>, FlatmmPipeline::GetName());
        // clang-format on
    }

    CK_TILE_HOST static constexpr auto GridSize(index_t M, index_t N, index_t KBatch)
    {
        assert(!UsePersistentKernel);
        return dim3(TilePartitioner::GridSize(M, N), 1, KBatch);
    }

    template <class ScaleM, class ScaleN>
    CK_TILE_HOST static constexpr auto
    GridSize(const FlatmmKernelArgs<ScaleM, ScaleN, DsDataType::size()>& kargs)
    {
        if constexpr(UsePersistentKernel)
        {
            hipDeviceProp_t prop;
            int deviceId = 0; // default device

            constexpr int block_size = FlatmmKernel::BlockSize().x;
            int dync_smem_size       = 0;
            int maxActiveBlocksPerCU = 0;

            [[maybe_unused]] auto e = hipGetDeviceProperties(&prop, deviceId);

            e = hipOccupancyMaxActiveBlocksPerMultiprocessor(
                &maxActiveBlocksPerCU,
                reinterpret_cast<void*>(
                    kentry<1, FlatmmKernel, FlatmmKernelArgs<ScaleM, ScaleN, DsDataType::size()>>),
                block_size,
                dync_smem_size);

            const int persistent_block_size = prop.multiProcessorCount * maxActiveBlocksPerCU;
            const int total_work_tile_cnt   = TilePartitioner::GridSize(kargs.M, kargs.N);

            // std::cout << "maxActiveBlocksPerCU: " << maxActiveBlocksPerCU
            //           << ", persistent_block_size: " << persistent_block_size
            //           << ", total_work_tile_cnt: " << total_work_tile_cnt << std::endl;

            assert(kargs.k_batch == 1);
            return dim3(min(persistent_block_size, total_work_tile_cnt), 1, kargs.k_batch);
        }
        else
        {
            return dim3(TilePartitioner::GridSize(kargs.M, kargs.N), 1, kargs.k_batch);
        }
    }

    CK_TILE_HOST static constexpr auto BlockSize() { return dim3(kBlockSize); }

    template <class ScaleM, class ScaleN>
    CK_TILE_HOST static constexpr FlatmmKernelArgs<ScaleM, ScaleN, DsDataType::size()>
    MakeKernelArgs(const ScaleFlatmmHostArgs<ScaleM, ScaleN, DsDataType::size()>& hostArgs)
    {
        return {hostArgs.a_ptr,
                hostArgs.b_ptr,
                hostArgs.ds_ptr,
                hostArgs.e_ptr,
                hostArgs.M,
                hostArgs.N,
                hostArgs.K,
                hostArgs.stride_A,
                hostArgs.stride_B,
                hostArgs.stride_Ds,
                hostArgs.stride_E,
                hostArgs.k_batch,
                hostArgs.scale_m,
                hostArgs.scale_n};
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemPingSize()
    {
        return max(FlatmmPipeline::GetSmemSize(), EpiloguePipeline::GetSmemSize());
    }
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemPongSize()
    {
        return FlatmmPipeline::GetSmemSize();
    }

    struct SplitKBatchOffset
    {
        template <class KernelArgs>
        __device__ SplitKBatchOffset(const KernelArgs& kargs, const std::size_t k_id = blockIdx.z)
        {
            constexpr auto N1   = BlockGemmShape::WarpTile::at(number<1>{});
            constexpr auto K1   = BlockGemmShape::WarpTile::at(number<2>{});
            const index_t K_t   = kargs.k_batch * K1;
            const index_t KRead = (kargs.K + K_t - 1) / K_t * K1;

            if constexpr(std::is_same_v<tensor_layout::gemm::RowMajor, ALayout>)
            {
                a_k_split_offset = k_id * KRead;
            }
            else if constexpr(std::is_same_v<tensor_layout::gemm::ColumnMajor, ALayout>)
            {
                a_k_split_offset = k_id * KRead * kargs.stride_A;
            }

            if constexpr(std::is_same_v<tensor_layout::gemm::RowMajor, BLayout>)
            {
                b_k_split_offset = k_id * KRead * kargs.stride_B * N1;
            }
            else if constexpr(std::is_same_v<tensor_layout::gemm::ColumnMajor, BLayout>)
            {
                b_k_split_offset = k_id * KRead * N1;
            }

            if(k_id < static_cast<uint32_t>(kargs.k_batch - 1))
            {
                splitted_k = KRead;
            }
            else
            {
                splitted_k = kargs.K - KRead * (kargs.k_batch - 1);
            }
        }

        index_t a_k_split_offset;
        index_t b_k_split_offset;
        index_t splitted_k;
    };

    template <class KernelArgs>
    CK_TILE_HOST static bool IsSupportedArgument(const KernelArgs& kargs)
    {
        if constexpr(EpiloguePipeline::GetVectorSizeC() % 2 != 0 &&
                     is_any_of<EDataType, fp16_t, bf16_t>::value)
        {
            if(kargs.k_batch != 1)
            {
                std::cerr << "Conditions not met for Kbatch >1 !" << std::endl;
                return false;
            }
        }
        if constexpr(UsePersistentKernel)
        {
            if(kargs.k_batch != 1)
            {
                std::cerr << "Persistent mode doesn't support Kbatch >1 !" << std::endl;
                return false;
            }
        }

        if constexpr(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>)
        {
            if(kargs.K % TilePartitioner::KPerBlock != 0 && FlatmmPipeline::kPadK == false)
            {
                std::cerr << "Can't support K that is not a multiple of KPerBlock"
                             " without padding!"
                          << std::endl;
                return false;
            }
            if(kargs.K % FlatmmPipeline::GetVectorSizeA() != 0)
            {
                std::cerr << "K is not a multiple of vector load size for A tensor!" << std::endl;
                return false;
            }
        }
        else
        {
            if(kargs.M % TilePartitioner::MPerBlock != 0 && FlatmmPipeline::kPadM == false)
            {
                std::cerr << "Can't support M that is not a multiple of MPerBlock"
                             " without padding!"
                          << std::endl;
                return false;
            }
            if(kargs.M % FlatmmPipeline::GetVectorSizeA() != 0)
            {
                std::cerr << "M is not a multiple of vector load size for A tensor!" << std::endl;
                return false;
            }
        }

        if constexpr(std::is_same_v<BLayout, tensor_layout::gemm::RowMajor>)
        {
            if(kargs.N % TilePartitioner::NPerBlock != 0 && FlatmmPipeline::kPadN == false)
            {
                std::cerr << "Can't support N that is not a multiple of NPerBlock"
                             " without padding!"
                          << std::endl;
                return false;
            }
            if(kargs.N % FlatmmPipeline::GetVectorSizeB() != 0)
            {
                std::cerr << "N is not a multiple of vector load size for B tensor!" << std::endl;
                return false;
            }
        }
        else
        {
            if(kargs.K % TilePartitioner::KPerBlock != 0 && FlatmmPipeline::kPadK == false)
            {
                std::cerr << "Can't support K that is not a multiple of KPerBlock"
                             " without padding!"
                          << std::endl;
                return false;
            }
            if(kargs.K % FlatmmPipeline::GetVectorSizeB() != 0)
            {
                std::cerr << "K is not a multiple of vector load size for B tensor!" << std::endl;
                return false;
            }
        }

        bool DTesnorIsValid = {true};
        static_for<0, NumDTensor, 1>{}([&](auto index) {
            using DiLayout = remove_cvref_t<std::tuple_element_t<index.value, DsLayout>>;
            if(std::is_same_v<DiLayout, ELayout> == false)
            {
                DTesnorIsValid = false;
            }
            if constexpr(std::is_same_v<DiLayout, tensor_layout::gemm::RowMajor>)
            {
                if(kargs.N % TilePartitioner::NPerBlock != 0 && FlatmmPipeline::kPadN == false)
                {
                    CK_TILE_ERROR("Can't support N for tensor D that is not a multiple of "
                                  "NPerBlock without padding!");
                    DTesnorIsValid = false;
                }
                if(kargs.N % EpiloguePipeline::GetVectorSizeD(index) != 0)
                {
                    CK_TILE_ERROR("N is not a multiple of vector load size for D tensor!");
                    DTesnorIsValid = false;
                }
            }
            else
            {
                if(kargs.M % TilePartitioner::MPerBlock != 0 && FlatmmPipeline::kPadM == false)
                {
                    CK_TILE_ERROR("Can't support M for tensor D that is not a multiple of "
                                  "MPerBlock without padding!");

                    DTesnorIsValid = false;
                }
                if(kargs.M % EpiloguePipeline::GetVectorSizeD(index) != 0)
                {
                    CK_TILE_ERROR("M is not a multiple of vector load size for D tensor!");
                    DTesnorIsValid = false;
                }
            }
        });

        if constexpr(std::is_same_v<ELayout, tensor_layout::gemm::RowMajor>)
        {
            if(kargs.N % TilePartitioner::NPerBlock != 0 && FlatmmPipeline::kPadN == false)
            {
                std::cerr << "Can't support N that is not a multiple of NPerBlock"
                             " without padding!"
                          << std::endl;
                return false;
            }
            if(kargs.N % EpiloguePipeline::GetVectorSizeC() != 0)
            {
                std::cerr << "N is not a multiple of vector load size for C tensor!" << std::endl;
                return false;
            }
        }
        else
        {
            if(kargs.M % TilePartitioner::MPerBlock != 0 && FlatmmPipeline::kPadM == false)
            {
                std::cerr << "Can't support M that is not a multiple of MPerBlock"
                             " without padding!"
                          << std::endl;
                return false;
            }
            if(kargs.M % EpiloguePipeline::GetVectorSizeC() != 0)
            {
                std::cerr << "M is not a multiple of vector load size for C tensor!" << std::endl;
                return false;
            }
        }
        return DTesnorIsValid;
    }

    template <typename KernelArgs>
    CK_TILE_DEVICE static auto MakeABlockWindow(const ADataType* a_ptr,
                                                const KernelArgs& kargs,
                                                const index_t k_size,
                                                const index_t block_idx_m)
    {
        // Step 1: Create tensor view
        const auto& a_tensor_view = [&]() {
            if constexpr(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>)
            {
                return make_naive_tensor_view<address_space_enum::global>(
                    a_ptr,
                    make_tuple(kargs.M, k_size),
                    make_tuple(kargs.stride_A, 1),
                    number<FlatmmPipeline::GetVectorSizeA()>{},
                    number<1>{});
            }
            else
            {
                return make_naive_tensor_view<address_space_enum::global>(
                    a_ptr,
                    make_tuple(k_size, kargs.M),
                    make_tuple(kargs.stride_A, 1),
                    number<FlatmmPipeline::GetVectorSizeA()>{},
                    number<1>{});
            }
        }();

        // Step 2: Create padded view
        const auto& a_pad_view = [&]() {
            if constexpr(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>)
            {
                return pad_tensor_view(a_tensor_view,
                                       make_tuple(number<TilePartitioner::MPerBlock>{},
                                                  number<TilePartitioner::KPerBlock>{}),
                                       sequence<false, FlatmmPipeline::kPadK>{});
            }
            else
            {
                return pad_tensor_view(a_tensor_view,
                                       make_tuple(number<TilePartitioner::KPerBlock>{},
                                                  number<TilePartitioner::MPerBlock>{}),
                                       sequence<false, FlatmmPipeline::kPadM>{});
            }
        }();

        // Step 3: Create tile window
        if constexpr(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>)
        {
            return make_tile_window(a_pad_view,
                                    make_tuple(number<TilePartitioner::MPerBlock>{},
                                               number<TilePartitioner::KPerBlock>{}),
                                    {block_idx_m, 0});
        }
        else
        {
            return make_tile_window(a_pad_view,
                                    make_tuple(number<TilePartitioner::KPerBlock>{},
                                               number<TilePartitioner::MPerBlock>{}),
                                    {0, block_idx_m});
        }
    }

    template <typename KernelArgs>
    CK_TILE_DEVICE static auto MakeBFlatBlockWindow(const BDataType* b_flat_ptr,
                                                    const KernelArgs& kargs,
                                                    const index_t block_idx_n)
    {
        // Step 1: Create tensor view
        index_t kFlatK =
            FlatmmPipeline::flatKPerWarp * (kargs.K / BlockGemmShape::WarpTile::at(I2));
        index_t kFlatN = kargs.N * kargs.K / kFlatK;

        const auto& b_flat_tensor_view = make_naive_tensor_view<address_space_enum::global>(
            b_flat_ptr,
            make_tuple(kFlatN, kFlatK),
            make_tuple(kFlatK, 1),
            number<FlatmmPipeline::GetVectorSizeB()>{},
            number<1>{});

        // Step 2: No padding needed for b_flat
        // Step 3: Create tile window
        return make_tile_window(
            b_flat_tensor_view,
            make_tuple(number<FlatmmPipeline::flatNPerWarp>{},
                       number<FlatmmPipeline::flatKPerWarp>{}),
            {static_cast<int>(block_idx_n / BlockGemmShape::WarpTile::at(I1)), 0});
    }

    template <typename KernelArgs>
    CK_TILE_DEVICE static auto MakeDBlockWindows(const std::array<const void*, NumDTensor>& ds_ptr,
                                                 const KernelArgs& kargs,
                                                 const index_t block_idx_m,
                                                 const index_t block_idx_n)
    {
        // Step 1: Create tensor views
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

        // Step 2: Create padded views
        const auto& ds_pad_view = generate_tuple(
            [&](auto i) {
                using DiLayout = remove_cvref_t<std::tuple_element_t<i.value, DsLayout>>;
                if constexpr(std::is_same_v<DiLayout, tensor_layout::gemm::RowMajor>)
                {
                    return pad_tensor_view(ds_tensor_view[i],
                                           make_tuple(number<TilePartitioner::MPerBlock>{},
                                                      number<TilePartitioner::NPerBlock>{}),
                                           sequence<false, FlatmmPipeline::kPadN>{});
                }
                else
                {
                    return pad_tensor_view(ds_tensor_view[i],
                                           make_tuple(number<TilePartitioner::NPerBlock>{},
                                                      number<TilePartitioner::MPerBlock>{}),
                                           sequence<false, FlatmmPipeline::kPadM>{});
                }
            },
            number<NumDTensor>{});

        // Step 3: Create tile windows
        return generate_tuple(
            [&](auto i) {
                using DiLayout = remove_cvref_t<std::tuple_element_t<i.value, DsLayout>>;
                if constexpr(std::is_same_v<DiLayout, tensor_layout::gemm::RowMajor>)
                {
                    return make_tile_window(ds_pad_view[i],
                                            make_tuple(number<TilePartitioner::MPerBlock>{},
                                                       number<TilePartitioner::NPerBlock>{}),
                                            {block_idx_m, block_idx_n});
                }
                else
                {
                    return make_tile_window(ds_pad_view[i],
                                            make_tuple(number<TilePartitioner::NPerBlock>{},
                                                       number<TilePartitioner::MPerBlock>{}),
                                            {block_idx_n, block_idx_m});
                }
            },
            number<NumDTensor>{});
    }

    template <memory_operation_enum DstInMemOp = memory_operation_enum::set, typename KernelArgs>
    CK_TILE_DEVICE static auto MakeEBlockWindow(EDataType* e_ptr,
                                                const KernelArgs& kargs,
                                                const index_t block_idx_m,
                                                const index_t block_idx_n)
    {
        // Step 1: Create tensor view
        const auto& e_tensor_view = [&]() {
            if constexpr(std::is_same_v<ELayout, tensor_layout::gemm::RowMajor>)
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
                    make_tuple(kargs.N, kargs.M),
                    make_tuple(kargs.stride_E, 1),
                    number<1>{},
                    number<1>{});
            }
        }();

        // Step 2: Create padded view
        const auto& e_pad_view = [&]() {
            if constexpr(std::is_same_v<ELayout, tensor_layout::gemm::RowMajor>)
            {
                return pad_tensor_view(e_tensor_view,
                                       make_tuple(number<TilePartitioner::MPerBlock>{},
                                                  number<TilePartitioner::NPerBlock>{}),
                                       sequence<false, FlatmmPipeline::kPadN>{});
            }
            else
            {
                return pad_tensor_view(e_tensor_view,
                                       make_tuple(number<TilePartitioner::MPerBlock>{},
                                                  number<TilePartitioner::NPerBlock>{}),
                                       sequence<FlatmmPipeline::kPadM, false>{});
            }
        }();

        // Step 3: Create tile window
        return make_tile_window(
            e_pad_view,
            make_tuple(number<TilePartitioner::MPerBlock>{}, number<TilePartitioner::NPerBlock>{}),
            {block_idx_m, block_idx_n});
    }

    template <typename KernelArgs>
    CK_TILE_DEVICE static auto MakeScaleMWindow(const KernelArgs& kargs,
                                                const SplitKBatchOffset& splitk_batch_offset,
                                                const index_t block_idx_m)
    {
        constexpr int ScaleGranularityM  = decltype(kargs.scale_m_ptr)::GranularityMN;
        constexpr int ScaleGranularityKA = decltype(kargs.scale_m_ptr)::GranularityK;

        auto scale_stride_m = ScaleGranularityM == 0 ? 0  // per-tensor scale
                                                     : 1; // per-token scale

        // Step 1: Create tensor view
        const auto scale_m_view = make_naive_tensor_view<address_space_enum::global>(
            kargs.scale_m_ptr.ptr,
            make_tuple(kargs.M / ScaleGranularityM,
                       ScaleGranularityKA == 0
                           ? 1
                           : (splitk_batch_offset.splitted_k / ScaleGranularityKA)),
            make_tuple(scale_stride_m, 0),
            number < ScaleGranularityM == 1 ? FlatmmPipeline::GetVectorSizeA() : 1 > {},
            number<1>{});

        // Step 2: Create tile window
        return make_tile_window(scale_m_view,
                                make_tuple(number<TilePartitioner::MPerBlock>{},
                                           number < ScaleGranularityKA == 0
                                               ? TilePartitioner::NPerBlock
                                               : TilePartitioner::KPerBlock > {}),
                                {block_idx_m, 0});
    }

    template <typename KernelArgs>
    CK_TILE_DEVICE static auto MakeScaleNWindow(const KernelArgs& kargs,
                                                const SplitKBatchOffset& splitk_batch_offset,
                                                const index_t block_idx_n)
    {
        constexpr int ScaleGranularityN  = decltype(kargs.scale_n_ptr)::GranularityMN;
        constexpr int ScaleGranularityKB = decltype(kargs.scale_n_ptr)::GranularityK;

        auto scale_stride_n = ScaleGranularityN == 0 ? 0  // per-tensor scale
                                                     : 1; // per-channel scale

        // Step 1: Create tensor view
        const auto scale_n_view = make_naive_tensor_view<address_space_enum::global>(
            kargs.scale_n_ptr.ptr,
            make_tuple(
                ScaleGranularityKB == 0 ? 1 : (splitk_batch_offset.splitted_k / ScaleGranularityKB),
                kargs.N / ScaleGranularityN),
            make_tuple(0, scale_stride_n),
            number < ScaleGranularityN == 1 ? FlatmmPipeline::GetVectorSizeB() : 1 > {},
            number<1>{});

        // Step 2: Create tile window
        return make_tile_window(scale_n_view,
                                make_tuple(number < ScaleGranularityKB == 0
                                               ? TilePartitioner::MPerBlock
                                               : TilePartitioner::KPerBlock > {},
                                           number<TilePartitioner::NPerBlock>{}),
                                {0, block_idx_n});
    }

    template <class ScaleM, class ScaleN, bool UseDefaultScheduler = true>
    CK_TILE_DEVICE static void
    RunFlatmm(const ADataType* a_ptr,
              const BDataType* b_flat_ptr,
              const std::array<const void*, NumDTensor>& ds_ptr,
              EDataType* e_ptr,
              void* smem_ptr_ping,
              void* smem_ptr_pong,
              const FlatmmKernelArgs<ScaleM, ScaleN, DsDataType::size()>& kargs,
              const SplitKBatchOffset& splitk_batch_offset,
              const index_t block_idx_m,
              const index_t block_idx_n)
    {
        // Create block windows using specialized methods
        const auto& a_block_window =
            MakeABlockWindow(a_ptr, kargs, splitk_batch_offset.splitted_k, block_idx_m);
        const auto& b_flat_block_window = MakeBFlatBlockWindow(b_flat_ptr, kargs, block_idx_n);
        const auto& ds_block_window = MakeDBlockWindows(ds_ptr, kargs, block_idx_m, block_idx_n);
        const auto& scale_m_window  = MakeScaleMWindow(kargs, splitk_batch_offset, block_idx_m);
        const auto& scale_n_window  = MakeScaleNWindow(kargs, splitk_batch_offset, block_idx_n);

        const index_t num_loop = TilePartitioner::GetLoopNum(splitk_batch_offset.splitted_k);

        // Run GEMM cooperatively by whole workgroup.
        const auto& c_block_tile = FlatmmPipeline{}.template operator()(
            a_block_window, b_flat_block_window, num_loop, smem_ptr_ping, smem_ptr_pong);

        // Run Epilogue Pipeline with k_batch dispatching
        if constexpr(ScaleM::GranularityMN != -1 || ScaleN::GranularityMN != -1)
        {
            if(kargs.k_batch == 1)
            {
                auto e_block_window = MakeEBlockWindow<memory_operation_enum::set>(
                    e_ptr, kargs, block_idx_m, block_idx_n);
                EpiloguePipeline{}
                    .template operator()<decltype(e_block_window),
                                         decltype(c_block_tile),
                                         decltype(ds_block_window)>(e_block_window,
                                                                    c_block_tile,
                                                                    ds_block_window,
                                                                    smem_ptr_ping,
                                                                    scale_m_window,
                                                                    scale_n_window);
            }
            else
            {
                auto e_block_window = MakeEBlockWindow<memory_operation_enum::atomic_add>(
                    e_ptr, kargs, block_idx_m, block_idx_n);
                EpiloguePipeline{}
                    .template operator()<decltype(e_block_window),
                                         decltype(c_block_tile),
                                         decltype(ds_block_window)>(e_block_window,
                                                                    c_block_tile,
                                                                    ds_block_window,
                                                                    smem_ptr_ping,
                                                                    scale_m_window,
                                                                    scale_n_window);
            }
        }
        else if(UseDefaultScheduler || (get_warp_id() == 0))
        {
            if(kargs.k_batch == 1)
            {
                auto e_block_window = MakeEBlockWindow<memory_operation_enum::set>(
                    e_ptr, kargs, block_idx_m, block_idx_n);
                EpiloguePipeline{}
                    .template operator()<decltype(e_block_window),
                                         decltype(c_block_tile),
                                         decltype(ds_block_window)>(
                        e_block_window, c_block_tile, ds_block_window, smem_ptr_ping);
            }
            else
            {
                auto e_block_window = MakeEBlockWindow<memory_operation_enum::atomic_add>(
                    e_ptr, kargs, block_idx_m, block_idx_n);
                EpiloguePipeline{}
                    .template operator()<decltype(e_block_window),
                                         decltype(c_block_tile),
                                         decltype(ds_block_window)>(
                        e_block_window, c_block_tile, ds_block_window, smem_ptr_ping);
            }
        }
    }

    template <class ScaleM, class ScaleN>
    CK_TILE_DEVICE void operator()(FlatmmKernelArgs<ScaleM, ScaleN, DsDataType::size()> kargs,
                                   int partition_idx = blockIdx.x) const
    {
        int total_work_tile_cnt = TilePartitioner::GridSize(kargs.M, kargs.N);

        do
        {
            const auto [iM, iN] =
                TilePartitioner{kargs.M, kargs.N}.GetOutputTileIndex(partition_idx);
            const index_t i_m = amd_wave_read_first_lane(iM * TilePartitioner::MPerBlock);
            const index_t i_n = amd_wave_read_first_lane(iN * TilePartitioner::NPerBlock);

            const SplitKBatchOffset splitk_batch_offset(kargs);
            // options
            const ADataType* a_ptr =
                static_cast<const ADataType*>(kargs.a_ptr) + splitk_batch_offset.a_k_split_offset;
            const BDataType* b_flat_ptr =
                static_cast<const BDataType*>(kargs.b_ptr) + splitk_batch_offset.b_k_split_offset;
            EDataType* e_ptr = static_cast<EDataType*>(kargs.e_ptr);

            // allocate LDS
            __shared__ char smem_ptr_ping[GetSmemPingSize()];
            __shared__ char smem_ptr_pong[GetSmemPongSize()];

            if constexpr(!(EpiloguePipeline::GetVectorSizeC() % 2 != 0 &&
                           is_any_of<EDataType, fp16_t, bf16_t>::value))
            {
                constexpr auto scheduler_type = (FlatmmPipeline::NumWaveGroups == 1);
                RunFlatmm<ScaleM, ScaleN, scheduler_type>(a_ptr,
                                                          b_flat_ptr,
                                                          kargs.ds_ptr,
                                                          e_ptr,
                                                          smem_ptr_ping,
                                                          smem_ptr_pong,
                                                          kargs,
                                                          splitk_batch_offset,
                                                          i_m,
                                                          i_n);
            }
            partition_idx += gridDim.x;
        } while(UsePersistentKernel && partition_idx < total_work_tile_cnt);
    }
};

} // namespace ck_tile
