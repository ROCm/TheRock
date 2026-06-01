// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/flatmm/kernel/flatmm_kernel.hpp"

namespace ck_tile {

template <class ScaleM       = FlatmmScalePointer<-1>,
          class ScaleN       = FlatmmScalePointer<-1>,
          index_t NumDTensor = 0>
struct GroupedFlatmmHostArgs
{
    CK_TILE_HOST GroupedFlatmmHostArgs() = default;
    CK_TILE_HOST GroupedFlatmmHostArgs(index_t group_count_,
                                       index_t* M_,
                                       index_t* N_,
                                       index_t* K_,
                                       const void** a_ptr_,
                                       index_t* stride_A_,
                                       const void** b_shuffle_ptr_,
                                       index_t* stride_B_,
                                       const std::array<const void*, NumDTensor>& ds_ptr_,
                                       const std::array<index_t, NumDTensor>& stride_Ds_,
                                       void** c_ptr_,
                                       index_t* stride_C_,
                                       index_t k_batch_,
                                       ScaleM* scale_m_ = nullptr,
                                       ScaleN* scale_n_ = nullptr)
        : group_count(group_count_),
          M(M_),
          N(N_),
          K(K_),
          a_ptr(a_ptr_),
          stride_A(stride_A_),
          b_shuffle_ptr(b_shuffle_ptr_),
          stride_B(stride_B_),
          ds_ptr(ds_ptr_),
          stride_Ds(stride_Ds_),
          c_ptr(c_ptr_),
          stride_C(stride_C_),
          k_batch(k_batch_),
          scale_m(scale_m_),
          scale_n(scale_n_)
    {
    }

    index_t group_count;
    index_t* M;
    index_t* N;
    index_t* K;
    const void** a_ptr;
    index_t* stride_A;
    const void** b_shuffle_ptr;
    index_t* stride_B;
    const std::array<const void*, NumDTensor> ds_ptr;
    const std::array<index_t, NumDTensor> stride_Ds;
    union
    {
        void** e_ptr;
        void** c_ptr;
    };
    index_t* stride_C;
    index_t k_batch;
    ScaleM* scale_m = nullptr;
    ScaleN* scale_n = nullptr;
};

template <class ScaleM       = FlatmmScalePointer<-1>,
          class ScaleN       = FlatmmScalePointer<-1>,
          index_t NumDTensor = 0>
struct ContiguousGroupedFlatmmHostArgs
{
    CK_TILE_HOST ContiguousGroupedFlatmmHostArgs() = default;
    CK_TILE_HOST ContiguousGroupedFlatmmHostArgs(index_t* M_indices_,
                                                 index_t M_,
                                                 index_t N_,
                                                 index_t K_,
                                                 const void* a_ptr_,
                                                 index_t stride_A_,
                                                 const void* b_shuffle_ptr_,
                                                 index_t stride_B_,
                                                 const std::array<const void*, NumDTensor>& ds_ptr_,
                                                 const std::array<index_t, NumDTensor>& stride_Ds_,
                                                 void* c_ptr_,
                                                 index_t stride_C_,
                                                 index_t k_batch_,
                                                 ScaleM scale_m_ = nullptr,
                                                 ScaleN scale_n_ = nullptr)
        : group_count(1),
          M_indices(M_indices_),
          M(M_),
          N(N_),
          K(K_),
          a_ptr(a_ptr_),
          stride_A(stride_A_),
          b_shuffle_ptr(b_shuffle_ptr_),
          stride_B(stride_B_),
          ds_ptr(ds_ptr_),
          stride_Ds(stride_Ds_),
          c_ptr(c_ptr_),
          stride_C(stride_C_),
          k_batch(k_batch_),
          scale_m(scale_m_),
          scale_n(scale_n_)
    {
    }
    index_t group_count;
    index_t* M_indices;
    index_t M;
    index_t N;
    index_t K;
    const void* a_ptr;
    index_t stride_A;
    const void* b_shuffle_ptr;
    index_t stride_B;
    const std::array<const void*, NumDTensor> ds_ptr;
    const std::array<index_t, NumDTensor> stride_Ds;
    union
    {
        void* e_ptr;
        void* c_ptr;
    };
    index_t stride_C;
    index_t k_batch;
    ScaleM scale_m = nullptr;
    ScaleN scale_n = nullptr;
};

template <class ScaleM       = FlatmmScalePointer<-1>,
          class ScaleN       = FlatmmScalePointer<-1>,
          index_t NumDTensor = 0>
struct MaskedGroupedFlatmmHostArgs
{
    CK_TILE_HOST MaskedGroupedFlatmmHostArgs() = default;
    CK_TILE_HOST MaskedGroupedFlatmmHostArgs(index_t* M_indices_,
                                             index_t group_count_,
                                             index_t Max_M_,
                                             index_t N_,
                                             index_t K_,
                                             const void* a_ptr_,
                                             index_t stride_A_,
                                             const void* b_shuffle_ptr_,
                                             index_t stride_B_,
                                             const std::array<const void*, NumDTensor>& ds_ptr_,
                                             const std::array<index_t, NumDTensor>& stride_Ds_,
                                             void* c_ptr_,
                                             index_t stride_C_,
                                             index_t k_batch_,
                                             ScaleM scale_m_ = nullptr,
                                             ScaleN scale_n_ = nullptr)
        : M_indices(M_indices_),
          group_count(group_count_),
          M(Max_M_),
          N(N_),
          K(K_),
          a_ptr(a_ptr_),
          stride_A(stride_A_),
          b_shuffle_ptr(b_shuffle_ptr_),
          stride_B(stride_B_),
          ds_ptr(ds_ptr_),
          stride_Ds(stride_Ds_),
          c_ptr(c_ptr_),
          stride_C(stride_C_),
          k_batch(k_batch_),
          scale_m(scale_m_),
          scale_n(scale_n_)
    {
    }

    index_t* M_indices;
    index_t group_count;
    index_t M;
    index_t N;
    index_t K;
    const void* a_ptr;
    index_t stride_A;
    const void* b_shuffle_ptr;
    index_t stride_B;
    const std::array<const void*, NumDTensor> ds_ptr;
    const std::array<index_t, NumDTensor> stride_Ds;
    union
    {
        void* e_ptr;
        void* c_ptr;
    };
    index_t stride_C;
    index_t k_batch;
    ScaleM scale_m = nullptr;
    ScaleN scale_n = nullptr;
};

template <typename TilePartitioner_, typename FlatmmPipeline_, typename EpiloguePipeline_>
struct GroupedFlatmmKernel : FlatmmKernel<TilePartitioner_, FlatmmPipeline_, EpiloguePipeline_>
{
    using UnderlyingGemmKernel = FlatmmKernel<TilePartitioner_, FlatmmPipeline_, EpiloguePipeline_>;
    using BlockGemmShape       = typename UnderlyingGemmKernel::BlockGemmShape;

    using TilePartitioner = remove_cvref_t<TilePartitioner_>;
    using FlatmmPipeline  = remove_cvref_t<FlatmmPipeline_>;

    using EpiloguePipeline = remove_cvref_t<EpiloguePipeline_>;

    using ADataType = remove_cvref_t<typename FlatmmPipeline::ADataType>;
    using BDataType = remove_cvref_t<typename FlatmmPipeline::BDataType>;
    // Below type is actually accumulation data type - the output of block GEMM.
    using CDataType  = remove_cvref_t<typename EpiloguePipeline::ODataType>;
    using DsLayout   = remove_cvref_t<typename EpiloguePipeline::DsLayout>;
    using DsDataType = remove_cvref_t<typename EpiloguePipeline::DsDataType>;

    static constexpr index_t NumDTensor = DsDataType::size();
    static constexpr index_t kBlockSize = FlatmmPipeline_::BlockSize;

    static constexpr auto I0 = number<0>();
    static constexpr auto I1 = number<1>();
    static constexpr auto I2 = number<2>();
    static constexpr auto I3 = number<3>();

    static_assert(DsLayout::size() == DsDataType::size(),
                  "The size of DsLayout and DsDataType should be the same");

    CK_TILE_HOST static const std::string GetName()
    {
        return concat(
            '_', "grouped_flatmm", gemm_prec_str<ADataType, BDataType>, FlatmmPipeline::GetName());
    }

    template <class ScaleM       = FlatmmScalePointer<-1>,
              class ScaleN       = FlatmmScalePointer<-1>,
              index_t NumDTensor = 0>
    CK_TILE_HOST_DEVICE static auto
    GridSize([[maybe_unused]] const GroupedFlatmmHostArgs<ScaleM, ScaleN, NumDTensor>& kernelArgs)
    {
        hipDeviceProp_t prop;
        int deviceId = 0; // default device

        constexpr int block_size = UnderlyingGemmKernel::BlockSize().x;
        int dync_smem_size       = 0;
        int maxActiveBlocksPerCU;

        [[maybe_unused]] auto e = hipGetDeviceProperties(&prop, deviceId);

        e = hipOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxActiveBlocksPerCU,
            reinterpret_cast<void*>(
                kentry<1, GroupedFlatmmKernel, GroupedFlatmmHostArgs<ScaleM, ScaleN, NumDTensor>>),
            block_size,
            dync_smem_size);

        const int persistent_block_size = prop.multiProcessorCount * maxActiveBlocksPerCU;

        // std::cout << "maxActiveBlocksPerCU: " << maxActiveBlocksPerCU
        //           << ", persistent_block_size: " << persistent_block_size << std::endl;

        assert(kernelArgs.k_batch == 1);
        return dim3(persistent_block_size, 1, kernelArgs.k_batch);
    }

    template <class ScaleM       = FlatmmScalePointer<-1>,
              class ScaleN       = FlatmmScalePointer<-1>,
              index_t NumDTensor = 0>
    CK_TILE_HOST_DEVICE static auto
    GridSize([[maybe_unused]] const ContiguousGroupedFlatmmHostArgs<ScaleM, ScaleN, NumDTensor>&
                 kernelArgs)
    {
        hipDeviceProp_t prop;
        int deviceId = 0; // default device

        constexpr int block_size = UnderlyingGemmKernel::BlockSize().x;
        int dync_smem_size       = 0;
        int maxActiveBlocksPerCU;

        [[maybe_unused]] auto e = hipGetDeviceProperties(&prop, deviceId);

        e = hipOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxActiveBlocksPerCU,
            reinterpret_cast<void*>(
                kentry<1,
                       GroupedFlatmmKernel,
                       ContiguousGroupedFlatmmHostArgs<ScaleM, ScaleN, NumDTensor>>),
            block_size,
            dync_smem_size);

        const int persistent_block_size = prop.multiProcessorCount * maxActiveBlocksPerCU;
        const int total_work_tile_cnt   = TilePartitioner::GridSize(kernelArgs.M, kernelArgs.N);

        // std::cout << "maxActiveBlocksPerCU: " << maxActiveBlocksPerCU
        //           << ", persistent_block_size: " << persistent_block_size
        //           << ", total_work_tile_cnt: " << total_work_tile_cnt << std::endl;

        assert(kernelArgs.k_batch == 1);
        return dim3(min(persistent_block_size, total_work_tile_cnt), 1, kernelArgs.k_batch);
    }

    template <class ScaleM       = FlatmmScalePointer<-1>,
              class ScaleN       = FlatmmScalePointer<-1>,
              index_t NumDTensor = 0>
    CK_TILE_HOST_DEVICE static auto GridSize(
        [[maybe_unused]] const MaskedGroupedFlatmmHostArgs<ScaleM, ScaleN, NumDTensor>& kernelArgs)
    {
        hipDeviceProp_t prop;
        int deviceId = 0; // default device

        constexpr int block_size = UnderlyingGemmKernel::BlockSize().x;
        int dync_smem_size       = 0;
        int maxActiveBlocksPerCU;

        [[maybe_unused]] auto e = hipGetDeviceProperties(&prop, deviceId);

        e = hipOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxActiveBlocksPerCU,
            reinterpret_cast<void*>(
                kentry<1,
                       GroupedFlatmmKernel,
                       MaskedGroupedFlatmmHostArgs<ScaleM, ScaleN, NumDTensor>>),
            block_size,
            dync_smem_size);

        const int persistent_block_size = prop.multiProcessorCount * maxActiveBlocksPerCU;
        // const int total_work_tile_cnt   = TilePartitioner::GridSize(kernelArgs.M, kernelArgs.N);

        // std::cout << "maxActiveBlocksPerCU: " << maxActiveBlocksPerCU
        //           << ", persistent_block_size: " << persistent_block_size << std::endl;

        assert(kernelArgs.k_batch == 1);
        return dim3(persistent_block_size, 1, kernelArgs.k_batch);
    }

    template <typename HostArgs>
    CK_TILE_HOST static constexpr auto MakeKernelArgs(const HostArgs& hostArgs)
    {
        return hostArgs;
    }
    // CK_TILE_HOST static constexpr auto
    // MakeKernelArgs(const ContiguousGroupedFlatmmHostArgs& hostArgs)
    // {
    //     return hostArgs;
    // }
    // CK_TILE_HOST static constexpr auto
    // MakeKernelArgs(const MaskedGroupedFlatmmHostArgs& hostArgs)
    // {
    //     return hostArgs;
    // }

    template <class ScaleM       = FlatmmScalePointer<-1>,
              class ScaleN       = FlatmmScalePointer<-1>,
              index_t NumDTensor = 0>
    CK_TILE_DEVICE void operator()(GroupedFlatmmHostArgs<ScaleM, ScaleN, NumDTensor> kargs) const
    {
        int group_idx        = 0;
        int block_linear_idx = blockIdx.x;
        int total_block_cnt  = gridDim.x;

        UnderlyingGemmKernel underlying_kernel{};
        for(; group_idx < kargs.group_count; ++group_idx)
        {
            const index_t M               = kargs.M[group_idx];
            const index_t N               = kargs.N[group_idx];
            const index_t group_block_cnt = TilePartitioner::GridSize(M, N);

            while(block_linear_idx < group_block_cnt)
            {
                // Found the group this block belongs to
                // create the kernel args for the underlying flatmm kernel
                FlatmmKernelArgs<ScaleM, ScaleN, NumDTensor> impl_kargs{
                    kargs.a_ptr[group_idx],
                    kargs.b_shuffle_ptr[group_idx],
                    kargs.ds_ptr,
                    kargs.c_ptr[group_idx],
                    kargs.M[group_idx],
                    kargs.N[group_idx],
                    kargs.K[group_idx],
                    kargs.stride_A[group_idx],
                    kargs.stride_B[group_idx],
                    kargs.stride_Ds,
                    kargs.stride_C[group_idx],
                    kargs.k_batch,
                    kargs.scale_m[group_idx],
                    kargs.scale_n[group_idx]};
                // call the underlying flatmm kernel
                underlying_kernel(impl_kargs, block_linear_idx);
                block_linear_idx += total_block_cnt;
            }
            block_linear_idx -= group_block_cnt;
        }
    }

    template <class ScaleM       = FlatmmScalePointer<-1>,
              class ScaleN       = FlatmmScalePointer<-1>,
              index_t NumDTensor = 0>
    CK_TILE_DEVICE void
    operator()(ContiguousGroupedFlatmmHostArgs<ScaleM, ScaleN, NumDTensor> kargs) const
    {
        int block_linear_idx    = blockIdx.x;
        int total_block_cnt     = gridDim.x;
        int total_work_tile_cnt = TilePartitioner::GridSize(kargs.M, kargs.N);

        UnderlyingGemmKernel underlying_kernel{};
        for(; block_linear_idx < total_work_tile_cnt; block_linear_idx += total_block_cnt)
        {
            auto [block_m_idx, block_n_idx] =
                TilePartitioner{kargs.M, kargs.N}.GetOutputTileIndex(block_linear_idx);
            // get the group index from the M_indices
            int group_idx = kargs.M_indices[block_m_idx * BlockGemmShape::kM];

            FlatmmKernelArgs<ScaleM, ScaleN, NumDTensor> impl_kargs{
                kargs.a_ptr,
                static_cast<const BDataType*>(kargs.b_shuffle_ptr) + group_idx * kargs.N * kargs.K,
                kargs.ds_ptr,
                kargs.c_ptr,
                kargs.M,
                kargs.N,
                kargs.K,
                kargs.stride_A,
                kargs.stride_B,
                kargs.stride_Ds,
                kargs.stride_C,
                kargs.k_batch,
                kargs.scale_m,
                kargs.scale_n};
            // call the underlying flatmm kernel
            underlying_kernel(impl_kargs, block_linear_idx);
        }
    }

    template <class ScaleM       = FlatmmScalePointer<-1>,
              class ScaleN       = FlatmmScalePointer<-1>,
              index_t NumDTensor = 0>
    CK_TILE_DEVICE void
    operator()(MaskedGroupedFlatmmHostArgs<ScaleM, ScaleN, NumDTensor> kargs) const
    {
        int group_idx        = 0;
        int block_linear_idx = blockIdx.x;
        int total_block_cnt  = gridDim.x;

        UnderlyingGemmKernel underlying_kernel{};
        for(; group_idx < kargs.group_count; ++group_idx)
        {
            const index_t valid_M         = kargs.M_indices[group_idx];
            const index_t N               = kargs.N;
            const index_t group_block_cnt = TilePartitioner::GridSize(valid_M, N);

            while(block_linear_idx < group_block_cnt)
            {
                // Found the group this block belongs to
                // create the kernel args for the underlying flatmm kernel
                FlatmmKernelArgs<ScaleM, ScaleN, NumDTensor> impl_kargs{
                    static_cast<const ADataType*>(kargs.a_ptr) + group_idx * kargs.M * kargs.K,
                    static_cast<const BDataType*>(kargs.b_shuffle_ptr) +
                        group_idx * kargs.N * kargs.K,
                    kargs.ds_ptr,
                    static_cast<CDataType*>(kargs.c_ptr) + group_idx * kargs.M * kargs.N,
                    valid_M,
                    kargs.N,
                    kargs.K,
                    kargs.stride_A,
                    kargs.stride_B,
                    kargs.stride_Ds,
                    kargs.stride_C,
                    kargs.k_batch,
                    kargs.scale_m + group_idx * kargs.M,
                    kargs.scale_n + group_idx * kargs.N};
                // call the underlying flatmm kernel
                underlying_kernel(impl_kargs, block_linear_idx);
                block_linear_idx += total_block_cnt;
            }
            block_linear_idx -= group_block_cnt;
        }
    }
};

} // namespace ck_tile
