// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <sstream>

#include "ck/utility/common_header.hpp"
#include "ck/utility/env.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_gemm.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/matrix_padder.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_dl_multiple_d.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename GridwiseGemm,
          typename GemmDesc,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
          bool HasMainKBlockLoop,
          bool HasDoubleTailKBlockLoop>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
    kernel_grouped_gemm_multiple_d_dl(const void CK_CONSTANT_ADDRESS_SPACE* gemm_descs_const,
                                      const index_t group_count,
                                      const AElementwiseOperation a_element_op,
                                      const BElementwiseOperation b_element_op,
                                      const CDEElementwiseOperation cde_element_op)
{
#if(defined(__gfx906__) || defined(__gfx908__) || defined(__gfx90a__) || defined(__gfx101__) || \
    defined(__gfx103__) || defined(__gfx11__) || defined(__gfx94__) || defined(__gfx12__))
    __shared__ char p_shared[GridwiseGemm::GetSharedMemoryNumberOfByte()];

    const index_t block_id = get_block_1d_id();

    const auto gemm_desc_ptr =
        reinterpret_cast<const GemmDesc*>(cast_pointer_to_generic_address_space(gemm_descs_const));

    index_t left     = 0;
    index_t right    = group_count;
    index_t group_id = index_t((left + right) / 2);
    while((!(block_id >= gemm_desc_ptr[group_id].BlockStart_ &&
             block_id < gemm_desc_ptr[group_id].BlockEnd_)) &&
          left <= right)
    {
        if(block_id < gemm_desc_ptr[group_id].BlockStart_)
        {
            right = group_id;
        }
        else
        {
            left = group_id;
        }
        group_id = index_t((left + right) / 2);
    }

    GridwiseGemm::Run(gemm_desc_ptr[group_id].a_ptr_,
                      gemm_desc_ptr[group_id].b_ptr_,
                      gemm_desc_ptr[group_id].ds_ptr_,
                      gemm_desc_ptr[group_id].e_ptr_,
                      p_shared,
                      a_element_op,
                      b_element_op,
                      cde_element_op,
                      gemm_desc_ptr[group_id].a_grid_desc_k0_m0_m1_k1_,
                      gemm_desc_ptr[group_id].b_grid_desc_k0_n0_n1_k1_,
                      gemm_desc_ptr[group_id].ds_grid_desc_m0_m10_m11_n0_n10_n11_,
                      gemm_desc_ptr[group_id].e_grid_desc_m0_m10_m11_n0_n10_n11_,
                      gemm_desc_ptr[group_id].block_2_etile_map_,
                      integral_constant<bool, HasMainKBlockLoop>{},
                      integral_constant<bool, HasDoubleTailKBlockLoop>{});
#else
    ignore = gemm_descs_const;
    ignore = group_count;
    ignore = a_element_op;
    ignore = b_element_op;
    ignore = cde_element_op;
#endif
}

template <typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename DsDataType,
          typename EDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
          GemmSpecialization GemmSpec,
          index_t BlockSize,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t K0PerBlock,
          index_t K1,
          index_t M1PerThread,
          index_t N1PerThread,
          index_t KPerThread,
          typename M1N1ThreadClusterM1Xs,
          typename M1N1ThreadClusterN1Xs,
          typename ABlockTransferThreadSliceLengths_K0_M0_M1_K1,
          typename ABlockTransferThreadClusterLengths_K0_M0_M1_K1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          typename ABlockTransferSrcVectorTensorLengths_K0_M0_M1_K1,
          typename ABlockTransferSrcVectorTensorContiguousDimOrder,
          typename ABlockTransferDstVectorTensorLengths_K0_M0_M1_K1,
          typename BBlockTransferThreadSliceLengths_K0_N0_N1_K1,
          typename BBlockTransferThreadClusterLengths_K0_N0_N1_K1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          typename BBlockTransferSrcVectorTensorLengths_K0_N0_N1_K1,
          typename BBlockTransferSrcVectorTensorContiguousDimOrder,
          typename BBlockTransferDstVectorTensorLengths_K0_N0_N1_K1,
          typename CThreadTransferSrcDstAccessOrder,
          index_t CThreadTransferSrcDstVectorDim,
          index_t CThreadTransferDstScalarPerVector,
          enable_if_t<
              is_same_v<AElementwiseOperation, ck::tensor_operation::element_wise::PassThrough> &&
                  is_same_v<BElementwiseOperation, ck::tensor_operation::element_wise::PassThrough>,
              bool> = false>
struct DeviceGroupedGemmMultipleD_Dl : public DeviceGroupedGemm<ALayout,
                                                                BLayout,
                                                                DsLayout,
                                                                ELayout,
                                                                ADataType,
                                                                BDataType,
                                                                DsDataType,
                                                                EDataType,
                                                                AElementwiseOperation,
                                                                BElementwiseOperation,
                                                                CDEElementwiseOperation>
{
    using DeviceOp                      = DeviceGroupedGemmMultipleD_Dl;
    static constexpr index_t NumDTensor = DsDataType::Size();

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};

    static constexpr auto K1Number = Number<K1>{};

    static auto MakeAGridDescriptor_K0_M_K1(index_t M, index_t K, index_t StrideA)
    {
        assert(K % K1 == 0);

        const index_t K0 = K / K1;

        const auto a_grid_desc_m_k = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, ALayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(M, K), make_tuple(StrideA, I1));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, ALayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(M, K), make_tuple(I1, StrideA));
            }
        }();

        if constexpr(GemmSpec == GemmSpecialization::MNPadding)
        {
            const auto PadM = (MPerBlock - M % MPerBlock) % MPerBlock;

            return transform_tensor_descriptor(
                a_grid_desc_m_k,
                make_tuple(make_unmerge_transform(make_tuple(K0, K1Number)),
                           make_right_pad_transform(M, PadM)),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
        }
        else
        {
            return transform_tensor_descriptor(
                a_grid_desc_m_k,
                make_tuple(make_unmerge_transform(make_tuple(K0, K1Number)),
                           make_pass_through_transform(M)),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
        }
    }

    static auto MakeBGridDescriptor_K0_N_K1(index_t K, index_t N, index_t StrideB)
    {
        assert(K % K1 == 0);

        const index_t K0 = K / K1;

        const auto b_grid_desc_k_n = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, BLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(K, N), make_tuple(StrideB, I1));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, BLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(K, N), make_tuple(I1, StrideB));
            }
        }();

        if constexpr(GemmSpec == GemmSpecialization::MNPadding)
        {
            const auto PadN = (NPerBlock - N % NPerBlock) % NPerBlock;

            return transform_tensor_descriptor(
                b_grid_desc_k_n,
                make_tuple(make_unmerge_transform(make_tuple(K0, K1Number)),
                           make_right_pad_transform(N, PadN)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
        }
        else
        {
            return transform_tensor_descriptor(
                b_grid_desc_k_n,
                make_tuple(make_unmerge_transform(make_tuple(K0, K1Number)),
                           make_pass_through_transform(N)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
        }
    }

    template <typename ELay>
    static auto MakeEGridDescriptor_M_N(index_t M, index_t N, index_t StrideE)
    {
        const auto c_grid_desc_m_n = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, ELay>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(M, N), make_tuple(StrideE, I1));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, ELay>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(M, N), make_tuple(I1, StrideE));
            }
        }();

        if constexpr(GemmSpec == GemmSpecialization::MNPadding)
        {
            const auto PadM = (MPerBlock - M % MPerBlock) % MPerBlock;
            const auto PadN = (NPerBlock - N % NPerBlock) % NPerBlock;

            return transform_tensor_descriptor(
                c_grid_desc_m_n,
                make_tuple(make_right_pad_transform(M, PadM), make_right_pad_transform(N, PadN)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));
        }
        else
        {

            return transform_tensor_descriptor(
                c_grid_desc_m_n,
                make_tuple(make_pass_through_transform(M), make_pass_through_transform(N)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));
        }
    }

    static auto MakeDsGridDescriptor_M_N(const std::array<index_t, NumDTensor>& MRaws,
                                         const std::array<index_t, NumDTensor>& NRaws,
                                         const std::array<index_t, NumDTensor>& DsStride)
    {
        return generate_tuple(
            [&](auto i) {
                using DLayout = remove_cvref_t<tuple_element_t<i.value, DsLayout>>;

                return DeviceOp::MakeEGridDescriptor_M_N<DLayout>(MRaws[i], NRaws[i], DsStride[i]);
            },
            Number<NumDTensor>{});
    }

    using AGridDesc_K0_M_K1 = decltype(MakeAGridDescriptor_K0_M_K1(1, 1, 1));
    using BGridDesc_K0_N_K1 = decltype(MakeBGridDescriptor_K0_N_K1(1, 1, 1));
    using DsGridDesc_M_N    = decltype(MakeDsGridDescriptor_M_N({}, {}, {}));
    using EGridDesc_M_N     = decltype(MakeEGridDescriptor_M_N<ELayout>(1, 1, 1));

    // GridwiseGemm
    using GridwiseGemm =
        GridwiseGemmDlMultipleD_km_kn_mn<BlockSize,
                                         ADataType,
                                         AccDataType,
                                         DsDataType,
                                         EDataType,
                                         AElementwiseOperation,
                                         BElementwiseOperation,
                                         CDEElementwiseOperation,
                                         InMemoryDataOperationEnum::Set,
                                         AGridDesc_K0_M_K1,
                                         BGridDesc_K0_N_K1,
                                         EGridDesc_M_N,
                                         MPerBlock,
                                         NPerBlock,
                                         K0PerBlock,
                                         K1,
                                         M1PerThread,
                                         N1PerThread,
                                         KPerThread,
                                         M1N1ThreadClusterM1Xs,
                                         M1N1ThreadClusterN1Xs,
                                         ABlockTransferThreadSliceLengths_K0_M0_M1_K1,
                                         ABlockTransferThreadClusterLengths_K0_M0_M1_K1,
                                         ABlockTransferThreadClusterArrangeOrder,
                                         ABlockTransferSrcAccessOrder,
                                         ABlockTransferSrcVectorTensorLengths_K0_M0_M1_K1,
                                         ABlockTransferSrcVectorTensorContiguousDimOrder,
                                         ABlockTransferDstVectorTensorLengths_K0_M0_M1_K1,
                                         BBlockTransferThreadSliceLengths_K0_N0_N1_K1,
                                         BBlockTransferThreadClusterLengths_K0_N0_N1_K1,
                                         BBlockTransferThreadClusterArrangeOrder,
                                         BBlockTransferSrcAccessOrder,
                                         BBlockTransferSrcVectorTensorLengths_K0_N0_N1_K1,
                                         BBlockTransferSrcVectorTensorContiguousDimOrder,
                                         BBlockTransferDstVectorTensorLengths_K0_N0_N1_K1,
                                         CThreadTransferSrcDstAccessOrder,
                                         CThreadTransferSrcDstVectorDim,
                                         CThreadTransferDstScalarPerVector>;

    using AGridDesc_K0_M0_M1_K1 =
        decltype(GridwiseGemm::MakeAGridDescriptor_K0_M0_M1_K1(AGridDesc_K0_M_K1{}));
    using BGridDesc_K0_N0_N1_K1 =
        decltype(GridwiseGemm::MakeBGridDescriptor_K0_N0_N1_K1(BGridDesc_K0_N_K1{}));
    using DsGridDesc_M0_M10_M11_N0_N10_N11 =
        decltype(GridwiseGemm::MakeDsGridDescriptor_M0_M10_M11_N0_N10_N11(DsGridDesc_M_N{}));
    using EGridDesc_M0_M10_M11_N0_N10_N11 =
        decltype(GridwiseGemm::MakeCGridDescriptor_M0_M10_M11_N0_N10_N11(EGridDesc_M_N{}));

    struct GroupedGemmBlock2ETileMap
    {
        using Block2ETileMap =
            remove_cvref_t<decltype(GridwiseGemm::MakeDefaultBlock2CTileMap(EGridDesc_M_N{}))>;

        GroupedGemmBlock2ETileMap()
        {
            block_2_etile_map_ = GridwiseGemm::MakeDefaultBlock2CTileMap(EGridDesc_M_N{});
            BlockStart_        = -1;
        }

        GroupedGemmBlock2ETileMap(const EGridDesc_M_N& e_grid_desc_m_n, ck::index_t BlockStart)
        {
            block_2_etile_map_ = GridwiseGemm::MakeDefaultBlock2CTileMap(e_grid_desc_m_n);
            BlockStart_        = BlockStart;
        }

        template <typename TopIdx>
        __host__ __device__ constexpr auto CalculateBottomIndex(const TopIdx& idx_top) const
        {
            return block_2_etile_map_.CalculateBottomIndex(
                make_multi_index(idx_top[I0] - BlockStart_));
        }

        // it's actually E-Tile
        template <typename CTileIdx, typename CTileDim>
        __host__ __device__ bool ValidCTileIndex(const CTileIdx& c_tile_idx,
                                                 const CTileDim& c_tile_dim) const
        {
            return block_2_etile_map_.ValidCTileIndex(c_tile_idx, c_tile_dim);
        }

        __host__ bool CheckValidity(const EGridDesc_M_N& e_grid_desc_m_n) const
        {
            return block_2_etile_map_.CheckValidity(e_grid_desc_m_n);
        }

        Block2ETileMap block_2_etile_map_;
        ck::index_t BlockStart_;
    };

    struct GemmKernelArg
    {
        // pointers
        const ADataType* a_ptr_;
        const BDataType* b_ptr_;
        typename GridwiseGemm::DsGridPointer ds_ptr_;
        EDataType* e_ptr_;

        // tensor descriptors for problem definiton
        AGridDesc_K0_M_K1 a_grid_desc_k0_m_k1_;
        BGridDesc_K0_N_K1 b_grid_desc_k0_n_k1_;
        DsGridDesc_M_N ds_grid_desc_m_n_;
        EGridDesc_M_N e_grid_desc_m_n_;

        // tensor descriptors for block/thread-wise copy
        AGridDesc_K0_M0_M1_K1 a_grid_desc_k0_m0_m1_k1_;
        BGridDesc_K0_N0_N1_K1 b_grid_desc_k0_n0_n1_k1_;
        DsGridDesc_M0_M10_M11_N0_N10_N11 ds_grid_desc_m0_m10_m11_n0_n10_n11_;
        EGridDesc_M0_M10_M11_N0_N10_N11 e_grid_desc_m0_m10_m11_n0_n10_n11_;

        // block-to-e-tile map
        GroupedGemmBlock2ETileMap block_2_etile_map_;
        ck::index_t BlockStart_, BlockEnd_;
    };

    // Argument
    struct Argument : public BaseArgument
    {
        Argument(std::vector<const void*>& p_As,
                 std::vector<const void*>& p_Bs,
                 std::vector<std::array<const void*, NumDTensor>>& p_Ds,
                 std::vector<void*>& p_Es,
                 std::vector<GemmDesc>& gemm_descs,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 CDEElementwiseOperation cde_element_op)
            : a_element_op_{a_element_op},
              b_element_op_{b_element_op},
              cde_element_op_{cde_element_op},
              gemm_kernel_host_args_{nullptr}
        {
            grid_size_ = 0;

            group_count_ = ck::type_convert<ck::index_t>(gemm_descs.size());

            if(!(group_count_ == ck::type_convert<ck::index_t>(p_As.size()) &&
                 group_count_ == ck::type_convert<ck::index_t>(p_Bs.size()) &&
                 group_count_ == ck::type_convert<ck::index_t>(p_Es.size())))
            {
                throw std::runtime_error("wrong! group_count_ != p_As/b/c.size");
            }

            gemm_desc_kernel_arg_.reserve(group_count_);

            skipped_group_count_ = 0;

            for(std::size_t i = 0; i < gemm_descs.size(); i++)
            {
                const index_t M = gemm_descs[i].M_;
                const index_t N = gemm_descs[i].N_;
                const index_t K = gemm_descs[i].K_;

                a_mtx_mraw_kraw_.emplace_back(M, K);
                b_mtx_nraw_kraw_.emplace_back(N, K);

                if(M == 0)
                {
                    skipped_group_count_++;
                    continue;
                }

                const index_t StrideA = gemm_descs[i].stride_A_;
                const index_t StrideB = gemm_descs[i].stride_B_;
                const index_t StrideE = gemm_descs[i].stride_C_;

                typename GridwiseGemm::DsGridPointer p_ds_grid{};
                DsGridDesc_M_N ds_grid_desc_m_n;

                static_for<0, NumDTensor, 1>{}([&](auto j) {
                    using DLayout   = remove_cvref_t<tuple_element_t<j.value, DsLayout>>;
                    using DDataType = remove_cvref_t<tuple_element_t<j.value, DsDataType>>;

                    p_ds_grid(j)        = static_cast<const DDataType*>(p_Ds[i][j]);
                    ds_grid_desc_m_n(j) = DeviceOp::MakeEGridDescriptor_M_N<DLayout>(
                        M, N, gemm_descs[i].stride_Ds_[j]);
                });

                // tensor descriptors for problem definiton
                const auto a_grid_desc_k0_m_k1 =
                    DeviceOp::MakeAGridDescriptor_K0_M_K1(M, K, StrideA);
                const auto b_grid_desc_k0_n_k1 =
                    DeviceOp::MakeBGridDescriptor_K0_N_K1(K, N, StrideB);
                const auto e_grid_desc_m_n =
                    DeviceOp::MakeEGridDescriptor_M_N<ELayout>(M, N, StrideE);

                if(GridwiseGemm::CheckValidity(
                       a_grid_desc_k0_m_k1, b_grid_desc_k0_n_k1, e_grid_desc_m_n))
                {

                    const index_t grid_size_grp =
                        GroupedGemmBlock2ETileMap(e_grid_desc_m_n, 0)
                            .block_2_etile_map_.CalculateGridSize(e_grid_desc_m_n);

                    const index_t BlockStart = grid_size_;
                    const index_t BlockEnd   = grid_size_ + grid_size_grp;

                    grid_size_ += grid_size_grp;

                    // block-to-e-tile map
                    const auto block_2_etile_map =
                        GroupedGemmBlock2ETileMap(e_grid_desc_m_n, BlockStart);

                    // tensor descriptors for block/thread-wise copy
                    const auto a_grid_desc_k0_m0_m1_k1 =
                        GridwiseGemm::MakeAGridDescriptor_K0_M0_M1_K1(a_grid_desc_k0_m_k1);
                    const auto b_grid_desc_k0_n0_n1_k1 =
                        GridwiseGemm::MakeBGridDescriptor_K0_N0_N1_K1(b_grid_desc_k0_n_k1);
                    const auto ds_grid_desc_m0_m10_m11_n0_n10_n11 =
                        GridwiseGemm::MakeDsGridDescriptor_M0_M10_M11_N0_N10_N11(ds_grid_desc_m_n);
                    const auto e_grid_desc_m0_m10_m11_n0_n10_n11 =
                        GridwiseGemm::MakeCGridDescriptor_M0_M10_M11_N0_N10_N11(e_grid_desc_m_n);

                    gemm_desc_kernel_arg_.push_back(
                        GemmKernelArg{static_cast<const ADataType*>(p_As[i]),
                                      static_cast<const BDataType*>(p_Bs[i]),
                                      p_ds_grid,
                                      static_cast<EDataType*>(p_Es[i]),
                                      a_grid_desc_k0_m_k1,
                                      b_grid_desc_k0_n_k1,
                                      ds_grid_desc_m_n,
                                      e_grid_desc_m_n,
                                      a_grid_desc_k0_m0_m1_k1,
                                      b_grid_desc_k0_n0_n1_k1,
                                      ds_grid_desc_m0_m10_m11_n0_n10_n11,
                                      e_grid_desc_m0_m10_m11_n0_n10_n11,
                                      block_2_etile_map,
                                      BlockStart,
                                      BlockEnd});
                }
            }
        }

        //  private:
        index_t group_count_;
        index_t skipped_group_count_;

        // TODO: A,B element op is unused since gridwise_gemm_dl_v1r3 does NOT support prologue
        //       for the time being.
        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        CDEElementwiseOperation cde_element_op_;

        std::vector<GemmKernelArg> gemm_desc_kernel_arg_;
        std::vector<Tuple<index_t, index_t>> a_mtx_mraw_kraw_;
        std::vector<Tuple<index_t, index_t>> b_mtx_nraw_kraw_;

        index_t grid_size_;
        void* gemm_kernel_host_args_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceOp::Argument;

        float Run(const Argument& arg,
                  const StreamConfig& stream_config = StreamConfig{},
                  hipStream_t cpy_stream            = nullptr,
                  hipEvent_t cpy_event              = nullptr)
        {
            auto K0 = arg.gemm_desc_kernel_arg_[0].a_grid_desc_k0_m_k1_.GetLength(I0);
            bool all_has_main_k_block_loop = GridwiseGemm::CalculateHasMainKBlockLoop(K0);
            bool all_has_double_tail_k_block_loop =
                GridwiseGemm::CalculateHasDoubleTailKBlockLoop(K0);

            for(std::size_t i = 0; i < arg.gemm_desc_kernel_arg_.size(); i++)
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "group: " << i << " arg.a_grid_desc_k0_m_k1_{"
                              << arg.gemm_desc_kernel_arg_[i].a_grid_desc_k0_m_k1_.GetLength(I0)
                              << ", "
                              << arg.gemm_desc_kernel_arg_[i].a_grid_desc_k0_m_k1_.GetLength(I1)
                              << ", "
                              << arg.gemm_desc_kernel_arg_[i].a_grid_desc_k0_m_k1_.GetLength(I2)
                              << "}" << std::endl;

                    std::cout << ", arg.b_grid_desc_k0_n_k1_{"
                              << arg.gemm_desc_kernel_arg_[i].b_grid_desc_k0_n_k1_.GetLength(I0)
                              << ", "
                              << arg.gemm_desc_kernel_arg_[i].b_grid_desc_k0_n_k1_.GetLength(I1)
                              << ", "
                              << arg.gemm_desc_kernel_arg_[i].b_grid_desc_k0_n_k1_.GetLength(I2)
                              << "}" << std::endl;

                    std::cout << ", arg.e_grid_desc_m_n_{ "
                              << arg.gemm_desc_kernel_arg_[i].e_grid_desc_m_n_.GetLength(I0) << ", "
                              << arg.gemm_desc_kernel_arg_[i].e_grid_desc_m_n_.GetLength(I1) << "}"
                              << std::endl;
                }

                if(!GridwiseGemm::CheckValidity(arg.gemm_desc_kernel_arg_[i].a_grid_desc_k0_m_k1_,
                                                arg.gemm_desc_kernel_arg_[i].b_grid_desc_k0_n_k1_,
                                                arg.gemm_desc_kernel_arg_[i].e_grid_desc_m_n_))
                {
                    throw std::runtime_error(
                        "wrong! GridwiseGemmDlMultipleD_km_kn_mn has invalid setting");
                }

                K0 = arg.gemm_desc_kernel_arg_[i].a_grid_desc_k0_m0_m1_k1_.GetLength(I0);
                bool not_all_has_main_k_block_loop_same =
                    all_has_main_k_block_loop xor GridwiseGemm::CalculateHasMainKBlockLoop(K0);
                bool not_all_has_double_tail_k_block_loop_same =
                    all_has_double_tail_k_block_loop xor
                    GridwiseGemm::CalculateHasDoubleTailKBlockLoop(K0);

                if(not_all_has_main_k_block_loop_same or not_all_has_double_tail_k_block_loop_same)
                {
                    std::ostringstream err;
                    err << "Not all gemms have same value for [main|double_tail]_k_block_loop! in "
                        << __FILE__ << ":" << __LINE__ << ", in function: " << __func__;
                    throw std::runtime_error(err.str());
                }
            }

            // If the user provides copy stream and copy event, we assume that they're also
            // responsible for providing allocated host memory (eg. pinned) which
            // would be used to copy kernel arguments to the device.
            if(cpy_stream && cpy_event)
            {
                if(arg.gemm_kernel_host_args_ == nullptr)
                {
                    std::ostringstream err;
                    err << "No memory has been allocated for gemm kernel host args "
                        << "when providing the copy stream and copy event! In " << __FILE__ << ":"
                        << __LINE__ << ", in function: " << __func__;
                    throw std::runtime_error(err.str());
                }
                hipGetErrorString(hipMemcpyAsync(arg.p_workspace_,
                                                 arg.gemm_kernel_host_args_,
                                                 arg.group_count_ * sizeof(GemmKernelArg),
                                                 hipMemcpyHostToDevice,
                                                 cpy_stream));
                hipGetErrorString(hipEventRecord(cpy_event, cpy_stream));
                hipGetErrorString(hipEventSynchronize(cpy_event));
            }
            else // In this case CK owns memory allocated on host.
            {
                hipGetErrorString(
                    hipMemcpyAsync(arg.p_workspace_,
                                   arg.gemm_desc_kernel_arg_.data(),
                                   arg.gemm_desc_kernel_arg_.size() * sizeof(GemmKernelArg),
                                   hipMemcpyHostToDevice,
                                   stream_config.stream_id_));
            }

            auto launch_kernel = [&](auto has_main_k_block_loop,
                                     auto has_double_tail_k_block_loop) {
                constexpr bool has_main_loop   = has_main_k_block_loop.value;
                constexpr bool has_double_loop = has_double_tail_k_block_loop.value;

                const auto kernel = kernel_grouped_gemm_multiple_d_dl<GridwiseGemm,
                                                                      GemmKernelArg,
                                                                      AElementwiseOperation,
                                                                      BElementwiseOperation,
                                                                      CDEElementwiseOperation,
                                                                      has_main_loop,
                                                                      has_double_loop>;

                return launch_and_time_kernel(
                    stream_config,
                    kernel,
                    dim3(arg.grid_size_),
                    dim3(BlockSize),
                    0,
                    cast_pointer_to_constant_address_space(arg.p_workspace_),
                    arg.gemm_desc_kernel_arg_.size(),
                    arg.a_element_op_,
                    arg.b_element_op_,
                    arg.cde_element_op_);
            };

            if(all_has_main_k_block_loop && all_has_double_tail_k_block_loop)
            {
                return launch_kernel(integral_constant<bool, true>{},
                                     integral_constant<bool, true>{});
            }
            else if(all_has_main_k_block_loop && !all_has_double_tail_k_block_loop)
            {
                return launch_kernel(integral_constant<bool, true>{},
                                     integral_constant<bool, false>{});
            }
            else if(!all_has_main_k_block_loop && all_has_double_tail_k_block_loop)
            {
                return launch_kernel(integral_constant<bool, false>{},
                                     integral_constant<bool, true>{});
            }
            else
            {
                return launch_kernel(integral_constant<bool, false>{},
                                     integral_constant<bool, false>{});
            }
        }

        // polymorphic
        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        }
    };

    static bool IsSupportedArgument(const Argument& arg)
    {
        if((ck::type_convert<ck::index_t>(arg.gemm_desc_kernel_arg_.size()) +
            arg.skipped_group_count_) != arg.group_count_)
        {
            return false;
        }

        if(ck::get_device_name() == "gfx906" || ck::is_xdl_supported() ||
           ck::is_gfx103_supported() || ck::is_gfx11_supported() || ck::is_gfx12_supported())
        {
            for(std::size_t i = 0; i < arg.gemm_desc_kernel_arg_.size(); i++)
            {
                if(!GridwiseGemm::CheckValidity(arg.gemm_desc_kernel_arg_[i].a_grid_desc_k0_m_k1_,
                                                arg.gemm_desc_kernel_arg_[i].b_grid_desc_k0_n_k1_,
                                                arg.gemm_desc_kernel_arg_[i].e_grid_desc_m_n_))
                {
                    return false;
                }
            }
            return true;
        }
        else
        {
            return false;
        }
    }

    // polymorphic
    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto MakeArgument(std::vector<const void*>& p_As,
                             std::vector<const void*>& p_Bs,
                             std::vector<std::array<const void*, NumDTensor>>& p_Ds,
                             std::vector<void*>& p_Es,
                             std::vector<GemmDesc> gemm_descs,
                             AElementwiseOperation a_element_op,
                             BElementwiseOperation b_element_op,
                             CDEElementwiseOperation cde_element_op)
    {
        return Argument{
            p_As, p_Bs, p_Ds, p_Es, gemm_descs, a_element_op, b_element_op, cde_element_op};
    }

    static auto MakeInvoker() { return Invoker{}; }

    // polymorphic
    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(std::vector<const void*>& p_As,
                        std::vector<const void*>& p_Bs,
                        std::vector<std::array<const void*, NumDTensor>>& p_Ds,
                        std::vector<void*>& p_Es,
                        std::vector<GemmDesc>& gemm_descs,
                        AElementwiseOperation a_element_op,
                        BElementwiseOperation b_element_op,
                        CDEElementwiseOperation cde_element_op) override
    {
        return std::make_unique<Argument>(
            p_As, p_Bs, p_Ds, p_Es, gemm_descs, a_element_op, b_element_op, cde_element_op);
    }

    // polymorphic
    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    // polymorphic
    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "DeviceGroupedGemmMultipleD_Dl"
            << "<"
            << BlockSize << ", "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << K0PerBlock << ", "
            << K1 << ", "
            << M1PerThread << ", "
            << N1PerThread << ", "
            << KPerThread
            << getGemmSpecializationString(GemmSpec)
            << ">";
        // clang-format on

        return str.str();
    }

    size_t GetWorkSpaceSize(const BaseArgument* p_arg) const override
    {
        return dynamic_cast<const Argument*>(p_arg)->group_count_ * sizeof(GemmKernelArg);
    }

    size_t GetDeviceKernelArgSize(const BaseArgument* p_arg) const override
    {
        return GetWorkSpaceSize(p_arg);
    }

    size_t GetHostKernelArgSize(const BaseArgument* p_arg) const { return GetWorkSpaceSize(p_arg); }

    void SetDeviceKernelArgs(BaseArgument* p_arg, void* p_dev_kernel_args) const override
    {
        return this->SetWorkSpacePointer(p_arg, p_dev_kernel_args);
    }

    //----------------------------------------------------------------------------------------------
    /// @brief      Sets the host kernel arguments pointer and copies that data on the host side.
    ///             This function can be utilised to use pinned memory for the host args and
    ///             achieve fully async data copy.
    ///
    /// @param      p_arg              The pointer to the Argument we're going to update.
    /// @param[in]  p_host_kernel_args The pointer to the host memory where the kernel
    ///                                arguments will be copied
    void SetHostKernelArgsPointer(BaseArgument* p_arg, void* p_host_kernel_args) const
    {
        Argument* pArg_ = dynamic_cast<Argument*>(p_arg);
        if(!pArg_)
        {
            throw std::runtime_error("Failed to cast argument pointer!");
        }

        pArg_->gemm_kernel_host_args_ = p_host_kernel_args;
        std::copy(pArg_->gemm_desc_kernel_arg_.begin(),
                  pArg_->gemm_desc_kernel_arg_.end(),
                  static_cast<GemmKernelArg*>(pArg_->gemm_kernel_host_args_));
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
