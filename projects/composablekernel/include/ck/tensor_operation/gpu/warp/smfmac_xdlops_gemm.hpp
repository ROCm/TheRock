// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/utility/math.hpp"
#include "ck/utility/amd_smfmac.hpp"

namespace ck {

enum struct SmfmacInstr
{
    smfmac_f32_16x16x32f16 = 0,
    smfmac_f32_32x32x16f16,
    smfmac_f32_16x16x32bf16,
    smfmac_f32_32x32x16bf16,
};

template <SmfmacInstr instr>
struct smfmac_type;

template <>
struct smfmac<SmfmacInstr::smfmac_f32_16x16x32f16>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 1;
    static constexpr index_t num_regs_per_blk    = 4;
    static constexpr index_t num_threads_per_blk = 16;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = 4;
    static constexpr index_t num_output_blks     = 1;
    static constexpr index_t m_per_blk           = 16;
    static constexpr index_t n_per_blk           = 16;
    static constexpr index_t k_per_blk           = 8;
    static constexpr bool is_k_reduction         = true;

    template <index_t MPerXdlops,
              index_t NPerXdlops,
              index_t idx_part,
              class FloatA,
              class FloatB,
              class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, const index_t& idx, FloatC& reg_c) const
    {
        intrin_smfmac_f32_16x16x32f16<MPerXdlops, NPerXdlops>::Run<FloatC, idx_part>(
            a, b, idx, reg_c);
    }
};

template <>
struct smfmac<SmfmacInstr::smfmac_f32_32x32x16f16>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 4;
    static constexpr index_t num_regs_per_blk    = 16;
    static constexpr index_t num_threads_per_blk = 32;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = 2;
    static constexpr index_t num_output_blks     = 1;
    static constexpr index_t m_per_blk           = 32;
    static constexpr index_t n_per_blk           = 32;
    static constexpr index_t k_per_blk           = 16;
    static constexpr bool is_k_reduction         = true;

    template <index_t MPerXdlops,
              index_t NPerXdlops,
              index_t idx_part,
              class FloatA,
              class FloatB,
              class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, const index_t& idx, FloatC& reg_c) const
    {
        intrin_smfmac_f32_32x32x16f16<MPerXdlops, NPerXdlops>::Run<FloatC, idx_part>(
            a, b, idx, reg_c);
    }
};

template <>
struct smfmac<SmfmacInstr::smfmac_f32_16x16x32bf16>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 1;
    static constexpr index_t num_regs_per_blk    = 4;
    static constexpr index_t num_threads_per_blk = 16;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = 4;
    static constexpr index_t num_output_blks     = 1;
    static constexpr index_t m_per_blk           = 16;
    static constexpr index_t n_per_blk           = 16;
    static constexpr index_t k_per_blk           = 8;
    static constexpr bool is_k_reduction         = true;

    template <index_t MPerXdlops,
              index_t NPerXdlops,
              index_t idx_part,
              class FloatA,
              class FloatB,
              class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, const index_t& idx, FloatC& reg_c) const
    {
        intrin_smfmac_f32_16x16x32bf16<MPerXdlops, NPerXdlops>::Run<FloatC, idx_part>(
            a, b, idx, reg_c);
    }
};

template <>
struct smfmac<SmfmacInstr::smfmac_f32_32x32x16bf16>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 4;
    static constexpr index_t num_regs_per_blk    = 16;
    static constexpr index_t num_threads_per_blk = 32;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = 2;
    static constexpr index_t num_output_blks     = 1;
    static constexpr index_t m_per_blk           = 32;
    static constexpr index_t n_per_blk           = 32;
    static constexpr index_t k_per_blk           = 16;
    static constexpr bool is_k_reduction         = true;

    template <index_t MPerXdlops,
              index_t NPerXdlops,
              index_t idx_part,
              class FloatA,
              class FloatB,
              class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, const index_t& idx, FloatC& reg_c) const
    {
        intrin_smfmac_f32_32x32x16bf16<MPerXdlops, NPerXdlops>::Run<FloatC, idx_part>(
            a, b, idx, reg_c);
    }
};

template <typename base_type,
          index_t MPerXdlops,
          index_t NPerXdlops,
          typename additional_type = base_type>
struct SmfmacSelector
{
    template <typename base_type_,
              index_t MPerXdlops_,
              index_t NPerXdlops_,
              typename additional_type_ = base_type_>
    static constexpr auto GetSmfmac();

    template <>
    static constexpr auto GetSmfmac<half_t, 16, 16>()
    {
        return SmfmacInstr::smfmac_f32_16x16x32f16;
    }

    template <>
    static constexpr auto GetSmfmac<half_t, 32, 32>()
    {
        return SmfmacInstr::smfmac_f32_32x32x16f16;
    }

    template <>
    static constexpr auto GetSmfmac<bhalf_t, 16, 16>()
    {
        return SmfmacInstr::smfmac_f32_16x16x32bf16;
    }

    template <>
    static constexpr auto GetSmfmac<bhalf_t, 32, 32>()
    {
        return SmfmacInstr::smfmac_f32_32x32x16bf16;
    }

    static constexpr auto selected_smfmac =
        smfmac_type<GetSmfmac<base_type, MPerXdlops, NPerXdlops, additional_type>()>{};

    __host__ __device__ constexpr SmfmacSelector()
    {
        static_assert(selected_smfmac.group_size * selected_smfmac.num_groups_per_blk ==
                          selected_smfmac.num_regs_per_blk,
                      "wrong! num_regs_per_blk");

        static_assert(selected_smfmac.num_threads_per_blk == selected_smfmac.n_per_blk,
                      "n_per_blk != num_threads_per_blk");

        static_assert(selected_smfmac.num_regs_per_blk * selected_smfmac.num_input_blks ==
                          selected_smfmac.m_per_blk,
                      "m_per_blk != num_input_blks * num_regs_per_blk");

        static_assert(selected_smfmac.num_output_blks == selected_smfmac.num_input_blks ||
                          selected_smfmac.num_output_blks == 1,
                      "incorrect num_output_blks");

        static_assert(selected_smfmac.num_regs_per_blk * selected_smfmac.wave_size ==
                          selected_smfmac.m_per_blk * selected_smfmac.n_per_blk,
                      "num_regs_per_blk incorrect");

        static_assert(selected_smfmac.is_k_reduction ||
                          (selected_smfmac.num_input_blks == selected_smfmac.num_output_blks),
                      "is_k_reduction wrong!");
    }

    static constexpr index_t GetKPerXdlops()
    {
        return (selected_smfmac.is_k_reduction ? selected_smfmac.num_input_blks : 1) *
               selected_smfmac.k_per_blk;
    }

    static constexpr index_t GetK1PerXdlops() { return selected_smfmac.k_per_blk; }
};

template <typename base_type,
          index_t MPerXdlops,
          index_t NPerXdlops,
          index_t KPack,
          typename additional_type = base_type>
struct SparseXdlopsGemm
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};

    using CIndex   = MultiIndex<2>;
    using CIndex4D = MultiIndex<4>;

    __device__ static constexpr index_t GetNumBlks() { return smfmac_instr.num_output_blks; }

    __device__ static constexpr index_t GetNumXdlops()
    {
        return MPerXdlops * NPerXdlops /
               (smfmac_instr.m_per_blk * smfmac_instr.n_per_blk * smfmac_instr.num_output_blks);
    }

    __host__ __device__ constexpr SparseXdlopsGemm()
    {
        static_assert(NPerXdlops == 16 || NPerXdlops == 32,
                      "Only support GemmNPerXdlops == 16 or 32 for smfmac xdlops");

        static_assert(MPerXdlops == 16 || MPerXdlops == 32,
                      "Only support GemmMPerXdlops == 16 or 32 for smfmac xdlops");

        static_assert(KPack % smfmac_instr.k_per_blk == 0, "KPack cannot be divided by k_per_blk");
    }

    // XDL output supporting C = A * B
    // M2_N2 -> M2_M3_M4_N2
    template <typename CDesc_M0_N0_M1_N1_M2_N2>
    __host__ __device__ static constexpr auto
    MakeCDescriptor_M0_N0_M1_N1_M2_M3_M4_N2(const CDesc_M0_N0_M1_N1_M2_N2& c_desc_m0_n0_m1_n1_m2_n2)
    {
        const auto M0 = c_desc_m0_n0_m1_n1_m2_n2.GetLength(I0);
        const auto N0 = c_desc_m0_n0_m1_n1_m2_n2.GetLength(I1);
        const auto M1 = c_desc_m0_n0_m1_n1_m2_n2.GetLength(I2);
        const auto N1 = c_desc_m0_n0_m1_n1_m2_n2.GetLength(I3);

        return transform_tensor_descriptor(
            c_desc_m0_n0_m1_n1_m2_n2,
            make_tuple(make_pass_through_transform(M0),
                       make_pass_through_transform(N0),
                       make_pass_through_transform(M1),
                       make_pass_through_transform(N1),
                       make_unmerge_transform(make_tuple(Number<smfmac_instr.num_groups_per_blk>{},
                                                         Number<smfmac_instr.num_input_blks>{},
                                                         Number<smfmac_instr.group_size>{})),
                       make_pass_through_transform(Number<smfmac_instr.num_threads_per_blk>{})),
            make_tuple(Sequence<0>{},
                       Sequence<1>{},
                       Sequence<2>{},
                       Sequence<3>{},
                       Sequence<4>{},
                       Sequence<5>{}),
            make_tuple(Sequence<0>{},
                       Sequence<1>{},
                       Sequence<2>{},
                       Sequence<3>{},
                       Sequence<4, 5, 6>{},
                       Sequence<7>{}));
    }

    template <typename CDesc_G_M0_N0_M1_N1_M2_N2>
    __host__ __device__ static constexpr auto MakeCDescriptor_G_M0_N0_M1_N1_M2_M3_M4_N2(
        const CDesc_G_M0_N0_M1_N1_M2_N2& c_desc_g_m0_n0_m1_n1_m2_n2)
    {
        const auto G  = c_desc_g_m0_n0_m1_n1_m2_n2.GetLength(I0);
        const auto M0 = c_desc_g_m0_n0_m1_n1_m2_n2.GetLength(I1);
        const auto N0 = c_desc_g_m0_n0_m1_n1_m2_n2.GetLength(I2);
        const auto M1 = c_desc_g_m0_n0_m1_n1_m2_n2.GetLength(I3);
        const auto N1 = c_desc_g_m0_n0_m1_n1_m2_n2.GetLength(I4);

        return transform_tensor_descriptor(
            c_desc_g_m0_n0_m1_n1_m2_n2,
            make_tuple(make_pass_through_transform(G),
                       make_pass_through_transform(M0),
                       make_pass_through_transform(N0),
                       make_pass_through_transform(M1),
                       make_pass_through_transform(N1),
                       make_unmerge_transform(make_tuple(smfmac_instr.num_groups_per_blk,
                                                         smfmac_instr.num_input_blks,
                                                         smfmac_instr.group_size)),
                       make_pass_through_transform(smfmac_instr.num_threads_per_blk)),
            make_tuple(Sequence<0>{},
                       Sequence<1>{},
                       Sequence<2>{},
                       Sequence<3>{},
                       Sequence<4>{},
                       Sequence<5>{},
                       Sequence<6>{}),
            make_tuple(Sequence<0>{},
                       Sequence<1>{},
                       Sequence<2>{},
                       Sequence<3>{},
                       Sequence<4>{},
                       Sequence<5, 6, 7>{},
                       Sequence<8>{}));
    }

    __device__ static constexpr index_t GetRegSizePerXdlops()
    {
        return MPerXdlops * NPerXdlops / smfmac_instr.wave_size;
    }

    __device__ static constexpr index_t GetWaveSize() { return smfmac_instr.wave_size; }

    template <class FloatA, class FloatB, class Idx, class FloatC>
    __device__ void
    Run(const FloatA& p_a_wave, const FloatB& p_b_wave, const Idx& idx, FloatC& p_c_thread) const
    {
        static_assert(is_same<base_type, half_t>::value || is_same<base_type, bhalf_t>::value,
                      "base base_type must be half or bfloat16!");

        static_for<0, KPack / smfmac_instr.k_per_blk, 1>{}([&](auto k) {
            smfmac_instr.template run<MPerXdlops, NPerXdlops, k % 4>(
                p_a_wave[k], p_b_wave[k], idx[k / 4], p_c_thread);
        });
    }

    __device__ static auto GetLaneId() { return get_thread_local_1d_id() % smfmac_instr.wave_size; }

    __device__ static auto GetBlkIdx()
    {
        const auto laneId = GetLaneId();

        constexpr auto threadidx_to_blk_idx_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_merge_transform(
                make_tuple(1, smfmac_instr.num_input_blks, smfmac_instr.num_threads_per_blk))),
            make_tuple(Sequence<0, 1, 2>{}),
            make_tuple(Sequence<0>{}));

        const auto blk_idx =
            threadidx_to_blk_idx_adaptor.CalculateBottomIndex(make_multi_index(laneId));

        const auto blk_id = blk_idx[I1];
        const auto blk_td = blk_idx[I2];

        return make_tuple(blk_id, blk_td);
    }

    __host__ __device__ static auto CalculateAThreadOriginDataIndex()
    {
        const auto laneId  = GetLaneId();
        const auto blk_idx = GetBlkIdx();

        const auto blk_id = blk_idx[I0];
        const auto blk_td = blk_idx[I1];

        if constexpr(smfmac_instr.is_k_reduction)
        {
            return make_tuple(blk_id, blk_td);
        }
        else
        {
            return make_tuple(0, laneId);
        }
    }

    __host__ __device__ static auto CalculateBThreadOriginDataIndex()
    {
        const auto laneId  = GetLaneId();
        const auto blk_idx = GetBlkIdx();

        const auto blk_id = blk_idx[I0];
        const auto blk_td = blk_idx[I1];

        if constexpr(smfmac_instr.is_k_reduction)
        {
            return make_tuple(blk_id, blk_td);
        }
        else
        {
            return make_tuple(0, laneId);
        }
    }

    __device__ static CIndex GetBeginOfThreadBlk(index_t xdlops_i, index_t blk_i)
    {
        const auto blk_idx = GetBlkIdx();

        const auto blk_id = blk_idx[I0];
        const auto blk_td = blk_idx[I1];

        index_t n_offset = blk_i * smfmac_instr.n_per_blk + blk_td;
        index_t m_offset = xdlops_i * smfmac_instr.m_per_blk + blk_id * smfmac_instr.group_size;

        return CIndex{m_offset, n_offset};
    }

    __device__ static CIndex4D GetBeginOfThreadBlk4D(index_t /* xdlops_i */, index_t /* blk_i */)
    {
        const auto blk_idx = GetBlkIdx();

        const auto blk_id = blk_idx[I0];
        const auto blk_td = blk_idx[I1];

        return CIndex4D{I0, blk_id, I0, blk_td};
    }

    static constexpr auto smfmac =
        SmfmacSelector<base_type, MPerXdlops, NPerXdlops, additional_type>{};

    static constexpr auto smfmac_instr = smfmac.selected_smfmac;

    static constexpr auto KPerXdlops  = smfmac.GetKPerXdlops();
    static constexpr auto K1PerXdlops = smfmac.GetK1PerXdlops();
    static constexpr auto K0PerXdlops = KPerXdlops / K1PerXdlops;

    __host__ __device__ static constexpr auto GetCM0M1M2NThreadBlkLengths()
    {
        return make_tuple(
            Number<smfmac_instr.num_groups_per_blk>{}, I1, Number<smfmac_instr.group_size>{}, I1);
    }
};

} // namespace ck
