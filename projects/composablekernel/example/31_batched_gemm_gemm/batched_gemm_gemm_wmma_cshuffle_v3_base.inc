// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/*
Gemm + Gemm fused operation. Computes C_g_m_n = (A_g_m_k * B0_g_k_l) * B1_g_l_n
                                                |------------------|
                                                        Gemm0
                                                |-----------------------------|
                                                             Gemm1
*/

static constexpr auto PipeSched   = ck::BlockGemmPipelineScheduler::Interwave;
static constexpr auto PipelineVer = ck::BlockGemmPipelineVersion::v1;
static constexpr auto GemmSpec    = ck::tensor_operation::device::GemmSpecialization::MNKOPadding;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

// clang-format off
// #define CK_MHA_USE_RCCR_LAYOUT
#define CK_MHA_USE_WAVE_1
// #define CK_MHA_USE_WAVE_2
// #define CK_MHA_USE_WAVE_4
// #define CK_MHA_USE_WAVE_8

#ifdef CK_MHA_USE_RCCR_LAYOUT
using DeviceMHAFactory = 
    std::tuple<
        ck::tensor_operation::device::DeviceBatchedGemmGemm_Wmma_CShuffleV3<
            Row, Col, Col, Row, 
            ADataType, B0DataType, B1DataType, CDataType, AccDataType, CShuffleDataType,
            AElementOp, B0ElementOp, Acc0ElementOp, B1ElementOp, CElementOp,
            GemmSpec,
            32,
            //      Gemm 0
            16, 64, 64, 64, 64, 8,  8,
            //      Gemm 1
                8,  
            16, 16,
            // Per repeat = wave_m = wave_num, wave_n = 1
            1, 4, 4,
            // ABlockTransfer MK -> K0 M K1
            S<2, 16, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, false,
            // B0BlockTransfer LK -> K0 L K1
            S<2, 16, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, false,
            // B1BlockTransfer NL -> L0 N L1
            S<2, 16, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, true,
            // CShuffleBlockTransfer MN
            1, 1, S<1, 16, 1, 2>, 8,
            PipeSched, PipelineVer>
    >;
#else
using DeviceMHAFactory = 
    std::tuple<
#ifdef CK_MHA_USE_WAVE_1
        // 1 wave, mrepeat = 1, nrepeat = 2, k/o repeat = 1~5
        ck::tensor_operation::device::DeviceBatchedGemmGemm_Wmma_CShuffleV3<
            Row, Col, Row, Row, 
            ADataType, B0DataType, B1DataType, CDataType, AccDataType, CShuffleDataType,
            AElementOp, B0ElementOp, Acc0ElementOp, B1ElementOp, CElementOp,
            GemmSpec,
            32,
            //      Gemm 0
            16, 128, 64, 64, 64, 8,  8,
            //      Gemm 1
                8,  
            16, 16,
            // Per repeat = wave_m = wave_num, wave_n = 1
            1, 8, 4,
            // ABlockTransfer MK -> K0 M K1
            S<2, 16, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, true,
            // B0BlockTransfer LK -> K0 L K1
            S<2, 16, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, true,
            // B1BlockTransfer NL -> L0 N L1
            S<2, 2, 8>, S<0, 2, 1>, S<0, 2, 1>, 1, 8, 1, false,
            // CShuffleBlockTransfer MN
            1, 1, S<1, 16, 1, 2>, 8, 
            PipeSched, PipelineVer>,
        ck::tensor_operation::device::DeviceBatchedGemmGemm_Wmma_CShuffleV3<
            Row, Col, Row, Row, 
            ADataType, B0DataType, B1DataType, CDataType, AccDataType, CShuffleDataType,
            AElementOp, B0ElementOp, Acc0ElementOp, B1ElementOp, CElementOp,
            GemmSpec,
            32,
            //      Gemm 0
            16, 64, 64, 64, 64, 8,  8,
            //      Gemm 1
                8,  
            16, 16,
            // Per repeat = wave_m = wave_num, wave_n = 1
            1, 4, 4,
            // ABlockTransfer MK -> K0 M K1
            S<2, 16, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, true,
            // B0BlockTransfer LK -> K0 L K1
            S<2, 16, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, true,
            // B1BlockTransfer NL -> L0 N L1
            S<2, 2, 8>, S<0, 2, 1>, S<0, 2, 1>, 1, 8, 1, false,
            // CShuffleBlockTransfer MN
            1, 1, S<1, 16, 1, 2>, 8,
            PipeSched, PipelineVer>
#endif
#ifdef CK_MHA_USE_WAVE_2
        ck::tensor_operation::device::DeviceBatchedGemmGemm_Wmma_CShuffleV3<
            Row, Col, Row, Row, 
            ADataType, B0DataType, B1DataType, CDataType, AccDataType, CShuffleDataType,
            AElementOp, B0ElementOp, Acc0ElementOp, B1ElementOp, CElementOp,
            GemmSpec,
            64,
            //      Gemm 0
            32, 128, 64, 64, 64, 8, 8,
            //      Gemm 1
                 8,  
            16, 16,
            // Per repeat = wave_m = wave_num, wave_n = 1
            1, 8, 4,
            // ABlockTransfer MK -> K0 M K1
            S<2, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, true,
            // B0BlockTransfer LK -> K0 L K1
            S<4, 16, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, true,
            // B1BlockTransfer NL -> L0 N L1
            S<2, 4, 8>, S<0, 2, 1>, S<0, 2, 1>, 1, 4, 1, false,
            // CShuffleBlockTransfer MN
            1, 1, S<1, 32, 1, 2>, 8,
            PipeSched, PipelineVer>,
        ck::tensor_operation::device::DeviceBatchedGemmGemm_Wmma_CShuffleV3<
            Row, Col, Row, Row, 
            ADataType, B0DataType, B1DataType, CDataType, AccDataType, CShuffleDataType,
            AElementOp, B0ElementOp, Acc0ElementOp, B1ElementOp, CElementOp,
            GemmSpec,
            64,
            //      Gemm 0
            32, 64, 64, 64, 64, 8, 8,
            //      Gemm 1
                8,  
            16, 16,
            // Per repeat = wave_m = wave_num, wave_n = 1
            1, 4, 4,
            // ABlockTransfer MK -> K0 M K1
            S<2, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, true,
            // B0BlockTransfer LK -> K0 L K1
            S<4, 16, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, true,
            // B1BlockTransfer NL -> L0 N L1
            S<2, 4, 8>, S<0, 2, 1>, S<0, 2, 1>, 1, 4, 1, false,
            // CShuffleBlockTransfer MN
            1, 1, S<1, 32, 1, 2>, 8,
            PipeSched, PipelineVer>
#endif
#ifdef CK_MHA_USE_WAVE_4
        ck::tensor_operation::device::DeviceBatchedGemmGemm_Wmma_CShuffleV3<
            Row, Col, Row, Row, 
            ADataType, B0DataType, B1DataType, CDataType, AccDataType, CShuffleDataType,
            AElementOp, B0ElementOp, Acc0ElementOp, B1ElementOp, CElementOp,
            GemmSpec,
            128,
            //      Gemm 0
            64, 128, 64, 64, 64, 8, 8,
            //      Gemm 1
                8,  
            16, 16,
            // Per repeat = wave_m = wave_num, wave_n = 1
            1, 8, 4,
            // ABlockTransfer MK -> K0 M K1
            S<2, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, true,
            // B0BlockTransfer LK -> K0 L K1
            S<8, 16, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, true,
            // B1BlockTransfer NL -> L0 N L1
            S<2, 8, 8>, S<0, 2, 1>, S<0, 2, 1>, 1, 2, 1, false,
            // CShuffleBlockTransfer MN
            1, 1, S<1, 64, 1, 2>, 8,
            PipeSched, PipelineVer>,
        ck::tensor_operation::device::DeviceBatchedGemmGemm_Wmma_CShuffleV3<
            Row, Col, Row, Row, 
            ADataType, B0DataType, B1DataType, CDataType, AccDataType, CShuffleDataType,
            AElementOp, B0ElementOp, Acc0ElementOp, B1ElementOp, CElementOp,
            GemmSpec,
            128,
            //      Gemm 0
            64, 64, 64, 64, 64, 8, 8,
            //      Gemm 1
                8,  
            16, 16,
            // Per repeat = wave_m = wave_num, wave_n = 1
            1, 4, 4,
            // ABlockTransfer MK -> K0 M K1
            S<2, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, true,
            // B0BlockTransfer LK -> K0 L K1
            S<8, 16, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, true,
            // B1BlockTransfer NL -> L0 N L1
            S<2, 8, 8>, S<0, 2, 1>, S<0, 2, 1>, 1, 2, 1, false,
            // CShuffleBlockTransfer MN
            1, 1, S<1, 64, 1, 2>, 8,
            PipeSched, PipelineVer>
#endif
#ifdef CK_MHA_USE_WAVE_8
        ck::tensor_operation::device::DeviceBatchedGemmGemm_Wmma_CShuffleV3<
            Row, Col, Row, Row, 
            ADataType, B0DataType, B1DataType, CDataType, AccDataType, CShuffleDataType,
            AElementOp, B0ElementOp, Acc0ElementOp, B1ElementOp, CElementOp,
            GemmSpec,
            256,
            //      Gemm 0
            128, 128, 64, 64, 64, 8, 8,   
            //      Gemm 1
                  8,  
            16, 16,
            // Per repeat = wave_m = wave_num, wave_n = 1
            1, 8, 4,
            // ABlockTransfer MK -> K0 M K1
            S<2, 128, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, true,
            // B0BlockTransfer LK -> K0 L K1
            S<8,  32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, true,
            // B1BlockTransfer NL -> L0 N L1
            S<2,  16, 8>, S<0, 2, 1>, S<0, 2, 1>, 1, 1, 1, false,
            // CShuffleBlockTransfer MN
            1, 1, S<1, 128, 1, 2>, 8,
            PipeSched, PipelineVer>,
        ck::tensor_operation::device::DeviceBatchedGemmGemm_Wmma_CShuffleV3<
            Row, Col, Row, Row, 
            ADataType, B0DataType, B1DataType, CDataType, AccDataType, CShuffleDataType,
            AElementOp, B0ElementOp, Acc0ElementOp, B1ElementOp, CElementOp,
            GemmSpec,
            256,
            //      Gemm 0
            128, 128, 64, 64, 64, 8, 8,   
            //      Gemm 1
                  8,  
            16, 16,
            // Per repeat = wave_m = wave_num, wave_n = 1
            1, 8, 4,
            // ABlockTransfer MK -> K0 M K1
            S<2, 128, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, true,
            // B0BlockTransfer LK -> K0 L K1
            S<8,  32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, true,
            // B1BlockTransfer NL -> L0 N L1
            S<2,  16, 8>, S<0, 2, 1>, S<0, 2, 1>, 1, 1, 1, false,
            // CShuffleBlockTransfer MN
            1, 1, S<1, 128, 1, 2>, 8,
            PipeSched, PipelineVer>
#endif
    >;
#endif

// clang-format on
// Ref Gemm0
using ReferenceGemm0Instance = ck::tensor_operation::host::ReferenceBatchedGemm<ADataType,
                                                                                B0DataType,
                                                                                AccDataType,
                                                                                AccDataType,
                                                                                AElementOp,
                                                                                B0ElementOp,
                                                                                Acc0ElementOp>;

// Ref Gemm1
using ReferenceGemm1Instance = ck::tensor_operation::host::ReferenceBatchedGemm<ADataType,
                                                                                B1DataType,
                                                                                CDataType,
                                                                                AccDataType,
                                                                                AElementOp,
                                                                                B1ElementOp,
                                                                                CElementOp>;

using Row    = ck::tensor_layout::gemm::RowMajor;
using Col    = ck::tensor_layout::gemm::ColumnMajor;
using Bypass = ck::tensor_layout::BypassLayoutVerification;

#include "run_batched_gemm_gemm_wmma_cshuffle_v3.inc"

int main(int argc, char* argv[])
{
    bool is_supported = ck::is_gfx11_supported() || ck::is_gfx12_supported();
    if(!is_supported)
    {
        std::cout << "WARNING: wmma example not supported on the platform " << ck::get_device_name()
                  << std::endl;
        return 0;
    }
    return run(argc, argv);
}
