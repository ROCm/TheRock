// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/common_header.hpp"

namespace ck {

template <index_t BlockSize,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t ABufferLoadWidth,
          index_t BBufferLoadWidth,
          index_t ALDSWriteWidth,
          index_t BLDSWriteWidth,
          index_t ALDSReadWidth,
          index_t BLDSReadWidth,
          index_t MRepeat,
          index_t NRepeat,
          index_t MPerWmma,
          index_t NPerWmma,
          index_t KPerWmma>
struct BlockwiseGemmWmmaops_pipeline_hotloop_inst
{
    static constexpr index_t WaveSize = 32;
    static constexpr index_t WaveNumM = MPerBlock / (MRepeat * MPerWmma);
    static constexpr index_t WaveNumN = NPerBlock / (NRepeat * NPerWmma);

    static constexpr index_t A_LDS_Read_Width = ALDSReadWidth;
    static constexpr index_t B_LDS_Read_Width = BLDSReadWidth;

    static constexpr index_t A_Buffer_Load_Inst_Num =
        MPerBlock * KPerBlock / (BlockSize * ABufferLoadWidth);
    static constexpr index_t B_Buffer_Load_Inst_Num =
        NPerBlock * KPerBlock / (BlockSize * BBufferLoadWidth);

    static constexpr index_t A_LDS_Write_Inst_Num =
        MPerBlock * KPerBlock / (BlockSize * ALDSWriteWidth);
    static constexpr index_t B_LDS_Write_Inst_Num =
        NPerBlock * KPerBlock / (BlockSize * BLDSWriteWidth);

    static constexpr index_t A_LDS_Read_Inst_Num =
        WaveNumN * MPerBlock * KPerBlock / (BlockSize * ALDSReadWidth);
    static constexpr index_t B_LDS_Read_Inst_Num =
        WaveNumM * NPerBlock * KPerBlock / (BlockSize * BLDSReadWidth);

    static constexpr index_t C_WMMA_Inst_Num = MPerBlock * NPerBlock * KPerBlock /
                                               (BlockSize / WaveSize) /
                                               (MPerWmma * NPerWmma * KPerWmma);

    static constexpr auto Print()
    {
        printf(" Blk/Wave Size: %d, %d, M/N/K PerBlk: %d, %d, %d, M/N/K PerWmma: %d, %d, %d\n",
               BlockSize,
               WaveSize,
               MPerBlock,
               NPerBlock,
               KPerBlock,
               MPerWmma,
               NPerWmma,
               KPerWmma);

        printf(" A/B buffer load inst: %d, %d\n A/B LDS write inst: %d, %d\n A/B LDS read inst: "
               "%d, %d\n C WMMA inst: %d\n"
               "A/B LDS read width: %d, %d, A/B LDS write width: %d, %d, A/B buffer load width: "
               "%d, %d\n",
               A_Buffer_Load_Inst_Num,
               B_Buffer_Load_Inst_Num,
               A_LDS_Write_Inst_Num,
               B_LDS_Write_Inst_Num,
               A_LDS_Read_Inst_Num,
               B_LDS_Read_Inst_Num,
               C_WMMA_Inst_Num,
               A_LDS_Read_Width,
               B_LDS_Read_Width,
               ALDSWriteWidth,
               BLDSWriteWidth,
               ABufferLoadWidth,
               BBufferLoadWidth);
    }
};

} // namespace ck
