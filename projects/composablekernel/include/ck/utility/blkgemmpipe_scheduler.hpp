// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/tensor_description/tensor_adaptor.hpp"
#include "ck/utility/common_header.hpp"
#include "ck/utility/scheduler_enum.hpp"

namespace ck {

enum SchedulerGroup : uint32_t
{
    SCHED_GROUP_MFMA      = 0x008, // Matrix FMA instructions
    SCHED_GROUP_VMEM      = 0x020, // Global memory operations
    SCHED_GROUP_LDS_READ  = 0x100, // LDS read operations
    SCHED_GROUP_LDS_WRITE = 0x200  // LDS write operations
};

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
          index_t MPerXDL,
          index_t NPerXDL,
          index_t KPerXDL,
          bool IsF4F6 = false>
struct BlockwiseGemmXdlops_pipeline_hotloop_inst
{
    static constexpr index_t WaveNumM = MPerBlock / (MRepeat * MPerXDL);
    static constexpr index_t WaveNumN = NPerBlock / (NRepeat * NPerXDL);
    static constexpr index_t WaveSize = BlockSize / WaveNumM / WaveNumN;

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

    static constexpr index_t C_MFMA_Inst_Num =
        MPerBlock * NPerBlock * KPerBlock / (BlockSize / WaveSize) / (MPerXDL * NPerXDL * KPerXDL);

    static constexpr index_t C_MFMA_SpeedUp = IsF4F6 ? 2 : 1;

    static constexpr index_t C_MFMA_Inst_Cycle = []() {
        if constexpr(NPerXDL == 16)
        {
            return KPerXDL == 128 ? 32 / C_MFMA_SpeedUp : 16 / C_MFMA_SpeedUp;
        }
        else if constexpr(NPerXDL == 32)
        {
            return KPerXDL == 64 ? 64 / C_MFMA_SpeedUp : 32 / C_MFMA_SpeedUp;
        }
    }();

    static constexpr auto Print()
    {
        printf(" Blk/Wave Size: %d, %d, M/N/K PerBlk: %d, %d, %d, M/N/K PerXdl: %d, %d, %d\n",
               BlockSize,
               WaveSize,
               MPerBlock,
               NPerBlock,
               KPerBlock,
               MPerXDL,
               NPerXDL,
               KPerXDL);

        printf(" A/B buffer load inst: %d, %d\n A/B LDS write inst: %d, %d\n A/B LDS read inst: "
               "%d, %d\n C MFMA inst: %d C MFMA cycle: %d\n"
               "A/B LDS read width: %d, %d, A/B LDS write width: %d, %d, A/B buffer load width: "
               "%d/ %d\n",
               A_Buffer_Load_Inst_Num,
               B_Buffer_Load_Inst_Num,
               A_LDS_Write_Inst_Num,
               B_LDS_Write_Inst_Num,
               A_LDS_Read_Inst_Num,
               B_LDS_Read_Inst_Num,
               C_MFMA_Inst_Num,
               C_MFMA_Inst_Cycle,
               A_LDS_Read_Width,
               B_LDS_Read_Width,
               ALDSWriteWidth,
               BLDSWriteWidth,
               ABufferLoadWidth,
               BBufferLoadWidth);
    }
};

} // namespace ck
