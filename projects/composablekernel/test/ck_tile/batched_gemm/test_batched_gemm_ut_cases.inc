// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

struct GemmParams
{
    int M;
    int N;
    int K;
    int batchCount;
};

struct StrideConfig
{
    int strideA;
    int strideB;
    int strideC;
    int batchStrideA;
    int batchStrideB;
    int batchStrideC;
};

TYPED_TEST(TestCkTileBatchedGemm, Basic)
{
    std::vector<GemmParams> gemmParams{{256, 256, 256, 1},
                                       {256, 256, 256, 2},
                                       {256, 256, 512, 2},
                                       {256, 256, 64, 2},
                                       {256, 256, 64, 3},
                                       {256, 256, 64, 4},
                                       {256, 256, 64, 8},
                                       {256, 256, 64, 16}};

    if(ck_tile::get_device_name() != "gfx950")
    {
        gemmParams.push_back({256, 256, 128, 2});
    }

    for(auto& params : gemmParams)
    {
        std::vector<StrideConfig> strideConfigs{{params.K,
                                                 params.N,
                                                 params.N,
                                                 params.M * params.K,
                                                 params.K * params.N,
                                                 params.M * params.N},
                                                {params.K,
                                                 params.K,
                                                 params.N,
                                                 params.M * params.K,
                                                 params.K * params.N,
                                                 params.M * params.N},
                                                {params.M,
                                                 params.N,
                                                 params.N,
                                                 params.M * params.K,
                                                 params.K * params.N,
                                                 params.M * params.N},
                                                {params.M,
                                                 params.K,
                                                 params.N,
                                                 params.M * params.K,
                                                 params.K * params.N,
                                                 params.M * params.N}};

        for(auto& conf : strideConfigs)
        {
            this->Run(params.M,
                      params.N,
                      params.K,
                      conf.strideA,
                      conf.strideB,
                      conf.strideC,
                      conf.batchStrideA,
                      conf.batchStrideB,
                      conf.batchStrideC,
                      params.batchCount);
        }
    }
}
