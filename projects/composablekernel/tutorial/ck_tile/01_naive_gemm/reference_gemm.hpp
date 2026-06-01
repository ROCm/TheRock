// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"

template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType>
void reference_basic_gemm(const ck_tile::HostTensor<ADataType>& a_m_k,
                          const ck_tile::HostTensor<BDataType>& b_n_k,
                          ck_tile::HostTensor<CDataType>& c_m_n)
{
    const int N = b_n_k.mDesc.get_lengths()[0];
    const int K = b_n_k.mDesc.get_lengths()[1];

    auto f = [&](auto m) {
        for(int n = 0; n < N; ++n)
        {
            AccDataType v_acc = 0;

            for(int k = 0; k < K; ++k)
            {
                ADataType v_a = a_m_k(m, k);
                BDataType v_b = b_n_k(n, k);

                v_acc += ck_tile::type_convert<AccDataType>(v_a) *
                         ck_tile::type_convert<AccDataType>(v_b);
            }

            c_m_n(m, n) = ck_tile::type_convert<CDataType>(v_acc);
        }
    };

    ck_tile::make_ParallelTensorFunctor(f, c_m_n.mDesc.get_lengths()[0])(1);
}
