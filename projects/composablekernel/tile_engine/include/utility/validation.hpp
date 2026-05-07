// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType>
auto calculate_rtol_atol(const ck_tile::index_t K,
                         const ck_tile::index_t kbatch,
                         const float max_accumulated_value)
{
    using ComputeType =
        std::conditional_t<sizeof(ADataType) < sizeof(BDataType), ADataType, BDataType>;
    // Calculate thresholds
    const auto rtol = ck_tile::get_relative_threshold<ComputeType, CDataType, AccDataType>(
        ck_tile::integer_divide_ceil(K, kbatch));
    const auto atol = ck_tile::get_absolute_threshold<ComputeType, CDataType, AccDataType>(
        max_accumulated_value / kbatch, ck_tile::integer_divide_ceil(K, kbatch));
    // Calculate error due to split_k accumulation
    const auto rtol_split_k =
        ck_tile::get_relative_threshold<CDataType, CDataType, CDataType>(kbatch);
    const auto atol_split_k = ck_tile::get_absolute_threshold<CDataType, CDataType, CDataType>(
        max_accumulated_value, kbatch);
    // Use higher threshold
    return ck_tile::make_tuple(std::max(rtol, rtol_split_k), std::max(atol, atol_split_k));
}

/// @brief Function to compare the results of the device and host computations
bool compare(std::string instanceName,
             ck_tile::index_t K,
             ck_tile::index_t kbatch,
             ck_tile::HostTensor<CDataType>& c_m_n_dev_result,
             ck_tile::HostTensor<CDataType>& c_m_n_host_result)
{
    const float max_accumulated_value =
        *std::max_element(c_m_n_host_result.mData.begin(), c_m_n_host_result.mData.end());
    const auto rtol_atol = calculate_rtol_atol<ADataType, BDataType, AccDataType, CDataType>(
        K, kbatch, max_accumulated_value);
    bool pass = ck_tile::check_err(c_m_n_dev_result,
                                   c_m_n_host_result,
                                   "Error: Incorrect results!",
                                   rtol_atol.at(ck_tile::number<0>{}),
                                   rtol_atol.at(ck_tile::number<1>{}));

    std::cout << "For " << instanceName << " Relative error threshold is "
              << rtol_atol.at(ck_tile::number<0>{}) << " Absolute error threshold is "
              << rtol_atol.at(ck_tile::number<1>{}) << std::endl;
    std::cout << "The verification result is:" << (pass ? "correct" : "fail") << std::endl;

    return pass;
}
