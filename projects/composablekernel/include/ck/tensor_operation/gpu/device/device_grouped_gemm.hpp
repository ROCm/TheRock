// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "device_base.hpp"
#include "ck/utility/ignore.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

///
/// @brief      Structure representing single GEMM problem arguments.
///
///             The pointer to the vector of those structures is passed to the GroupedGEMM entry
///             point kernel.
///
/// @tparam     NumDTensor  The number of D input tensors.
///
template <index_t NumDTensor = 0>
struct GroupedGemmKernelArgument
{
    __host__ __device__ GroupedGemmKernelArgument(const void* p_a_grid_,
                                                  const void* p_b_grid_,
                                                  std::array<const void*, NumDTensor> p_ds_grid_,
                                                  void* p_e_grid_,
                                                  index_t M_,
                                                  index_t N_,
                                                  index_t K_,
                                                  index_t StrideA_,
                                                  index_t StrideB_,
                                                  std::array<index_t, NumDTensor> StrideDs_,
                                                  index_t StrideE_)
        : p_a_grid{p_a_grid_},
          p_b_grid{p_b_grid_},
          p_ds_grid{p_ds_grid_},
          p_e_grid{p_e_grid_},
          M{M_},
          N{N_},
          K{K_},
          StrideA{StrideA_},
          StrideB{StrideB_},
          StrideDs{StrideDs_},
          StrideE{StrideE_}
    {
    }

    const void* p_a_grid;
    const void* p_b_grid;
    std::array<const void*, NumDTensor> p_ds_grid;
    void* p_e_grid;
    index_t M;
    index_t N;
    index_t K;
    index_t StrideA;
    index_t StrideB;
    std::array<index_t, NumDTensor> StrideDs;
    index_t StrideE;

    void Print() const
    {
        std::stringstream str;
        for(auto sd : StrideDs)
            str << sd << ",";

        std::cout << "arg {" << "M:" << M << ", " << "N:" << N << ", " << "K:" << K << ", "
                  << "SA:" << StrideA << ", " << "SB:" << StrideB << ", " << "SE:" << StrideE
                  << ", " << "SDs: {" << str.str() << "}" << "}" << std::endl;
    }
};

struct GemmDesc
{
    ck::index_t M_, N_, K_;
    ck::index_t stride_A_, stride_B_, stride_C_;

    std::vector<ck::index_t> stride_Ds_;
};

template <typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename EDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          typename ComputeDataType = ADataType>
struct DeviceGroupedGemm : public BaseOperator
{
    static constexpr index_t NumDTensor = DsDataType::Size();

    static_assert(DsLayout::Size() == DsDataType::Size(), "wrong! inconsistent NumDTensor");

    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(std::vector<const void*>& p_a,
                        std::vector<const void*>& p_b,
                        std::vector<std::array<const void*, NumDTensor>>& p_ds,
                        std::vector<void*>& p_e,
                        std::vector<GemmDesc>& gemm_desc,
                        AElementwiseOperation a_element_op,
                        BElementwiseOperation b_element_op,
                        CElementwiseOperation c_element_op) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;

    //---------------------------------------------------------------------------------------------
    /// @brief      Sets the device kernel arguments pointer and may copy data to device.
    ///
    /// TODO: Add which kernels are using this (TileLoop * FixedNK ??)
    ///
    /// @param      p_arg               The pointer to the Argument we're going to update.
    /// @param[in]  p_dev_kernel_args   The pointer to the device memory which will contain kernel
    ///                                 arguments.
    /// @param[in]  p_host_kernel_args  The pointer to the host memory which contains kernel
    ///                                 arguments that should be copied to device memory.
    ///
    virtual void SetDeviceKernelArgs(BaseArgument* p_arg,
                                     void* p_dev_kernel_args,
                                     const void* p_host_kernel_args) const
    {
        ignore = p_arg;
        ignore = p_dev_kernel_args;
        ignore = p_host_kernel_args;

        std::ostringstream err;
        err << "This function is not implemented by the kernel: " << this->GetTypeString()
            << __FILE__ << ":" << __LINE__ << ", in function: " << __func__;
        throw std::runtime_error(err.str());
    }

    //----------------------------------------------------------------------------------------------
    /// @brief      Sets the device kernel arguments pointer and may copy data to device.
    ///
    /// @param      p_arg              The pointer to the Argument we're going to update.
    /// @param[in]  p_dev_kernel_args  The pointer to the device memory which contains kernel
    ///                                arguments.
    ///
    virtual void SetDeviceKernelArgs(BaseArgument* p_arg, void* p_dev_kernel_args) const
    {
        ignore = p_arg;
        ignore = p_dev_kernel_args;

        std::ostringstream err;
        err << "This function is not implemented by the kernel: " << this->GetTypeString()
            << __FILE__ << ":" << __LINE__ << ", in function: " << __func__;
        throw std::runtime_error(err.str());
    }

    //----------------------------------------------------------------------------------------------
    /// @brief      Gets the device kernel argument size.
    ///
    /// @param[in]  p_arg  The pointer to the Device op Argument.
    ///
    /// @return     The device kernel argument size.
    ///
    virtual size_t GetDeviceKernelArgSize(const BaseArgument* p_arg) const
    {
        ignore = p_arg;

        std::ostringstream err;
        err << "This function is not implemented by the kernel: " << this->GetTypeString()
            << __FILE__ << ":" << __LINE__ << ", in function: " << __func__;
        throw std::runtime_error(err.str());
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
