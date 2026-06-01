// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once
namespace ck {

// Do following things to avoid "alloca" in LLVM-IR, which would cause scratch memory
// and sometimes useless instructions:
//   1. Don't save a reference to tensor descriptor in class, pass in tensor descriptor as argument
//   instead
//   2. Don't construct a new tensor coordinate everytime when using it, update and reuse the same
//   tensor coordinate instead
//   3. Don't use a pointer to VGPR buffer, use vector instead

namespace detail {
// TODO: How to fix this? It uses an struct instead of lambda because lambda
// doesn't have constructor
template <index_t VectorDim, index_t ScalarPerVector>
struct lambda_scalar_per_access
{
    __host__ __device__ constexpr auto operator()(index_t i) const
    {
        return (i == VectorDim) ? ScalarPerVector : 1;
    }
};

template <index_t VectorDim>
struct lambda_scalar_step_in_vector
{
    __host__ __device__ constexpr auto operator()(index_t i) const
    {
        return (i == VectorDim) ? 1 : 0;
    }
};

// TODO: How to fix this? It uses an struct instead of lambda because lambda
// doesn't have constructor
template <index_t SrcVectorDim,
          index_t SrcScalarPerVector,
          index_t DstVectorDim,
          index_t DstScalarPerVector>
struct lambda_scalar_per_access_for_src_and_dst
{
    __host__ __device__ constexpr auto operator()(index_t i) const
    {
        if(i == SrcVectorDim && i == DstVectorDim)
        {
            return math::lcm(SrcScalarPerVector, DstScalarPerVector);
        }
        else if(i == SrcVectorDim)
        {
            return SrcScalarPerVector;
        }
        else if(i == DstVectorDim)
        {
            return DstScalarPerVector;
        }
        else
        {
            return 1;
        }
    }
};

template <index_t WaveNum, index_t nDim>
struct lambda_wave_cluster_dimension
{
    __host__ __device__ constexpr auto operator()(index_t i) const
    {
        if((nDim - i) == 3)
            return WaveNum;
        else
            return 1;
    }
};

} // namespace detail

} // namespace ck
