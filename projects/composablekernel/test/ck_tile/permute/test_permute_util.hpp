// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include "permute.hpp"
#include "ck_tile/host.hpp"

#include <array>
#include <cassert>
#include <cstring>
#include <functional>
#include <numeric>
#include <ostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#if !CK_TILE_USE_WMMA
#ifdef PERMUTE_USE_ALTERNATIVE_IMPL
#include "alternative_impl/matrix_core_swizzle.hpp"
#endif
#endif

namespace detail {
template <int bytes>
struct to_integer_type;

template <>
struct to_integer_type<4>
{
    using type = int32_t;
};
template <>
struct to_integer_type<2>
{
    using type = int16_t;
};
template <>
struct to_integer_type<1>
{
    using type = int8_t;
};
} // namespace detail

template <int bytes>
using to_integer_type = typename detail::to_integer_type<bytes>::type;

// host API (should come from codegen)
template <typename DataType>
float permute(permute_args a, const ck_tile::stream_config& s)
{
    using PipelineProblem = ck_tile::GenericPermuteProblem<DataType>;
    using Kernel          = ck_tile::GenericPermute<PipelineProblem>;

    auto kargs = Kernel::MakeKargs(a);

    const dim3 grids  = Kernel::GridSize(a);
    const dim3 blocks = Kernel::BlockSize();

    float ave_time =
        ck_tile::launch_kernel(s, ck_tile::make_kernel<1>(Kernel{}, grids, blocks, 0, kargs));

    return ave_time;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    using size_type = typename std::vector<T>::size_type;

    os << "[";
    for(size_type idx = 0; idx < v.size(); ++idx)
    {
        if(0 < idx)
        {
            os << ", ";
        }
        os << v[idx];
    }
    return os << "]";
}

// different threshold for different dtype
template <typename DataType>
auto get_elimit(std::string /*init_method*/)
{
    double rtol = 1e-3;
    double atol = 1e-3;
    return ck_tile::make_tuple(rtol, atol);
}

template <>
auto get_elimit<ck_tile::bf16_t>(std::string /*init_method*/)
{
    double rtol = 1e-2;
    double atol = 1e-2;
    return ck_tile::make_tuple(rtol, atol);
}

template <>
auto get_elimit<ck_tile::fp8_t>(std::string init_method)
{
    if(init_method == "ui" || init_method == "ni")
    {
        unsigned max_rounding_point_distance = 0;
        double atol                          = 2e-3;
        return ck_tile::make_tuple(max_rounding_point_distance, atol);
    }
    else
    {
        unsigned max_rounding_point_distance = 1;
        double atol                          = 0.0625;
        return ck_tile::make_tuple(max_rounding_point_distance, atol);
    }
}

// "1,2,3,4" -> vector{1,2,3,4}
std::vector<ck_tile::index_t> decode_vec(std::string q_val)
{
#define _S2I_(str_) static_cast<ck_tile::index_t>(std::atoi((str_).c_str()))
    std::string::size_type pos = 0;
    std::vector<ck_tile::index_t> v;
    while(true)
    {
        auto found = q_val.find(',', pos);
        ck_tile::index_t n =
            _S2I_(q_val.substr(pos, found == std::string::npos ? found : found - pos));
        v.push_back(n);
        if(found == std::string::npos)
        {
            break;
        }
        pos = found + 1;
    }
    return v;
#undef _S2I_
}

template <typename Tuple>
class TestCkTilePermute : public ::testing::Test
{

    protected:
    using DataType = std::tuple_element_t<0, Tuple>;

    void Run(std::vector<ck_tile::index_t>& shape, std::string& perm)
    {
        std::string data_type                  = get_precision_string();
        std::vector<ck_tile::index_t> perm_vec = decode_vec(perm);
        int seed                               = 11939;

        assert(shape.size() == perm_vec.size());
        ck_tile::index_t rank = perm_vec.size();
        if(rank > ck_tile::GenericPermuteHostArgs::kMaxRanks)
        {
            printf("rank %d permute is not support yet\n", rank);
            EXPECT_TRUE(false);
        }

        ck_tile::HostTensor<DataType> x(shape);
        ck_tile::FillUniformDistributionIntegerValue<DataType>{-15, 15, seed}(x);

        std::vector<ck_tile::index_t> y_shape = [&]() {
            std::vector<ck_tile::index_t> tmp(rank, 0);

            for(int i = 0; i < static_cast<int>(rank); i++)
            {
                tmp[i] = shape[perm_vec[i]];
            }

            return tmp;
        }();

        ck_tile::HostTensor<DataType> y(y_shape);

        ck_tile::DeviceMem x_buf(x.get_element_space_size_in_bytes());
        ck_tile::DeviceMem y_buf(y.get_element_space_size_in_bytes());

        x_buf.ToDevice(x.data());

        std::cout << "[" << data_type << "] shape:" << shape << "->" << y_shape
                  << ", permute:" << perm_vec << std::endl;

        ck_tile::stream_config stream_config{nullptr, false, 0, 0, 1};

        auto run_permute = [&]() {
            permute_args a;
            a.p_src = x_buf.GetDeviceBuffer();
            a.p_dst = y_buf.GetDeviceBuffer();
            a.rank  = rank;
            std::copy(shape.begin(), shape.end(), a.shape);
            std::copy(perm_vec.begin(), perm_vec.end(), a.perm);

            return permute<DataType>(a, stream_config);
        };
#if !CK_TILE_USE_WMMA
#ifdef PERMUTE_USE_ALTERNATIVE_IMPL
        // batch* n0*n1*n2*k0*k1*k2 -> batch* n0*k0*n1*k1*n2*k2
        if((perm == std::string("0,1,4,2,5,3,6") || perm == std::string("0,1,2,4,5,3,6") ||
            perm == std::string("0,1,3,4,2,5")))
        {
            if(perm == std::string("0,1,3,4,2,5"))
            {
                // b_nr_kr_kw_nw_kv = 2,   // 0,1,3,4,2,5
                matrix_core_swizzle_traits t;
                t.permute = perm;

                matrix_core_swizzle_args a;
                a.p_src = x_buf.GetDeviceBuffer();
                a.p_dst = y_buf.GetDeviceBuffer();
                a.batch = shape[0];

                auto nr = shape[1];
                auto nw = shape[2];
                auto kr = shape[3];
                auto kw = shape[4];
                auto kv = shape[5];
                a.n     = nr * nw;
                a.k     = kr * kw * kv;
                if(kv == 8 && kw == 4 && nw == 16 && nr % 4 == 0 && kr % 8 == 0)
                {
                    t.inst = "16x16x16";
                    std::cout << ", matrix_core_swizzle_waveflatten_" << t.inst << std::flush;

                    matrix_core_swizzle<DataType>(t, a, stream_config);
                }
                else if(kv == 8 && kw == 2 && nw == 32 && nr % 4 == 0 && kr % 8 == 0)
                {
                    t.inst = "32x32x8";
                    std::cout << ", matrix_core_swizzle_waveflatten_" << t.inst << std::flush;

                    matrix_core_swizzle<DataType>(t, a, stream_config);
                }
                else
                {
                    run_permute();
                }
            }
            else
            {
                matrix_core_swizzle_traits t;
                t.permute = perm;

                matrix_core_swizzle_args a;
                a.p_src = x_buf.GetDeviceBuffer();
                a.p_dst = y_buf.GetDeviceBuffer();
                a.batch = shape[0];
                a.n     = shape[1] * shape[2] * shape[3];
                a.k     = shape[4] * shape[5] * shape[6];
                if(shape[6] == 8 && shape[3] == 32 && shape[5] == 2 && shape[2] == 4 &&
                   shape[4] % 8 == 0 && shape[1] % 2 == 0)
                {
                    // 32x32x8 inst
                    // perm=0,1,4,2,5,3,6
                    // y_shape=*,2x,8x,4,2,32,8 (3,6,16,4,2,32,8)
                    // shape = *,2x,4,32,8x,2,8 (3,6,4,32,16,2,8)

                    t.inst = "32x32x8";
                    std::cout << ", matrix_core_swizzle_" << t.inst << std::flush;

                    matrix_core_swizzle<DataType>(t, a, stream_config);
                }
                else if(shape[6] == 8 && shape[3] == 16 && shape[5] == 4 && shape[2] == 4 &&
                        shape[4] % 4 == 0 && shape[1] % 4 == 0)
                {
                    // 16x16x16 inst
                    // perm=0,1,4,2,5,3,6
                    // y_shape=*,4x,4x,4,4,16,8
                    // shape = *,4x,4,16,4x,4,8 (3,8,4,16,16,4,8)
                    t.inst = "16x16x16";
                    std::cout << ", matrix_core_swizzle_" << t.inst << std::flush;

                    matrix_core_swizzle<DataType>(t, a, stream_config);
                }
                else
                {
                    run_permute();
                }
            }
        }
        else
#endif
#endif
        {
            run_permute();
        }

        bool pass = true;

        // Do Validation
        reference_permute(x, y, perm_vec);

        ck_tile::HostTensor<DataType> y_dev(y.get_lengths());

        y_buf.FromDevice(y_dev.data());

        pass = std::equal(
            y_dev.begin(), y_dev.end(), y.begin(), [&](const DataType& d, const DataType& h) {
                using itype = to_integer_type<sizeof(DataType)>;
                itype i_d   = ck_tile::bit_cast<itype>(d);
                itype i_h   = ck_tile::bit_cast<itype>(h);
                return i_d == i_h;
            });
        std::cout << ", valid:" << (pass ? "y" : "n") << std::flush;

        std::cout << std::endl;

        EXPECT_TRUE(pass);
    }

    static std::string get_precision_string()
    {
        if constexpr(std::is_same_v<DataType, ck_tile::fp16_t>)
        {
            return "fp16";
        }
        else if(std::is_same_v<DataType, ck_tile::fp8_t>)
        {
            return "fp8";
        }
        else if(std::is_same_v<DataType, float>)
        {
            return "fp32";
        }
        else
        {
            throw std::runtime_error("invalid precision");
        }
    }
};
