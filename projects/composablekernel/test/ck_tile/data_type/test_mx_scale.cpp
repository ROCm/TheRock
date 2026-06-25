// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"
#include <hip/hip_runtime.h>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"

using ck_tile::bf16_t;
using ck_tile::bf16x2_t;
using ck_tile::fp16_t;
using ck_tile::fp16x2_t;
using ck_tile::fp32_t;
using ck_tile::fp32x2_t;
using ck_tile::number;
using ck_tile::pk_fp4_t;

template <typename SRC, typename DST, bool is_device>
CK_TILE_HOST void test_convert();

using ck_tile::e8m0_raw_t;
using ck_tile::e8m0_t;

TEST(OCP_Scale, NumericLimits)
{
    EXPECT_EQ(ck_tile::numeric<e8m0_t>::has_inf(), false);
    EXPECT_EQ(ck_tile::numeric<e8m0_t>::zero(), ck_tile::numeric<e8m0_t>::signaling_NaN());
    EXPECT_EQ(ck_tile::numeric<e8m0_t>::min(), e8m0_t{e8m0_raw_t{0b00000000}});
    EXPECT_EQ(ck_tile::numeric<e8m0_t>::max(), e8m0_t{e8m0_raw_t{0b11111110}});
}
TEST(OCP_Scale, NumericBasic)
{
    auto scale_1 = e8m0_t{1.0f};
    auto scale_2 = e8m0_t{e8m0_raw_t{ck_tile::numeric_traits<e8m0_t>::bias}}; // 2^0
    EXPECT_EQ(scale_1, scale_2);

    auto scale_3 = e8m0_t{8.0f};
    auto scale_4 = e8m0_t{e8m0_raw_t{3 + ck_tile::numeric_traits<e8m0_t>::bias}}; // 2^3
    EXPECT_EQ(scale_3, scale_4);
}

TEST(OCP_Scale, ScaledConvertDevice)
{
    constexpr bool is_device = true;
    test_convert<fp32_t, fp32_t, is_device>(); // fp32 -> fp4 -> fp32
    test_convert<fp16_t, fp16_t, is_device>();
    test_convert<bf16_t, bf16_t, is_device>();
    test_convert<fp32_t, fp16_t, is_device>();
    test_convert<fp32_t, bf16_t, is_device>();
    test_convert<fp16_t, fp32_t, is_device>();
    test_convert<bf16_t, fp32_t, is_device>();
}
TEST(OCP_Scale, ScaledConvertHost)
{
    constexpr bool is_device = false;
    test_convert<fp32_t, fp32_t, is_device>(); // fp32 -> fp4 -> fp32
    test_convert<fp16_t, fp16_t, is_device>();
    test_convert<bf16_t, bf16_t, is_device>();
    test_convert<fp32_t, fp16_t, is_device>();
    test_convert<fp32_t, bf16_t, is_device>();
    test_convert<fp16_t, fp32_t, is_device>();
    test_convert<bf16_t, fp32_t, is_device>();
}
TEST(OCP_Scale, tensorInit)
{
    using scale_t = e8m0_t;
    ck_tile::HostTensor<scale_t> scales({10, 10});
    ck_tile::FillUniformDistribution<scale_t>{1.f, 1.f}(scales);
    scales.SetZero();
}

#define toPF4(x, y) ck_tile::scaled_type_convert<pk_fp4_t>(x, y)
#define toDST(x, y) ck_tile::scaled_type_convert<DST>(x, y)
#define toDSTx2(x, y) ck_tile::scaled_type_convert<DSTx2_t>(x, y)

#define toF32(x) ck_tile::type_convert<float>(x)
#define toPF4_(x) ck_tile::type_convert<pk_fp4_t>(x)
#define toSRC(x) ck_tile::type_convert<SRC>(x)
#define toDST_(x) ck_tile::type_convert<DST>(x)

template <typename Kernel, typename... Args>
__global__ void MyKernel(Args... args)
{
    Kernel{}(args...);
}
template <typename SRC, typename DST, int N>
struct SrcPkfp4Dst
{
    CK_TILE_HOST_DEVICE void
    operator()(const SRC* src, DST* dst, e8m0_t scale1, e8m0_t scale2) const
    {

        using SRCx2_t = ck_tile::ext_vector_t<SRC, 2>;
        using DSTx2_t = ck_tile::ext_vector_t<DST, 2>;

        ck_tile::static_for<0, N, 2>{}([&](auto i) {
            const auto input2 = SRCx2_t{src[i], src[i + 1]};

            if(i % 4 == 0)
            {
                // ex: fp32_t -> fp4 -> bf16_t
                dst[i] = toDST(toPF4(src[i], scale1), scale2);
                // ex: fp32x2_t -> pk_fp4 -> unpack<0> -> bf16_t
                dst[i + 1] = toDST(toPF4_(toPF4(input2, scale1).unpack(number<1>{})), scale2);
            }
            else
            {
                // ex: fp32x2_t -> pk_fp4_t -> bf16x2_t
                reinterpret_cast<DSTx2_t*>(dst)[i >> 1] = toDSTx2(toPF4(input2, scale1), scale2);
            }
        });
    }
};

template <typename SRC, typename DST, bool is_device>
CK_TILE_HOST void test_convert()
{
    const auto test_data = std::array{4.f, 6.f, 8.f, 10.f};
    const auto ref_data  = std::array{8.f, 16.f, 16.f, 16.f};
    const auto scale1    = e8m0_t{8.0f};
    const auto scale2    = e8m0_t{16.0f};

    static_assert(test_data.size() == ref_data.size());
    static_assert(test_data.size() % 2 == 0);

    constexpr int N = test_data.size();
    std::array<SRC, N> in;
    std::array<DST, N> ref, out;

    // prepare input and ground truth in host
    for(int i = 0; i < N; ++i)
    {
        in[i]  = toSRC(test_data[i]);
        ref[i] = toDST_(ref_data[i]);
        EXPECT_EQ(test_data[i], toF32(in[i]));
        EXPECT_EQ(ref_data[i], toF32(ref[i]));
    }

    using job = SrcPkfp4Dst<SRC, DST, N>;

    if constexpr(is_device)
    {
        auto in_d  = std::make_unique<ck_tile::DeviceMem>(in.size() * sizeof(SRC));
        auto out_d = std::make_unique<ck_tile::DeviceMem>(out.size() * sizeof(DST));
        in_d->ToDevice(in.data());

        MyKernel<job><<<1, 1>>>(reinterpret_cast<const SRC*>(in_d->GetDeviceBuffer()),
                                reinterpret_cast<DST*>(out_d->GetDeviceBuffer()),
                                scale1,
                                scale2);

        out_d->FromDevice(out.data());
    }
    else
    {
        job{}(in.data(), out.data(), scale1, scale2);
    }

    for(int i = 0; i < N; ++i)
        EXPECT_EQ(ref[i], out[i]) << "i:" << i;
}
