// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"
#include <vector>
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

TEST(PackedFp4, NumericLimits)
{
    EXPECT_EQ(ck_tile::numeric<pk_fp4_t>::has_inf(), false);
    EXPECT_EQ(ck_tile::numeric<pk_fp4_t>::zero(), pk_fp4_t{0b00000000});
    EXPECT_EQ(ck_tile::numeric<pk_fp4_t>::min(), pk_fp4_t{0b00100010});
    EXPECT_EQ(ck_tile::numeric<pk_fp4_t>::max(), pk_fp4_t{0b01110111});
    EXPECT_EQ(ck_tile::numeric<pk_fp4_t>::lowest(), pk_fp4_t{0b11111111});
    EXPECT_EQ(ck_tile::numeric<pk_fp4_t>::epsilon(), pk_fp4_t{0b00010001});
    EXPECT_EQ(ck_tile::numeric<pk_fp4_t>::round_error(), pk_fp4_t{0b00010001});
}
TEST(PackedFp4, fill)
{
    std::vector<pk_fp4_t> v_fp4(4);
    ck_tile::FillUniformDistribution<pk_fp4_t>{1.f, 1.f}(v_fp4);
    EXPECT_EQ(v_fp4[0].get(), pk_fp4_t{0b00100010}.get());
}
TEST(PackedFp4, ConvertBasic)
{
    EXPECT_EQ(ck_tile::convert_to_type<pk_fp4_t>(0.0f), pk_fp4_t{0b00000000}.get());
    EXPECT_EQ(ck_tile::convert_to_type<pk_fp4_t>(-0.0f), pk_fp4_t{0b00001000}.get());
    EXPECT_EQ(ck_tile::convert_to_type<pk_fp4_t>(-1.0f), pk_fp4_t{0b00001010}.get());
    EXPECT_EQ(ck_tile::type_convert<pk_fp4_t>(0.0f), pk_fp4_t{0b00000000});
    EXPECT_EQ(ck_tile::type_convert<pk_fp4_t>(-0.0f), pk_fp4_t{0b00001000});
    EXPECT_EQ(ck_tile::type_convert<pk_fp4_t>(-1.0f), pk_fp4_t{0b00001010});
    EXPECT_EQ(pk_fp4_t(0.0f), pk_fp4_t{0b00000000});
    EXPECT_EQ(pk_fp4_t(-0.0f), pk_fp4_t{0b00001000});
    EXPECT_EQ(pk_fp4_t(-1.0f), pk_fp4_t{0b00001010});
    EXPECT_EQ(pk_fp4_t{0.0f}, pk_fp4_t{0b00000000});
    EXPECT_EQ(pk_fp4_t{-0.0f}, pk_fp4_t{0b00001000});
    EXPECT_EQ(pk_fp4_t{-1.0f}, pk_fp4_t{0b00001010});
}
TEST(PackedFp4, NumericBasic)
{
    auto f1  = pk_fp4_t{1.5f};
    auto f2  = pk_fp4_t{3.0f};
    auto ref = pk_fp4_t{-1.5f};
    EXPECT_EQ(f1 - f2, ref);
}
TEST(PackedFp4, ConvertDevice)
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
TEST(PackedFp4, ConvertHost)
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

#define toF32(x) ck_tile::type_convert<float>(x)
#define toPF4(x) ck_tile::type_convert<pk_fp4_t>(x)
#define toSRC(x) ck_tile::type_convert<SRC>(x)
#define toDST(x) ck_tile::type_convert<DST>(x)
#define toDSTx2(x) ck_tile::type_convert<DSTx2_t>(x)

template <typename Kernel, typename... Args>
__global__ void MyKernel(Args... args)
{
    Kernel{}(args...);
}
template <typename SRC, typename DST, int N>
struct SrcPkfp4Dst
{
    CK_TILE_HOST_DEVICE void operator()(const SRC* src, DST* dst) const
    {

        using SRCx2_t = ck_tile::ext_vector_t<SRC, 2>;
        using DSTx2_t = ck_tile::ext_vector_t<DST, 2>;

        ck_tile::static_for<0, N, 2>{}([&](auto i) {
            const auto input2 = SRCx2_t{src[i], src[i + 1]};

            if(i % 4 == 0)
            {
                // ex: fp32_t -> fp4 -> bf16_t
                dst[i] = toDST(toPF4(src[i]));
                // ex: fp32x2_t -> pk_fp4 -> unpack<0> -> bf16_t
                dst[i + 1] = toDST(toPF4(input2).unpack(number<1>{}));
            }
            else
            {
                // ex: fp32x2_t -> pk_fp4_t -> bf16x2_t
                reinterpret_cast<DSTx2_t*>(dst)[i >> 1] = toDSTx2(toPF4(input2));
            }
        });
    }
};

template <typename SRC, typename DST, bool is_device>
CK_TILE_HOST void test_convert()
{
    const auto test_data = std::array{0.f,  0.25f,  0.5f,  0.75f,  1.f,  1.25f,  1.5f,    1.75f,
                                      -0.f, -0.25f, -0.5f, -0.75f, -1.f, -1.25f, -1.5f,   -1.75f,
                                      2.f,  2.5f,   3.f,   3.5f,   4.f,  5.f,    5.0625f, 6.f};
    const auto ref_data =
        std::array{0.f,  0.f,  0.5f,  1.f,  1.f, 1.f, 1.5f, 2.f, -0.f, -0.f, -0.5f, -1.f,
                   -1.f, -1.f, -1.5f, -2.f, 2.f, 2.f, 3.f,  4.f, 4.f,  4.f,  6.f,   6.f};

    static_assert(test_data.size() == ref_data.size());
    static_assert(test_data.size() % 2 == 0);

    constexpr int N = test_data.size();
    std::array<SRC, N> in;
    std::array<DST, N> ref, out;

    // prepare input and ground truth in host
    for(int i = 0; i < N; ++i)
    {
        in[i]  = toSRC(test_data[i]);
        ref[i] = toDST(ref_data[i]);
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
                                reinterpret_cast<DST*>(out_d->GetDeviceBuffer()));

        out_d->FromDevice(out.data());
    }
    else
    {
        job{}(in.data(), out.data());
    }

    for(int i = 0; i < N; ++i)
        EXPECT_EQ(ref[i], out[i]) << "i:" << i;
}
