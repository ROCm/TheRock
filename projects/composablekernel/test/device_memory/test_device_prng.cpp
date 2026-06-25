// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <vector>
#include <gtest/gtest.h>
#include <random>
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/utility/common_header.hpp"
#include "ck/ck.hpp"

template <typename inType, typename outType>
void convertTypeFromDevice(std::vector<inType>& fromDevice,
                           std::vector<outType>& res,
                           uint64_t num_elements)
{
    for(uint64_t i = 0; i < num_elements / ck::packed_size_v<inType>; i++)
    {
        // since the CPU dosen't have non-standard data types, we need to convert to float
        if constexpr(ck::is_same_v<ck::remove_cvref_t<inType>, ck::f4x2_pk_t>)
        {
            ck::float2_t tmp = ck::type_convert<ck::float2_t, ck::f4x2_t>(fromDevice[i]);
            res[i * 2]       = tmp.x;
            res[i * 2 + 1]   = tmp.y;
        }
        else if constexpr(ck::is_same_v<ck::remove_cvref_t<inType>, ck::pk_i4_t>)
        {
            uint8_t packed = fromDevice[i].data;

            int hi         = (packed >> 4) & 0x0f;
            int lo         = packed & 0x0f;
            res[i * 2]     = static_cast<outType>(hi - 8);
            res[i * 2 + 1] = static_cast<outType>(lo - 8);
        }
        else
        {
            res[i] = ck::type_convert<outType, inType>(fromDevice[i]);
        }
    }
}

template <typename T>
void TDevRanUniGenInt(int min_val, int max_val, uint64_t num_elements)
{

    size_t packed_size = ck::packed_size_v<T>;

    ck::DeviceMem test_buf(sizeof(T) * num_elements / packed_size);
    std::vector<T> from_host(num_elements / packed_size);
    std::vector<int> host_elements(num_elements);

    test_buf.FillUniformRandInteger<T>(min_val, max_val);
    test_buf.FromDevice(&from_host[0]);

    uint64_t num_equal = 0;
    bool in_range      = true;

    convertTypeFromDevice(from_host, host_elements, num_elements);
    // very basic checks: check if all data points are in range and
    // hf data is within 6 sigma of expected value
    for(uint64_t i = 0; i < num_elements; i++)
    {
        if(host_elements[i] >= max_val || host_elements[i] < min_val)
        {
            in_range = false;
        }
        if(i > 0)
        {
            if(host_elements[i] == host_elements[i - 1])
            {
                num_equal++;
            }
        }
    }
    EXPECT_TRUE(in_range);

    double expected_mean =
        (static_cast<double>(num_elements) - 1.0) / (static_cast<double>(max_val - min_val));
    double std_dev     = std::sqrt(expected_mean);
    double upper_bound = expected_mean + 6 * std_dev;
    double lower_bound = expected_mean - 6 * std_dev;

    // in these cases the test parameters are unsuitable
    EXPECT_TRUE(lower_bound > 1.0);
    EXPECT_TRUE(upper_bound < static_cast<double>(num_elements) - 2.0);

    // printf("lower bound: %f upper bound: %f actual: %d\n",
    //        lower_bound,
    //        upper_bound,
    //        static_cast<int>(num_equal));
    EXPECT_TRUE(static_cast<double>(num_equal) > lower_bound);
    EXPECT_TRUE(static_cast<double>(num_equal) < upper_bound);
}

template <typename T>
void TDevRanUniGenFp(double min_val,
                     double max_val,
                     uint64_t num_elements,
                     double std_err_tolerance = 6.0)
{
    size_t packed_size = ck::packed_size_v<T>;
    ck::DeviceMem test_buf(sizeof(T) * num_elements / packed_size);
    std::vector<T> host_buf(num_elements / packed_size);
    std::vector<float> host_elements(num_elements);

    test_buf.FillUniformRandFp<T>(min_val, max_val);
    test_buf.FromDevice(&host_buf[0]);

    bool in_range         = true;
    double accum_mean     = 0.0;
    double accum_variance = 0.0;

    // #kabraham: with floats, we can actually do some more extensive tests,
    //  compute mean, std_dev and std_err and compare these to expected values
    convertTypeFromDevice(host_buf, host_elements, num_elements);
    for(uint64_t i = 0; i < num_elements; i++)
    {
        if(host_elements[i] > max_val || host_elements[i] < min_val)
        {
            in_range = false;
        }
        accum_mean += host_elements[i];
    }
    EXPECT_TRUE(in_range);
    EXPECT_TRUE(accum_mean != 0.0);
    double mean = accum_mean / num_elements;

    for(uint64_t i = 0; i < num_elements; i++)
    {
        accum_variance += std::pow(host_elements[i] - mean, 2);
    }
    double std_dev = std::sqrt(accum_variance) / num_elements;

    double expected_mean    = (min_val + max_val) / 2.0;
    double expected_std_dev = (max_val - min_val) / std::sqrt(12 * num_elements);
    double std_err          = expected_std_dev / sqrt(num_elements);
    //    printf(
    //        "Expected: mean: %f std_dev: %f std_err : %f\n", expected_mean, expected_std_dev,
    //        std_err);
    //    printf("  Actual: mean: %f std_dev: %f \n", mean, std_dev);
    EXPECT_TRUE(abs(mean - expected_mean) < 6 * expected_std_dev);
    EXPECT_TRUE(abs(std_dev - expected_std_dev) < std_err_tolerance * std_err);
}

template <typename T>
void TDevRanNormGenFp(double sigma,
                      double mean,
                      uint64_t num_elements,
                      double ERRF_BUCKET_SIZE  = 0.1,
                      double ERRF_BUCKET_RANGE = 3.0,
                      double sig_tolerence     = 6.0)
{
    ck::DeviceMem test_buf(sizeof(T) * num_elements);
    std::vector<T> host_buf(num_elements);
    std::vector<float> host_elements(num_elements);

    test_buf.FillNormalRandFp<T>(sigma, mean);
    test_buf.FromDevice(&host_buf[0]);

    convertTypeFromDevice(host_buf, host_elements, num_elements);

    // #kabraham: compute errf buckets and compare with expected vaules
    int ERRF_NUM_BUCKETS = 2 * ERRF_BUCKET_RANGE / ERRF_BUCKET_SIZE + 1;

    std::vector<int64_t> errf_buckets(ERRF_NUM_BUCKETS, 0);
    for(uint64_t i = 0; i < num_elements; i++)
    {
        for(int bucket = 0; bucket < ERRF_NUM_BUCKETS; bucket++)
        {
            // #kabraham: count exact hits as half (kind of relevant for utra-low-precision formats)
            if(host_elements[i] < sigma * (-ERRF_BUCKET_RANGE + bucket * ERRF_BUCKET_SIZE) + mean)
            {
                errf_buckets[bucket] += 2;
            }
            else if(host_elements[i] <=
                    sigma * (-ERRF_BUCKET_RANGE + bucket * ERRF_BUCKET_SIZE) + mean)
            {
                errf_buckets[bucket] += 1;
            }
        }
    }

    for(int bucket = 0; bucket < ERRF_NUM_BUCKETS; bucket++)
    {
        double expected_num_entries =
            (std::erfc((ERRF_BUCKET_RANGE - bucket * ERRF_BUCKET_SIZE) / std::sqrt(2))) * 0.5 *
            num_elements;
        double noise_range = std::sqrt(expected_num_entries);
        //     printf("Expected for bucket %d: %d. Actual: %d \n",
        //            bucket,
        //            static_cast<int>(expected_num_entries),
        //            static_cast<int>(errf_buckets[bucket] / 2));
        EXPECT_TRUE(errf_buckets[bucket] / 2 >= expected_num_entries - sig_tolerence * noise_range);
        EXPECT_TRUE(errf_buckets[bucket] / 2 <= expected_num_entries + sig_tolerence * noise_range);
    }
}

TEST(TDevIntegerRanUniGen, U8) { TDevRanUniGenInt<uint8_t>(0, 2, 15000); }
TEST(TDevIntegerRanUniGen, U16) { TDevRanUniGenInt<uint16_t>(0, 100, 100000); }
TEST(TDevIntegerRanUniGen, U32) { TDevRanUniGenInt<uint32_t>(0, 10000, 10000000); }
TEST(TDevIntegerRanUniGen, I4) { TDevRanUniGenInt<ck::pk_i4_t>(-2, 2, 10000000); }

TEST(TDevIntegerRanUniGen, F32) { TDevRanUniGenInt<float>(-2, 2, 10000000); }
TEST(TDevIntegerRanUniGen, F16) { TDevRanUniGenInt<ck::half_t>(-2, 2, 1000000); }

TEST(TDevFpRanUniGen, F32_1) { TDevRanUniGenFp<float>(0, 1, 100000); }
TEST(TDevFpRanUniGen, F32_2) { TDevRanUniGenFp<float>(0, 37, 73000); }
TEST(TDevFpRanUniGen, F32_3) { TDevRanUniGenFp<float>(-2, 1, 84000); }

TEST(TDevFpRanUniGen, F16) { TDevRanUniGenFp<ck::half_t>(-1, 1, 100000); }
TEST(TDevFpRanUniGen, BF16) { TDevRanUniGenFp<ck::bhalf_t>(0, 2, 100000); }
TEST(TDevFpRanUniGen, F8) { TDevRanUniGenFp<ck::f8_t>(0, 2, 100000); }
TEST(TDevFpRanUniGen, BF8) { TDevRanUniGenFp<ck::bf8_t>(-5, 5, 100000); }
TEST(TDevFpRanUniGen, F4) { TDevRanUniGenFp<ck::f4x2_pk_t>(-5, 5, 100000, 20.0); }

TEST(TDevRanNormGenFp, F32_1) { TDevRanNormGenFp<float>(1, 0, 1000000); }
TEST(TDevRanNormGenFp, F32_2) { TDevRanNormGenFp<float>(5, -2, 10000000, 0.2, 5.0); }

TEST(TDevRanNormGenFp, F16) { TDevRanNormGenFp<ck::half_t>(5, -2, 100000); }
TEST(TDevRanNormGenFp, BF16) { TDevRanNormGenFp<ck::bhalf_t>(5, -2, 100000); }

TEST(TDevRanNormGenFp, F8) { TDevRanNormGenFp<ck::f8_t>(2, 0, 100000, 0.5, 2.0, 10.0); }
TEST(TDevRanNormGenFp, BF8) { TDevRanNormGenFp<ck::bf8_t>(16, 0, 100000, 0.5, 2.0, 30.0); }

TEST(TDevRanNormGenFp, F4) { TDevRanNormGenFp<ck::f4x2_pk_t>(2, 0, 100000, 0.5, 3.0, 30.0); }
