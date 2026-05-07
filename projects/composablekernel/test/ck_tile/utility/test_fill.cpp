// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/host/fill.hpp"
#include "ck_tile/host/joinable_thread.hpp"
#include <chrono>
#include <cstring>
#include <gtest/gtest.h>
#include <vector>

using namespace ck_tile;

namespace test {

// Test fixture for FillUniformDistribution tests
template <typename T>
class FillUniformDistributionTest : public ::testing::Test
{
    public:
    static constexpr uint32_t seed = 42;
    static constexpr float a       = -5.0f;
    static constexpr float b       = 5.0f;
};

using TestTypes = ::testing::Types<float, fp16_t, fp8_t, pk_fp4_t>;
TYPED_TEST_SUITE(FillUniformDistributionTest, TestTypes);

// Test that multiple runs with the same seed produce identical results
#ifndef _WIN32
TYPED_TEST(FillUniformDistributionTest, ConsistencyWithSameSeed)
{
    using T         = TypeParam;
    const auto a    = this->a;
    const auto b    = this->b;
    const auto seed = this->seed;

    constexpr size_t size = 1024 * 1024 * 1024 / sizeof(T); // 1G

    std::vector<T> vec1(size);
    auto start = std::chrono::high_resolution_clock::now();
    FillUniformDistribution<T>{a, b, seed}(vec1.begin(), vec1.end());
    auto end   = std::chrono::high_resolution_clock::now();
    double sec = std::chrono::duration<double>(end - start).count();
    std::cout << "Taking " << sec << " sec to fill 1GB of data of type " << typeid(T).name()
              << std::endl;

    const auto cpu_cores = max(32U, get_available_cpu_cores());
    for(auto num_threads_diff : {-3, -1})
    {
        cpu_core_guard cg(min(max(cpu_cores + num_threads_diff, 1U), get_available_cpu_cores()));
        std::vector<T> vec2(size);
        FillUniformDistribution<T>{a, b, seed}(vec2.begin(), vec2.end());
        EXPECT_EQ(0, std::memcmp(vec1.data(), vec2.data(), size * sizeof(T)))
            << "First and second fill should be identical";
    }
}
#endif

// Test consistency across different data sizes (which affects threading)
TYPED_TEST(FillUniformDistributionTest, ConsistencyAcrossSizes)
{
    using T         = TypeParam;
    const auto a    = this->a;
    const auto b    = this->b;
    const auto seed = this->seed;

    std::vector<size_t> test_sizes = {
        100,     // Small - likely single threaded
        10000,   // Medium
        1000000, // Large - will use multiple threads
        5000000  // Very large - will use many threads
    };

    for(size_t size : test_sizes)
    {
        std::vector<T> reference(size);
        std::vector<T> test_vec(size);

        FillUniformDistribution<T>{a, b, seed}(reference.begin(), reference.end());

        // Run multiple times to ensure consistency
        for(int run = 0; run < 3; ++run)
        {
            std::fill(test_vec.begin(), test_vec.end(), T{});
            FillUniformDistribution<T>{a, b, seed}(test_vec.begin(), test_vec.end());

            EXPECT_EQ(0, std::memcmp(reference.data(), test_vec.data(), size * sizeof(T)))
                << "Mismatch for size=" << size << " run=" << run;
        }
    }
}

// Test that different seeds produce different results
TYPED_TEST(FillUniformDistributionTest, CommonPrefix)
{
    using T         = TypeParam;
    const auto a    = this->a;
    const auto b    = this->b;
    const auto seed = this->seed;

    std::vector<size_t> test_sizes = {
        100,     // Small - likely single threaded
        10000,   // Medium
        1000000, // Large - will use multiple threads
        5000000  // Very large - will use many threads
    };

    auto longest = std::make_unique<std::vector<T>>(test_sizes[0]);
    FillUniformDistribution<T>{a, b, seed}(longest->begin(), longest->end());
    for(size_t i = 1; i < test_sizes.size(); ++i)
    {
        auto current = std::make_unique<std::vector<T>>(test_sizes[i]);
        FillUniformDistribution<T>{a, b, seed}(current->begin(), current->end());
        size_t min_size = std::min(longest->size(), current->size());
        EXPECT_EQ(0, std::memcmp(longest->data(), current->data(), min_size * sizeof(T)))
            << "Different sizes with same seed should have the same prefix";
        if(current->size() > longest->size())
        {
            longest = std::move(current);
        }
    }
}

// Test edge cases
TYPED_TEST(FillUniformDistributionTest, EdgeCases)
{
    using T         = TypeParam;
    const auto a    = this->a;
    const auto b    = this->b;
    const auto seed = this->seed;

    // Empty range
    std::vector<T> empty_vec;
    EXPECT_NO_THROW((FillUniformDistribution<T>{a, b, seed}(empty_vec.begin(), empty_vec.end())));

    // Single element
    std::vector<T> single1(1);
    std::vector<T> single2(1);
    FillUniformDistribution<T>{a, b, seed}(single1.begin(), single1.end());
    FillUniformDistribution<T>{a, b, seed}(single2.begin(), single2.end());

    EXPECT_EQ(0, std::memcmp(single1.data(), single2.data(), sizeof(T)))
        << "Single element should be consistent";

    // Small sizes that might affect threading decisions
    std::vector<size_t> small_sizes = {2, 3, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65};
    for(size_t size : small_sizes)
    {
        std::vector<T> vec1(size);
        std::vector<T> vec2(size);
        FillUniformDistribution<T>{a, b, seed}(vec1.begin(), vec1.end());
        FillUniformDistribution<T>{a, b, seed}(vec2.begin(), vec2.end());

        EXPECT_EQ(0, std::memcmp(vec1.data(), vec2.data(), size * sizeof(T)))
            << "Edge case failed for size=" << size;
    }
}
} // namespace test
