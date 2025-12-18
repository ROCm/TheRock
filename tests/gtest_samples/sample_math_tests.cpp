// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/**
 * Sample Math Tests with Logging
 * ===============================
 * Demonstrates GTest integration with TheRock logging framework
 */

#include <gtest/gtest.h>
#include "test_logging.hpp"
#include <cmath>
#include <stdexcept>

// Test fixture with logging support
class MathTest : public ::testing::Test {
protected:
    TestLogger logger;
    
    void SetUp() override {
        logger = TestLogger("MathTest");
        logger.info("Setting up MathTest");
    }
    
    void TearDown() override {
        logger.info("Tearing down MathTest");
    }
};

// Basic arithmetic tests
TEST_F(MathTest, Addition) {
    logger.info("Testing addition");
    
    int a = 5, b = 3;
    int result = a + b;
    
    logger.debug("Testing: {} + {} = {}", a, b, result);
    EXPECT_EQ(result, 8);
    
    logger.info("Addition test passed");
}

TEST_F(MathTest, Subtraction) {
    logger.info("Testing subtraction");
    
    int a = 10, b = 4;
    int result = a - b;
    
    logger.debug("Testing: {} - {} = {}", a, b, result);
    EXPECT_EQ(result, 6);
    
    logger.info("Subtraction test passed");
}

TEST_F(MathTest, Multiplication) {
    logger.info("Testing multiplication");
    
    int a = 7, b = 6;
    int result = a * b;
    
    logger.debug("Testing: {} * {} = {}", a, b, result);
    EXPECT_EQ(result, 42);
    
    logger.info("Multiplication test passed");
}

TEST_F(MathTest, Division) {
    logger.info("Testing division");
    
    double a = 15.0, b = 3.0;
    double result = a / b;
    
    logger.debug("Testing: {} / {} = {}", a, b, result);
    EXPECT_DOUBLE_EQ(result, 5.0);
    
    logger.info("Division test passed");
}

TEST_F(MathTest, DivisionByZero) {
    logger.warning("Testing division by zero (expected to fail safely)");
    
    double a = 10.0;
    double b = 0.0;
    
    // This should produce infinity
    double result = a / b;
    
    logger.debug("Result of {} / {}: {}", a, b, result);
    EXPECT_TRUE(std::isinf(result));
    
    logger.info("Division by zero test passed");
}

// Floating point tests
TEST_F(MathTest, SquareRoot) {
    logger.info("Testing square root");
    
    double value = 16.0;
    double result = std::sqrt(value);
    
    logger.debug("sqrt({}) = {}", value, result);
    EXPECT_DOUBLE_EQ(result, 4.0);
    
    logger.info("Square root test passed");
}

TEST_F(MathTest, Power) {
    logger.info("Testing power function");
    
    double base = 2.0;
    double exponent = 10.0;
    double result = std::pow(base, exponent);
    
    logger.debug("{}^{} = {}", base, exponent, result);
    EXPECT_DOUBLE_EQ(result, 1024.0);
    
    logger.info("Power test passed");
}

// Edge cases
TEST_F(MathTest, LargeNumbers) {
    logger.info("Testing with large numbers");
    
    long long a = 1000000000LL;
    long long b = 999999999LL;
    long long result = a + b;
    
    logger.debug("Large number addition: {} + {} = {}", a, b, result);
    EXPECT_EQ(result, 1999999999LL);
    
    logger.info("Large numbers test passed");
}

TEST_F(MathTest, NegativeNumbers) {
    logger.info("Testing with negative numbers");
    
    int a = -5;
    int b = -3;
    int result = a * b;
    
    logger.debug("Negative multiplication: {} * {} = {}", a, b, result);
    EXPECT_EQ(result, 15);
    
    logger.info("Negative numbers test passed");
}

// Performance test with timing
TEST_F(MathTest, PerformanceTest) {
    logger.info("Running performance test");
    
    auto start = std::chrono::high_resolution_clock::now();
    
    double sum = 0.0;
    for (int i = 0; i < 1000000; ++i) {
        sum += std::sqrt(i);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    logger.info("Performance test completed in {} ms", duration.count());
    logger.debug("Sum result: {}", sum);
    
    EXPECT_GT(sum, 0.0);
}

// Main function
int main(int argc, char **argv) {
    // Initialize Google Test
    ::testing::InitGoogleTest(&argc, argv);
    
    // Initialize logging
    TestLogger logger("main");
    logger.info("Starting Math Tests");
    logger.info("Google Test version: {}.{}.{}", 
                GTEST_VERSION_MAJOR, 
                GTEST_VERSION_MINOR,
                GTEST_VERSION_PATCH);
    
    // Run all tests
    int result = RUN_ALL_TESTS();
    
    if (result == 0) {
        logger.info("All tests passed!");
    } else {
        logger.error("Some tests failed!");
    }
    
    return result;
}

