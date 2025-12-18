// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/**
 * Sample Integration Tests with Logging
 * ======================================
 * Demonstrates more complex integration scenarios
 */

#include <gtest/gtest.h>
#include "test_logging.hpp"
#include <vector>
#include <map>
#include <memory>
#include <thread>
#include <chrono>

// Simple class for integration testing
class DataProcessor {
public:
    DataProcessor(const std::string& name) : name_(name), processed_count_(0) {}
    
    void process(const std::vector<int>& data) {
        for (int value : data) {
            results_.push_back(value * 2);
            processed_count_++;
        }
    }
    
    const std::vector<int>& getResults() const { return results_; }
    int getProcessedCount() const { return processed_count_; }
    std::string getName() const { return name_; }

private:
    std::string name_;
    std::vector<int> results_;
    int processed_count_;
};

class IntegrationTest : public ::testing::Test {
protected:
    TestLogger logger;
    std::unique_ptr<DataProcessor> processor;
    
    void SetUp() override {
        logger = TestLogger("IntegrationTest");
        logger.info("Setting up IntegrationTest");
        processor = std::make_unique<DataProcessor>("TestProcessor");
    }
    
    void TearDown() override {
        logger.info("Tearing down IntegrationTest");
        processor.reset();
    }
};

TEST_F(IntegrationTest, DataProcessing) {
    logger.info("Testing data processing");
    
    std::vector<int> input = {1, 2, 3, 4, 5};
    logger.debug("Input data size: {}", input.size());
    
    processor->process(input);
    
    const auto& results = processor->getResults();
    logger.debug("Processed {} items", results.size());
    
    EXPECT_EQ(results.size(), 5);
    EXPECT_EQ(results[0], 2);
    EXPECT_EQ(results[4], 10);
    
    logger.info("Data processing test passed");
}

TEST_F(IntegrationTest, MultipleProcessingRounds) {
    logger.info("Testing multiple processing rounds");
    
    std::vector<int> round1 = {1, 2, 3};
    std::vector<int> round2 = {4, 5, 6};
    std::vector<int> round3 = {7, 8, 9};
    
    processor->process(round1);
    logger.debug("Round 1 completed: {} items processed", processor->getProcessedCount());
    
    processor->process(round2);
    logger.debug("Round 2 completed: {} items processed", processor->getProcessedCount());
    
    processor->process(round3);
    logger.debug("Round 3 completed: {} items processed", processor->getProcessedCount());
    
    EXPECT_EQ(processor->getProcessedCount(), 9);
    EXPECT_EQ(processor->getResults().size(), 9);
    
    logger.info("Multiple processing rounds test passed");
}

TEST_F(IntegrationTest, EmptyDataHandling) {
    logger.info("Testing empty data handling");
    
    std::vector<int> empty_data;
    processor->process(empty_data);
    
    logger.debug("Processed count: {}", processor->getProcessedCount());
    EXPECT_EQ(processor->getProcessedCount(), 0);
    EXPECT_TRUE(processor->getResults().empty());
    
    logger.info("Empty data handling test passed");
}

TEST_F(IntegrationTest, LargeDataSet) {
    logger.info("Testing large data set processing");
    
    std::vector<int> large_data;
    for (int i = 0; i < 10000; ++i) {
        large_data.push_back(i);
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    processor->process(large_data);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    logger.info("Large data set processed in {} ms", duration.count());
    
    EXPECT_EQ(processor->getProcessedCount(), 10000);
    EXPECT_EQ(processor->getResults()[9999], 19998);
    
    logger.info("Large data set test passed");
}

TEST_F(IntegrationTest, MapOperations) {
    logger.info("Testing map operations");
    
    std::map<std::string, int> data_map;
    data_map["rocm"] = 6;
    data_map["hip"] = 5;
    data_map["amd"] = 3;
    
    logger.debug("Map size: {}", data_map.size());
    
    EXPECT_EQ(data_map.size(), 3);
    EXPECT_EQ(data_map["rocm"], 6);
    EXPECT_TRUE(data_map.find("hip") != data_map.end());
    
    logger.info("Map operations test passed");
}

TEST_F(IntegrationTest, MemoryManagement) {
    logger.info("Testing memory management");
    
    std::vector<std::unique_ptr<DataProcessor>> processors;
    
    for (int i = 0; i < 100; ++i) {
        processors.push_back(std::make_unique<DataProcessor>("Processor_" + std::to_string(i)));
    }
    
    logger.debug("Created {} processors", processors.size());
    EXPECT_EQ(processors.size(), 100);
    
    // Process some data
    std::vector<int> test_data = {1, 2, 3};
    for (auto& proc : processors) {
        proc->process(test_data);
    }
    
    logger.debug("All processors completed work");
    
    // Cleanup happens automatically
    processors.clear();
    logger.debug("All processors cleaned up");
    
    logger.info("Memory management test passed");
}

TEST_F(IntegrationTest, ExceptionHandling) {
    logger.info("Testing exception handling");
    
    try {
        std::vector<int> data = {1, 2, 3};
        // Access out of bounds
        int invalid = data.at(100);
        FAIL() << "Expected exception was not thrown";
    } catch (const std::out_of_range& e) {
        logger.warning("Caught expected exception: {}", e.what());
        SUCCEED();
    }
    
    logger.info("Exception handling test passed");
}

TEST_F(IntegrationTest, ConcurrencySimulation) {
    logger.info("Testing concurrency simulation");
    
    const int num_threads = 4;
    std::vector<std::thread> threads;
    std::vector<int> thread_results(num_threads, 0);
    
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([i, &thread_results, this]() {
            logger.debug("Thread {} starting", i);
            
            int sum = 0;
            for (int j = 0; j < 1000; ++j) {
                sum += j;
            }
            thread_results[i] = sum;
            
            logger.debug("Thread {} completed with sum: {}", i, sum);
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    logger.debug("All threads completed");
    
    for (int i = 0; i < num_threads; ++i) {
        EXPECT_EQ(thread_results[i], 499500);
    }
    
    logger.info("Concurrency simulation test passed");
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    
    TestLogger logger("main");
    logger.info("Starting Integration Tests");
    logger.info("Running comprehensive integration test suite");
    
    int result = RUN_ALL_TESTS();
    
    if (result == 0) {
        logger.info("All integration tests passed!");
    } else {
        logger.error("Some integration tests failed!");
    }
    
    return result;
}

