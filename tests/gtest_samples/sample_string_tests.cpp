// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/**
 * Sample String Tests with Logging
 * =================================
 * Demonstrates string operations testing with logging
 */

#include <gtest/gtest.h>
#include "test_logging.hpp"
#include <string>
#include <algorithm>
#include <sstream>

class StringTest : public ::testing::Test {
protected:
    TestLogger logger;
    
    void SetUp() override {
        logger = TestLogger("StringTest");
        logger.info("Setting up StringTest");
    }
    
    void TearDown() override {
        logger.info("Tearing down StringTest");
    }
};

TEST_F(StringTest, BasicConcatenation) {
    logger.info("Testing string concatenation");
    
    std::string str1 = "Hello";
    std::string str2 = "World";
    std::string result = str1 + " " + str2;
    
    logger.debug("Concatenating: '{}' + ' ' + '{}' = '{}'", str1, str2, result);
    EXPECT_EQ(result, "Hello World");
    
    logger.info("Concatenation test passed");
}

TEST_F(StringTest, StringLength) {
    logger.info("Testing string length");
    
    std::string text = "TheRock";
    size_t length = text.length();
    
    logger.debug("Length of '{}': {}", text, length);
    EXPECT_EQ(length, 7);
    
    logger.info("String length test passed");
}

TEST_F(StringTest, StringComparison) {
    logger.info("Testing string comparison");
    
    std::string str1 = "AMD";
    std::string str2 = "AMD";
    std::string str3 = "ROCm";
    
    logger.debug("Comparing: '{}' == '{}'", str1, str2);
    EXPECT_EQ(str1, str2);
    
    logger.debug("Comparing: '{}' != '{}'", str1, str3);
    EXPECT_NE(str1, str3);
    
    logger.info("String comparison test passed");
}

TEST_F(StringTest, SubstringExtraction) {
    logger.info("Testing substring extraction");
    
    std::string text = "TheRock Project";
    std::string sub = text.substr(0, 7);
    
    logger.debug("Substring of '{}' from 0 to 7: '{}'", text, sub);
    EXPECT_EQ(sub, "TheRock");
    
    logger.info("Substring test passed");
}

TEST_F(StringTest, StringSearch) {
    logger.info("Testing string search");
    
    std::string text = "AMD ROCm Platform";
    size_t pos = text.find("ROCm");
    
    logger.debug("Position of 'ROCm' in '{}': {}", text, pos);
    EXPECT_EQ(pos, 4);
    
    logger.info("String search test passed");
}

TEST_F(StringTest, CaseConversion) {
    logger.info("Testing case conversion");
    
    std::string text = "HeLLo WoRLd";
    std::string lower = text;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    
    logger.debug("Lowercase conversion: '{}' -> '{}'", text, lower);
    EXPECT_EQ(lower, "hello world");
    
    std::string upper = text;
    std::transform(upper.begin(), upper.end(), upper.begin(), ::toupper);
    
    logger.debug("Uppercase conversion: '{}' -> '{}'", text, upper);
    EXPECT_EQ(upper, "HELLO WORLD");
    
    logger.info("Case conversion test passed");
}

TEST_F(StringTest, StringReplace) {
    logger.info("Testing string replace");
    
    std::string text = "Hello World";
    size_t pos = text.find("World");
    
    if (pos != std::string::npos) {
        text.replace(pos, 5, "AMD");
    }
    
    logger.debug("After replacement: '{}'", text);
    EXPECT_EQ(text, "Hello AMD");
    
    logger.info("String replace test passed");
}

TEST_F(StringTest, EmptyString) {
    logger.info("Testing empty string");
    
    std::string empty;
    
    logger.debug("Empty string length: {}", empty.length());
    EXPECT_TRUE(empty.empty());
    EXPECT_EQ(empty.length(), 0);
    
    logger.info("Empty string test passed");
}

TEST_F(StringTest, StringSplit) {
    logger.info("Testing string split");
    
    std::string text = "one,two,three,four";
    std::stringstream ss(text);
    std::string token;
    std::vector<std::string> tokens;
    
    while (std::getline(ss, token, ',')) {
        tokens.push_back(token);
    }
    
    logger.debug("Split '{}' into {} parts", text, tokens.size());
    EXPECT_EQ(tokens.size(), 4);
    EXPECT_EQ(tokens[0], "one");
    EXPECT_EQ(tokens[3], "four");
    
    logger.info("String split test passed");
}

TEST_F(StringTest, StringReverse) {
    logger.info("Testing string reverse");
    
    std::string text = "TheRock";
    std::string reversed = text;
    std::reverse(reversed.begin(), reversed.end());
    
    logger.debug("Reversed '{}': '{}'", text, reversed);
    EXPECT_EQ(reversed, "kcoRehT");
    
    logger.info("String reverse test passed");
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    
    TestLogger logger("main");
    logger.info("Starting String Tests");
    
    int result = RUN_ALL_TESTS();
    
    if (result == 0) {
        logger.info("All string tests passed!");
    } else {
        logger.error("Some string tests failed!");
    }
    
    return result;
}


