// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <fstream>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "ck/library/utility/convolution_parameter.hpp"

namespace ck {
namespace test {

namespace fs = std::filesystem;

// Helper function to find test_data directory relative to the test binary
static std::string GetTestDataPath()
{
    // Get the path to the current executable
    fs::path exe_path = fs::read_symlink("/proc/self/exe");

    // Get the directory containing the executable
    fs::path current_dir = exe_path.parent_path();

    // Search for test_data directory by going up the directory tree
    // This makes the code robust regardless of build directory depth
    while(current_dir != current_dir.root_path())
    {
        fs::path test_data_path = current_dir / "test_data";
        if(fs::exists(test_data_path) && fs::is_directory(test_data_path))
        {
            return test_data_path.string();
        }
        current_dir = current_dir.parent_path();
    }

    // If not found, return empty string
    std::cerr << "ERROR: Could not find test_data directory relative to executable" << std::endl;
    return "";
}

// CSV Reader Function for Loading Test Cases
// Reads convolution parameters from CSV file and returns vector of ConvParam structures
inline std::vector<ck::utils::conv::ConvParam> load_csv_test_cases(const std::string& filename)
{
    std::vector<ck::utils::conv::ConvParam> conv_params; // Return vector
    std::ifstream file(filename);                        // Open CSV file

    if(!file.is_open())
    {
        std::cerr << "ERROR: Cannot open CSV file: " << filename << std::endl;
        return conv_params; // Return empty vector on error
    }

    std::string line;
    int line_number = 0;

    // Read file line by line
    while(std::getline(file, line))
    {
        line_number++;
        // Skip comment lines (starting with #) and empty lines
        if(line.empty() || line[0] == '#')
        {
            continue;
        }

        // Skip header line (contains column names)
        if(line.find("NDim,Groups,BatchSize") != std::string::npos)
        {
            continue;
        }

        // Parse CSV line using stringstream
        std::stringstream ss(line);
        std::string cell;
        std::vector<std::string> row;

        // Split line by commas
        while(std::getline(ss, cell, ','))
        {
            row.push_back(cell);
        }

        // Validate row has correct number of columns
        if(row.size() < 19)
        { // Need at least 19 columns for 2D (excluding TestName)
            std::cerr << "WARNING: Line " << line_number << " has insufficient columns ("
                      << row.size() << "), skipping" << std::endl;
            continue;
        }

        try
        {
            // Parse CSV data into ConvParam structure
            // CSV Format:
            // NDim,Groups,BatchSize,OutChannels,InChannels,KernelH,KernelW,InputH,InputW,OutputH,OutputW,StrideH,StrideW,DilationH,DilationW,LeftPadH,LeftPadW,RightPadH,RightPadW,TestName
            int NDim        = std::stoi(row[0]);
            int Groups      = std::stoi(row[1]);
            int BatchSize   = std::stoi(row[2]);
            int OutChannels = std::stoi(row[3]);
            int InChannels  = std::stoi(row[4]);

            if(NDim == 1)
            {
                // 1D Convolution: Need fewer columns for 1D parameters
                if(row.size() < 13)
                {
                    std::cerr << "WARNING: 1D convolution on line " << line_number
                              << " needs 13+ columns, has " << row.size() << ", skipping"
                              << std::endl;
                    continue;
                }
                // 1D Convolution: {NDim, Groups, BatchSize, OutChannels, InChannels,
                // {KernelW}, {InputW}, {StrideW}, {DilationW}, {LeftPadW}, {RightPadW}}
                ck::utils::conv::ConvParam param = {
                    NDim,                 // NDim = 1
                    Groups,               // Groups
                    BatchSize,            // Batch size
                    OutChannels,          // Output channels
                    InChannels,           // Input channels
                    {std::stoi(row[5])},  // Kernel: {W}
                    {std::stoi(row[7])},  // Input: {W}
                    {std::stoi(row[11])}, // Stride: {W}
                    {std::stoi(row[13])}, // Dilation: {W}
                    {std::stoi(row[15])}, // Left pad: {W}
                    {std::stoi(row[17])}  // Right pad: {W}
                };
                conv_params.push_back(param);
            }
            else if(NDim == 2)
            {
                // 2D Convolution: {NDim, Groups, BatchSize, OutChannels, InChannels,
                // {KernelH,KernelW}, {InputH,InputW}, {StrideH,StrideW}, {DilationH,DilationW},
                // {LeftPadH,LeftPadW}, {RightPadH,RightPadW}}
                ck::utils::conv::ConvParam param = {
                    NDim,                                     // NDim = 2
                    Groups,                                   // Groups
                    BatchSize,                                // Batch size
                    OutChannels,                              // Output channels
                    InChannels,                               // Input channels
                    {std::stoi(row[5]), std::stoi(row[6])},   // Kernel: {H, W}
                    {std::stoi(row[7]), std::stoi(row[8])},   // Input: {H, W}
                    {std::stoi(row[11]), std::stoi(row[12])}, // Stride: {H, W}
                    {std::stoi(row[13]), std::stoi(row[14])}, // Dilation: {H, W}
                    {std::stoi(row[15]), std::stoi(row[16])}, // Left pad: {H, W}
                    {std::stoi(row[17]), std::stoi(row[18])}  // Right pad: {H, W}
                };
                conv_params.push_back(param);
            }
            else if(NDim == 3)
            {
                // 3D Convolution: Need more columns for 3D parameters
                if(row.size() < 26)
                {
                    std::cerr << "WARNING: 3D convolution on line " << line_number
                              << " needs 26+ columns, has " << row.size() << ", skipping"
                              << std::endl;
                    continue;
                }
                // 3D Convolution: {NDim, Groups, BatchSize, OutChannels, InChannels,
                // {KernelD,KernelH,KernelW}, {InputD,InputH,InputW}, {OutputD,OutputH,OutputW},
                // {StrideD,StrideH,StrideW}, {DilationD,DilationH,DilationW},
                // {LeftPadD,LeftPadH,LeftPadW}, {RightPadD,RightPadH,RightPadW}}
                ck::utils::conv::ConvParam param = {
                    NDim,                                                       // NDim = 3
                    Groups,                                                     // Groups
                    BatchSize,                                                  // Batch size
                    OutChannels,                                                // Output channels
                    InChannels,                                                 // Input channels
                    {std::stoi(row[5]), std::stoi(row[6]), std::stoi(row[7])},  // Kernel: {D, H, W}
                    {std::stoi(row[8]), std::stoi(row[9]), std::stoi(row[10])}, // Input: {D, H, W}
                    {std::stoi(row[14]),
                     std::stoi(row[15]),
                     std::stoi(row[16])}, // Stride: {D, H, W}
                    {std::stoi(row[17]),
                     std::stoi(row[18]),
                     std::stoi(row[19])}, // Dilation: {D, H, W}
                    {std::stoi(row[20]),
                     std::stoi(row[21]),
                     std::stoi(row[22])}, // Left pad: {D, H, W}
                    {std::stoi(row[23]),
                     std::stoi(row[24]),
                     std::stoi(row[25])} // Right pad: {D, H, W}
                };
                conv_params.push_back(param);
            }
            else
            {
                std::cerr << "WARNING: Unsupported NDim=" << NDim << " on line " << line_number
                          << ", skipping" << std::endl;
            }
        }
        catch(const std::exception& e)
        {
            std::cerr << "ERROR: Failed to parse line " << line_number << ": " << e.what()
                      << std::endl;
            continue;
        }
    }

    file.close();
    std::cout << "Loaded " << conv_params.size() << " test cases from " << filename << std::endl;
    return conv_params;
}

// Helper function to load CSV test cases and populate conv_params vector
// Returns true if loading succeeded, false otherwise
inline bool load_and_populate_test_cases(const std::vector<std::string>& csv_paths,
                                         std::vector<ck::utils::conv::ConvParam>& conv_params,
                                         const std::string& dimension_label)
{
    for(const auto& csv_path : csv_paths)
    {
        auto csv_cases = load_csv_test_cases(csv_path);
        if(!csv_cases.empty())
        {
            // Successfully loaded CSV data - add all test cases to conv_params
            for(const auto& test_case : csv_cases)
            {
                conv_params.push_back(test_case);
            }
            std::cout << "Loaded " << csv_cases.size() << " " << dimension_label
                      << " test cases from " << csv_path << std::endl;
            return true;
        }
    }

    // Failed to load from any path
    std::cerr << "ERROR: Failed to load CSV test data from any of these locations:" << std::endl;
    for(const auto& path : csv_paths)
    {
        std::cerr << "  - " << path << std::endl;
    }
    std::cerr << "\nPlease ensure CSV test data exists in one of these locations." << std::endl;
    std::cerr << "Run generate_test_dataset.sh in test_data/ to create test datasets." << std::endl;

    return false;
}

} // namespace test
} // namespace ck
