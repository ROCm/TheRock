# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

CC=g++

$CC -Wall -std=c++20 -Iinclude -O3 block_swizzle_test.cpp -o block_swizzle_test.exe