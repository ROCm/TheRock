/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2024-2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#include "TensorDataManipulation.hpp"
#include <array>
#include <cstddef>
#include <iostream>

int main(int argc, char** argv)
{
    constexpr size_t m{18};
    constexpr size_t k{34};
    auto             weight = Tensor::Manipulation::Tensor::create<int>({m, k});

    for(size_t i = 0; i < m; ++i)
    {
        for(size_t j = 0; j < k; ++j)
        {
            weight.setValue<int>({i, j}, i * k + j);
        }
    }

    std::cout << "Original weight:\n";
    Tensor::Manipulation::printTensorDataMultiDims<int>(std::cout, weight);
    constexpr size_t            MiM       = 16;
    constexpr size_t            MiK       = 16;
    constexpr size_t            MiKv      = 4;
    constexpr size_t            PackK     = 2;
    constexpr auto              MultipleM = MiM;
    constexpr auto              MultipleK = MiK * PackK;
    const auto                  paddedM   = (m / MultipleM + !!(m % MultipleM)) * MultipleM;
    const auto                  paddedK   = (k / MultipleK + !!(k % MultipleK)) * MultipleK;
    Tensor::Manipulation::Shape paddedShape{paddedM, paddedK};
    auto                        paddedWeight = ::Tensor::Manipulation::pad(weight, paddedShape, 0);
    std::cout << "Padded weight:\n";
    Tensor::Manipulation::printTensorDataMultiDims<int>(std::cout, paddedWeight);
    paddedWeight.reshape({paddedM / MiM, MiM, paddedK / (MiK * PackK), MiK / MiKv, MiKv * PackK});
    Tensor::Manipulation::Tensor permuted = permute(paddedWeight, {0, 2, 3, 1, 4});
    std::cout << "Swizzle weight:\n";
    Tensor::Manipulation::printTensorDataMultiDims<int>(std::cout, permuted);
    return 0;
}
