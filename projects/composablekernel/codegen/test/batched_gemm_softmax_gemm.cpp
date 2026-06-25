// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/host/device_batched_gemm_softmax_gemm/problem.hpp"
#include "ck/host/stringutils.hpp"
#include "ck/host/utils.hpp"
#include "common.hpp"
#include <rtc/compile_kernel.hpp>
#include <rtc/hip.hpp>
#include <test.hpp>
#include <cmath>

using half = _Float16;

const std::string gemm_compile_check = R"__ck__(
#include <${include}>

extern "C" __global__ void f(const ck::half_t* a, const ck::half_t* b, const ck::half_t* b1, ck::half_t* c) {
    using G = ${template};
    constexpr auto desc = G::make_descriptor(ck::make_naive_tensor_descriptor(ck::make_tuple(${m}, ${k}), ck::make_tuple(${m}, 1)),
                                             ck::make_naive_tensor_descriptor(ck::make_tuple(${n}, ${k}), ck::make_tuple(${n}, 1)),
                                             ck::make_naive_tensor_descriptor(ck::make_tuple(${n}, ${o}), ck::make_tuple(1, ${n})),
                                             ck::make_naive_tensor_descriptor(ck::make_tuple(${m}, ${o}), ck::make_tuple(${m}, 1)));

    static_assert(desc.IsValid(), "Invalid ck gemm.");

    if constexpr(desc.IsValid())
    {
        ${template}::Run(desc,
               1.0,
               a,
               b,
               b1,
               c);
    }
}

)__ck__";

TEST_CASE(test_problem_kernel)
{
    ck::host::device_batched_gemm_softmax_gemm::Problem prob;
    prob.M      = 1024;
    prob.N      = 1024;
    prob.K      = 1024;
    prob.O      = 1024;
    prob.TransB = true;
    check_all<half> check;
    auto a  = to_gpu(generate_buffer<half>(1024 * 1024, 0));
    auto b  = to_gpu(generate_buffer<half>(1024 * 1024, 1));
    auto b1 = to_gpu(generate_buffer<half>(1024 * 1024, 2));
    auto c  = to_gpu(generate_buffer<half>(1024 * 1024, 3));

    auto solutions = prob.GetSolutions("gfx90a");
    std::cout << "Num solutions: " << solutions.size() << std::endl;
    for(auto i = 0; i < solutions.size(); ++i)
    {
        std::cout << "Testing solution " << std::to_string(i + 1) << std::endl;
        auto&& solution = solutions[i];
        auto src        = ck::host::InterpolateString(gemm_compile_check,
                                                      {{"include", prob.GetIncludeHeader()},
                                                       {"template", solution.ToTemplateString()},
                                                       {"m", std::to_string(prob.M)},
                                                       {"n", std::to_string(prob.N)},
                                                       {"k", std::to_string(prob.K)},
                                                       {"o", std::to_string(prob.O)}});
        auto srcs       = get_headers_for_test();
        srcs.push_back({"main.cpp", src});
        rtc::compile_options options;
        options.kernel_name = "f";
        auto k              = rtc::compile_kernel(srcs, options);
        auto block_size     = solution.GetTemplateParameter<std::size_t>("BlockSize");
        auto m_per_block    = solution.GetTemplateParameter<std::size_t>("Gemm01MPerBlock");
        auto n_per_block    = solution.GetTemplateParameter<std::size_t>("Gemm1NPerBlock");
        auto grid_size      = ck::host::integer_divide_ceil(prob.M, m_per_block) *
                         ck::host::integer_divide_ceil(prob.N, n_per_block);
        k.launch(nullptr, grid_size * block_size, block_size)(
            a.data(), b.data(), b1.data(), c.data());

        // NOTE: Solutions where MaskOutUpperTriangle is True don't produce consistent results
        CHECK(report(solution, check(rtc::from_gpu(c))));
    }
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
