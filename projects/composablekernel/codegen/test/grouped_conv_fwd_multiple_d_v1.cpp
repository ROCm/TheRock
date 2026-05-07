// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/host/device_grouped_conv_fwd_multiple_d/conv_fwd_op.hpp"
#include "ck/host/device_grouped_conv_fwd_multiple_d/conv_fwd_problem.hpp"
#include "ck/host/headers.hpp"
#include "ck/host/stringutils.hpp"
#include "ck/host/utils.hpp"
#include "ck/tensor_operation/gpu/device/helper.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_conv_fwd.hpp"
#include <test.hpp>
#include <rtc/compile_kernel.hpp>
#include <rtc/hip.hpp>
#include "common.hpp"
#include <fstream>

// Need this for verification
/**struct Epilogue
{
    Epilogue(float alpha, float beta) : alpha_(alpha), beta_(beta){};

    template <typename E, typename D>
    __host__ __device__ constexpr void operator()(E& e, const D& d) const;

    template <>
    __host__ __device__ constexpr void operator()<ck::half_t, ck::half_t>(ck::half_t& e,
                                                                          const ck::half_t& d) const
    {
        e = ck::type_convert<ck::half_t>(alpha_ * e + beta_ * ck::type_convert<float>(d));
    }

    float alpha_;
    float beta_;
};**/
const std::string conv_compile_check = R"__ck__(
#include <${include}>

${template};

)__ck__";

TEST_CASE(test_problem_kernel)
{
    // set up problem specification
    ck::host::conv::Problem_Conv_Fwd prob;
    prob.NumDim = 2;
    prob.G      = 32;
    prob.N      = 256;
    prob.C      = 32;
    prob.K      = 64;
    prob.Y      = 3;
    prob.X      = 3;
    prob.Hi     = 28;
    prob.Wi     = 28;
    prob.Ho     = 28;
    prob.Wo     = 28;
    check_all<ck::half_t> check;

    // user provided fusion operations
    std::string epilogue = R"(
struct Epilogue
{
    __host__ __device__ Epilogue(float alpha, float beta) : alpha_(alpha), beta_(beta){};

    template <typename E, typename D>
    __host__ __device__ constexpr void operator()(E& e, const D& d) const;

    template <>
    __host__ __device__ constexpr void operator()<ck::half_t, ck::half_t>(ck::half_t& e,
                                                                          const ck::half_t& d) const
    {
        e = ck::type_convert<ck::half_t>(alpha_ * e + beta_ * ck::type_convert<float>(d));
    }

    float alpha_;
    float beta_;
};
)";
    std::string prologue = "";

    // length+stride arrays
    ck::Array<ck::index_t, 5> in_lengths{static_cast<int>(prob.G),
                                         static_cast<int>(prob.N),
                                         static_cast<int>(prob.C),
                                         static_cast<int>(prob.Hi),
                                         static_cast<int>(prob.Wi)};
    ck::Array<ck::index_t, 5> out_lengths{static_cast<int>(prob.G),
                                          static_cast<int>(prob.N),
                                          static_cast<int>(prob.K),
                                          static_cast<int>(prob.Ho),
                                          static_cast<int>(prob.Wo)};
    ck::Array<ck::index_t, 5> wei_lengths{static_cast<int>(prob.G),
                                          static_cast<int>(prob.K),
                                          static_cast<int>(prob.C),
                                          static_cast<int>(prob.Y),
                                          static_cast<int>(prob.X)};

    ck::Array<ck::index_t, 5> in_strides{static_cast<int>(prob.C),
                                         static_cast<int>(prob.Hi * prob.Wi * prob.G * prob.C),
                                         1,
                                         static_cast<int>(prob.Wi * prob.G * prob.C),
                                         static_cast<int>(prob.G * prob.C)};
    ck::Array<ck::index_t, 5> out_strides{static_cast<int>(prob.K),
                                          static_cast<int>(prob.Ho * prob.Wo * prob.G * prob.K),
                                          1,
                                          static_cast<int>(prob.Wo * prob.G * prob.K),
                                          static_cast<int>(prob.G * prob.K)};
    ck::Array<ck::index_t, 5> wei_strides{static_cast<int>(prob.K * prob.Y * prob.X * prob.C),
                                          static_cast<int>(prob.Y * prob.X * prob.C),
                                          1,
                                          static_cast<int>(prob.X * prob.C),
                                          static_cast<int>(prob.C)};

    ck::Array<ck::index_t, 2> conv_filter_strides   = {2, 2};
    ck::Array<ck::index_t, 2> conv_filter_dilations = {1, 1};
    ck::Array<ck::index_t, 2> input_left_pads       = {1, 1};
    ck::Array<ck::index_t, 2> input_right_pads      = {1, 1};

    // move the data onto the device
    auto in_dev =
        to_gpu(generate_buffer<ck::half_t, ck::Array<ck::index_t, 5>>(in_lengths, in_strides, 0));
    auto wei_dev =
        to_gpu(generate_buffer<ck::half_t, ck::Array<ck::index_t, 5>>(wei_lengths, wei_strides, 1));
    auto out_dev =
        to_gpu(generate_buffer<ck::half_t, ck::Array<ck::index_t, 5>>(out_lengths, out_strides, 2));

    // CK Verficiation: Reference Kernel
    /**bool pass = true;
    Tensor<ck::half_t> in_host(in_lengths, in_strides);
    in_host.GenerateTensorValue(GeneratorTensor_1<ck::half_t>{1});
    Tensor<ck::half_t> wei_host(wei_lengths, wei_strides);
    wei_host.GenerateTensorValue(GeneratorTensor_1<ck::half_t>{1});
    Tensor<ck::half_t> out_host(out_lengths, out_strides);

    std::vector<ck::index_t> conv_filter_strides_   = {2, 2};
    std::vector<ck::index_t> conv_filter_dilations_ = {1, 1};
    std::vector<ck::index_t> input_left_pads_       = {1, 1};
    std::vector<ck::index_t> input_right_pads_      = {1, 1};

    auto ref_conv = ck::tensor_operation::host::ReferenceConvFwd<
        2,
        ck::half_t,
        ck::half_t,
        ck::half_t,
        ck::tensor_operation::element_wise::PassThrough,
        ck::tensor_operation::element_wise::PassThrough,
        Epilogue>();

    auto ref_invoker  = ref_conv.MakeInvoker();
    auto ref_argument = ref_conv.MakeArgument(in_host,
                                              wei_host,
                                              out_host,
                                              conv_filter_strides_,
                                              conv_filter_dilations_,
                                              input_left_pads_,
                                              input_right_pads_,
                                              ck::tensor_operation::element_wise::PassThrough{},
                                              ck::tensor_operation::element_wise::PassThrough{},
                                              Epilogue{1.0f, 1.0f});
    out_host.SetZero();
    ref_invoker.Run(ref_argument);**/
    int i = 0;
    for(auto solution : prob.GetSolutions("gfx908", prologue, epilogue))
    {
        std::cout << "Testing solution " << std::to_string(++i) << std::endl;
        // substitute instance values into the template
        auto src = ck::host::InterpolateString(
            conv_compile_check,
            {{"include", prob.GetIncludeHeader()}, {"template", solution.ToTemplateString()}});

        auto srcs = get_headers_for_test();
        srcs.push_back({"main.cpp", src});
        rtc::compile_options options;
        auto name           = solution.GetTemplateParameter<std::string>("name");
        options.kernel_name = "run_" + name;
        auto k              = rtc::compile_kernel(srcs, options);

        // Grid size calculation
        auto block_size = solution.GetTemplateParameter<ck::index_t>("BlockSize");

        auto tmp = get_launch_params(solution, out_lengths, out_strides);

        auto grid_size = tmp * in_lengths[1];

        // launch the kernel with arguments needed for the argument pointer
        k.launch(nullptr, grid_size * block_size, block_size)(in_dev.data(),
                                                              wei_dev.data(),
                                                              out_dev.data(),
                                                              in_lengths,
                                                              in_strides,
                                                              wei_lengths,
                                                              wei_strides,
                                                              out_lengths,
                                                              out_strides,
                                                              conv_filter_strides,
                                                              conv_filter_dilations,
                                                              input_left_pads,
                                                              input_right_pads);

        // auto res = rtc::from_gpu(out_dev);
        // pass &= ck::utils::check_err(res, out_host, "Error: incorrect results!", 1e-5f, 1e-4f);
        // assert(pass);

        // Simple check: this checks that the output from each instance matches the output from the
        // first instance
        CHECK(report(solution, check(rtc::from_gpu(out_dev))));
    }
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
