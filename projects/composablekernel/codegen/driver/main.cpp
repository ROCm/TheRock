// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <functional>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include "ck/host/device_gemm_multiple_d/operation.hpp"
#include "ck/host/device_grouped_conv_fwd_multiple_d/conv_fwd_op.hpp"
#include "ck/host/stringutils.hpp"

using ck::host::Transform;

struct Emitters
{
    // retrieve the hard-coded instances provided, template them, and then store them in a map
    std::unordered_map<std::string, std::function<std::vector<std::string>()>> m;

    template <class T>
    void Register(const std::string& name, const std::string& prologue, const std::string& epilogue)
    {
        m[name] = [&] {
            auto configs = T::CreateOperations(prologue, epilogue);

            return Transform(configs, [](const auto& ops) { return ToTuple(ops); });
        };
    }

    // takes in an operation instance and uses it to substitute the correct values into the template
    template <class T>
    static std::string ToTuple(const T& ops)
    {
        auto templates = Transform(
            ops, [](const auto& op) { return "    " + op.ToSolution().ToTemplateString(); });
        return "std::tuple<\n" + ck::host::JoinStrings(templates, ",\n") + ">";
    }

    // Join together all the strings in the map
    std::string Emit(const std::string& name) { return ck::host::JoinStrings(m.at(name)(), "\n"); }

    std::vector<std::string> List() const
    {
        return Transform(m, [](auto&& p) { return p.first; });
    }
};

int main(int argc, const char* argv[])
{
    std::string prog = argv[0];
    std::vector<std::string> args(argv + 1, argv + argc);

    // Specify problem type and problem size
    ck::host::device_gemm_multiple_d::Problem prob;
    prob.M = 1024;
    prob.N = 1024;
    prob.K = 1024;

    // user provided fusion
    std::string prologue = "";
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
};)";

    // Load in operations into the Register
    Emitters e;
    e.Register<ck::host::device_gemm_multiple_d::Operation_Xdl_CShuffle>(
        "DeviceGemmMultipleD_Xdl_CShuffle", prologue, epilogue);

    if(args.empty() or std::any_of(args.begin(), args.end(), [](auto arg) {
           return arg == "-h" or arg == "--help";
       }))
    {
        std::cout << "USAGE:" << std::endl;
        std::cout << "    " << prog << " [TEMPLATE]" << std::endl;
        std::cout << std::endl;
        std::cout << "FLAGS:" << std::endl;
        std::cout << "    -h, --help                     Show help" << std::endl;
        std::cout << std::endl;
        std::cout << "TEMPLATES:" << std::endl;
        for(auto x : e.List())
            std::cout << "    " << x << std::endl;
        std::cout << std::endl;
        return 0;
    }

    // print out all the instances for the operation that was chosen at the command line
    for(auto name : args)
        std::cout << e.Emit(name) << std::endl;

    return 0;
}
