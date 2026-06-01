// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/ck.hpp"
#include "ck/host_utility/device_prop.hpp"

#include "s_prefetch_op_util.hpp"

template <typename T, uint32_t NUM_THREADS, uint32_t NUM_SCALARS>
bool run_test(bool time_kernels)
{
    bool pass = true;

    const auto s_prefetch_kernel =
        ck::s_prefetch_op_util::kernel_with_prefetch<T,
                                                     NUM_THREADS,
                                                     NUM_SCALARS,
                                                     ck::s_prefetch_op_util::SPrefetchDataOp<T>>;
    const auto s_buffer_prefetch_kernel = ck::s_prefetch_op_util::kernel_with_prefetch<
        T,
        NUM_THREADS,
        NUM_SCALARS,
        ck::s_prefetch_op_util::SBufferPrefetchDataOp<T, NUM_SCALARS>>;

    const auto prefetch_kernel_container =
        std::make_tuple(s_prefetch_kernel, s_buffer_prefetch_kernel);

    ck::static_for<0, 2, 1>{}([&](auto i) {
        std::string kernel_name = (i == 1 ? "s_buffer_prefetch" : "s_prefetch");

        auto kernel = std::get<ck::Number<i>{}>(prefetch_kernel_container);

        pass &= ck::s_prefetch_op_util::
            test_prefetch_impl<decltype(kernel), T, NUM_THREADS, NUM_SCALARS>(
                time_kernels, kernel, kernel_name);
    });

    return pass;
}

int main(int argc, char* argv[])
{
    if(!ck::is_gfx12_supported())
    {
        std::cout << "This feature is not supported by current HW, skipping tests." << std::endl;
        return 0;
    }

    bool time_kernels = false;

    if(argc == 2)
    {
        time_kernels = std::stoi(argv[1]);
    }

    bool pass = true;

    std::cout << "=== Testing Constant Cache Prefetch ===" << std::endl;

    // Test different data types
    pass &= run_test<float, 4096, 1024>(time_kernels);
    pass &= run_test<double, 4096, 512>(time_kernels);

    std::cout << "TestConstantPrefetch ..... " << (pass ? "SUCCESS" : "FAILURE") << std::endl;
    return pass ? 0 : 1;
}
