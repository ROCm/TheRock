// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <getopt.h>

#include "ck/library/utility/host_common_util.hpp"
#include "profiler/profile_reduce_impl.hpp"
#include <gtest/gtest.h>
using namespace ck;

static ck::index_t param_mask     = 0xffff;
static ck::index_t instance_index = -1;

struct ReduceParam
{
    bool do_verification{true};
    bool propagateNan{false};
    bool useIndex{false};
    bool time_kernel{false};
    bool do_dumpout{false};
    int init_method{2};
    float alpha{1.0f};
    float beta{0.0f};
    std::vector<size_t> inLengths{64, 4, 280, 82};
    std::vector<int> reduceDims{0, 1, 2, 3};
};

std::vector<std::vector<int>> SetGenericReduceDim()
{
    return {{0, 1, 2, 3}, {0, 1, 2}, {0, 1, 3}, {0, 2, 3}, {1, 2, 3}, {0}, {1}, {2}, {3}};
}

template <typename T>
class ReduceWithIndexTest : public ::testing::Test
{
    protected:
    using InDataType  = std::tuple_element_t<0, T>;
    using AccDataType = std::tuple_element_t<1, T>;
    using OutDataType = std::tuple_element_t<2, T>;

    static std::vector<ReduceParam> params;

    static void SetUpTestSuite()
    {
        // set testcase variables
        ReduceParam set;
        const auto setReduceDim = SetGenericReduceDim();

        for(std::size_t i(0); i < setReduceDim.size(); ++i)
        {
            set.reduceDims = setReduceDim[i];
            params.emplace_back(set);
        }
    }

    template <ReduceTensorOp ReduceOpIdType>
    void Run()
    {
        for(size_t i = 0; i < this->params.size(); i++)
        {
            if((param_mask & (1 << i)) == 0)
            {
                continue;
            }
            auto& param  = this->params[i];
            bool success = ck::profiler::profile_reduce_impl<InDataType, AccDataType, OutDataType>(
                param.do_verification,
                param.init_method,
                param.do_dumpout,
                param.time_kernel,
                param.inLengths,
                param.reduceDims,
                ReduceOpIdType,
                param.propagateNan,
                param.useIndex,
                param.alpha,
                param.beta,
                instance_index);
            EXPECT_TRUE(success);
        }
    }
};

template <typename T>
std::vector<ReduceParam> ReduceWithIndexTest<T>::params = {};

using Reduce_float_types       = ::testing::Types<std::tuple<float, float, float>>;
using Reduce_double_types      = ::testing::Types<std::tuple<double, double, double>>;
using Reduce_int8t_types       = ::testing::Types<std::tuple<int8_t, int8_t, int8_t>>;
using Reduce_half_types        = ::testing::Types<std::tuple<ck::half_t, ck::half_t, ck::half_t>>;
using Reduce_bhalf_float_Types = ::testing::Types<std::tuple<ck::bhalf_t, float, ck::bhalf_t>>;

template <typename TType>
class ReduceWithNoIndexFloat : public ReduceWithIndexTest<TType>
{
};

template <typename TType>
class ReduceWithNoIndexDouble : public ReduceWithIndexTest<TType>
{
};

template <typename TType>
class ReduceWithNoIndexInt8 : public ReduceWithIndexTest<TType>
{
};

template <typename TType>
class ReduceWithNoIndexHalf : public ReduceWithIndexTest<TType>
{
};

template <typename TType>
class ReduceWithNoIndexBHalfFloat : public ReduceWithIndexTest<TType>
{
};

TYPED_TEST_SUITE(ReduceWithNoIndexFloat, Reduce_float_types);
TYPED_TEST_SUITE(ReduceWithNoIndexDouble, Reduce_double_types);
TYPED_TEST_SUITE(ReduceWithNoIndexInt8, Reduce_int8t_types);
TYPED_TEST_SUITE(ReduceWithNoIndexHalf, Reduce_half_types);
TYPED_TEST_SUITE(ReduceWithNoIndexBHalfFloat, Reduce_bhalf_float_Types);

TYPED_TEST(ReduceWithNoIndexFloat, ReduceWithNoIndexTestFloat_AMAX)
{
    // trigger Run() -> Generic
    this->template Run<ReduceTensorOp::AMAX>();
}

TYPED_TEST(ReduceWithNoIndexFloat, ReduceWithNoIndexTestFloat_MIN)
{
    // trigger Run() -> Generic
    this->template Run<ReduceTensorOp::MIN>();
}

TYPED_TEST(ReduceWithNoIndexFloat, ReduceWithNoIndexTestFloat_MAX)
{
    // trigger Run() -> Generic
    this->template Run<ReduceTensorOp::MAX>();
}

TYPED_TEST(ReduceWithNoIndexDouble, ReduceWithNoIndexTestDouble_AMAX)
{
    // trigger Run() -> Generic
    this->template Run<ReduceTensorOp::AMAX>();
}

TYPED_TEST(ReduceWithNoIndexDouble, ReduceWithNoIndexTestDouble_MIN)
{
    // trigger Run() -> Generic
    this->template Run<ReduceTensorOp::MIN>();
}

TYPED_TEST(ReduceWithNoIndexDouble, ReduceWithNoIndexTestDouble_MAX)
{
    // trigger Run() -> Generic
    this->template Run<ReduceTensorOp::MAX>();
}

TYPED_TEST(ReduceWithNoIndexInt8, ReduceWithNoIndexTestInt8_AMAX)
{
    // trigger Run() -> Generic
    this->template Run<ReduceTensorOp::AMAX>();
}

TYPED_TEST(ReduceWithNoIndexInt8, ReduceWithNoIndexTestInt8_MIN)
{
    // trigger Run() -> Generic
    this->template Run<ReduceTensorOp::MIN>();
}

TYPED_TEST(ReduceWithNoIndexInt8, ReduceWithNoIndexTestInt8_MAX)
{
    // trigger Run() -> Generic
    this->template Run<ReduceTensorOp::MAX>();
}

TYPED_TEST(ReduceWithNoIndexHalf, ReduceWithNoIndexTestHalf_AMAX)
{
    // trigger Run() -> Generic
    this->template Run<ReduceTensorOp::AMAX>();
}

TYPED_TEST(ReduceWithNoIndexHalf, ReduceWithNoIndexTestHalf_MIN)
{
    // trigger Run() -> Generic
    this->template Run<ReduceTensorOp::MIN>();
}

TYPED_TEST(ReduceWithNoIndexHalf, ReduceWithNoIndexTestHalf_MAX)
{
    // trigger Run() -> Generic
    this->template Run<ReduceTensorOp::MAX>();
}

TYPED_TEST(ReduceWithNoIndexBHalfFloat, ReduceWithNoIndexTesBHalfFloat_AMAX)
{
    // trigger Run() -> Generic
    this->template Run<ReduceTensorOp::AMAX>();
}

TYPED_TEST(ReduceWithNoIndexBHalfFloat, ReduceWithNoIndexTestBHalfFloat_MIN)
{
    // trigger Run() -> Generic
    this->template Run<ReduceTensorOp::MIN>();
}

TYPED_TEST(ReduceWithNoIndexBHalfFloat, ReduceWithNoIndexTestBHalfFloat_MAX)
{
    // trigger Run() -> Generic
    this->template Run<ReduceTensorOp::MAX>();
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    if(argc == 1) {}
    else if(argc == 3)
    {
        param_mask     = strtol(argv[1], nullptr, 0);
        instance_index = atoi(argv[2]);
    }
    else
    {
        std::cout << "Usage of " << argv[0] << std::endl;
        std::cout << "Arg1,2: param_mask instance_index(-1 means all)" << std::endl;
    }
    return RUN_ALL_TESTS();
}
