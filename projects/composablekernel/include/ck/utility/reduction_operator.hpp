// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/ck.hpp"
#include "ck/utility/data_type.hpp"
#include "ck/utility/type.hpp"
#include "ck/utility/type_convert.hpp"

namespace ck {

namespace reduce {

// Every binary operator used in reduction is represented by a templated functor class. Each functor
// class must provide at least
// three members:
// 1) GetIdentityValue() -- the interface to return the "identity element" for the binary
// operator, "identity element" is the unique
//                    element in the algebraic space that doesn't affect the value of other elements
//                    when operated against them, and the concept is similar to zero vector in
//                    vector space
//                    (http://pages.cs.wisc.edu/~matthewb/pages/notes/pdf/linearalgebra/VectorSpaces.pdf).
// 2) IsCompatibleInMemoryDataOperation() -- return true if the reduction task corresponding to this
// operator can use the InMemoryDataOperation to finalize, or else it return false
// 3) operator() -- the first argument of the operator must be both an input & output, and the
//                  corresponding variable usually stores
//                  the accumulated result of many operator() calls; the second argument is only an
//                  input. For indexable binary
//                  operator, the second version of operator() has third argument (which is an
//                  output) to indicate whether the
//                  accumulated value (the first argument) has changed, in which case the recorded
//                  accumulated index also need be
//                  changed.

struct Add
{
    template <typename T>
    __host__ __device__ static constexpr T GetIdentityValue()
    {
        return type_convert<T>(0.0f);
    };

    __host__ __device__ static constexpr bool
    IsCompatibleInMemoryDataOperation(InMemoryDataOperationEnum operation)
    {
        return operation == InMemoryDataOperationEnum::AtomicAdd ||
               operation == InMemoryDataOperationEnum::Set;
    };

    template <typename T>
    __host__ __device__ inline constexpr void operator()(T& a, T b) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, int32_t>::value || is_same<T, int8_t>::value,
                      "The data type is not supported by the Add accumulator!");

        a = a + b;
    }

    __host__ __device__ inline constexpr void operator()(f8_t& a, f8_t b) const
    {
        float a_ = type_convert<float>(a);
        float b_ = type_convert<float>(b);

        a = type_convert<f8_t>(a_ + b_);
    }

    __host__ __device__ inline constexpr void operator()(half_t& a, half_t b) const
    {
        float a_ = type_convert<float>(a);
        float b_ = type_convert<float>(b);

        a = type_convert<half_t>(a_ + b_);
    }

    __host__ __device__ inline constexpr void operator()(bhalf_t& a, bhalf_t b) const
    {
        float a_ = type_convert<float>(a);
        float b_ = type_convert<float>(b);

        a = type_convert<bhalf_t>(a_ + b_);
    }
};

struct SquaredAdd
{
    template <class T>
    __host__ __device__ static constexpr T GetIdentityValue()
    {
        return type_convert<T>(0.0f);
    };

    __host__ __device__ static constexpr bool
    IsCompatibleInMemoryDataOperation(InMemoryDataOperationEnum operation)
    {
        return operation == InMemoryDataOperationEnum::AtomicAdd ||
               operation == InMemoryDataOperationEnum::Set;
    };

    template <class T>
    __host__ __device__ inline constexpr void operator()(T& a, T b) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, half_t>::value || is_same<T, int32_t>::value ||
                          is_same<T, int8_t>::value,
                      "The data type is not supported by the SquaredAdd accumulator!");

        a = a + b * b;
    }
};

struct Mul
{
    template <typename T>
    __host__ __device__ static constexpr T GetIdentityValue()
    {
        return type_convert<T>(1.0f);
    };

    __host__ __device__ static constexpr bool
    IsCompatibleInMemoryDataOperation(InMemoryDataOperationEnum operation)
    {
        return operation == InMemoryDataOperationEnum::Set;
    };

    template <typename T>
    __host__ __device__ inline constexpr void operator()(T& a, T b) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, int32_t>::value || is_same<T, int8_t>::value,
                      "The data type is not supported by the Mul accumulator!");

        a = a * b;
    }

    __host__ __device__ inline constexpr void operator()(f8_t& a, f8_t b) const
    {
        float a_ = type_convert<float>(a);
        float b_ = type_convert<float>(b);

        a = type_convert<f8_t>(a_ * b_);
    }

    __host__ __device__ inline constexpr void operator()(half_t& a, half_t b) const
    {
        float a_ = type_convert<float>(a);
        float b_ = type_convert<float>(b);

        a = type_convert<half_t>(a_ * b_);
    }

    __host__ __device__ inline constexpr void operator()(bhalf_t& a, bhalf_t b) const
    {
        float a_ = type_convert<float>(a);
        float b_ = type_convert<float>(b);

        a = type_convert<bhalf_t>(a_ * b_);
    }
};

struct Max
{
    template <typename T>
    __host__ __device__ static constexpr T GetIdentityValue()
    {
        if constexpr(is_same_v<T, bhalf_t>)
        {
            float val = NumericLimits<float>::Lowest();
            return type_convert<bhalf_t>(val);
        }
        if constexpr(is_same_v<T, f8_t>)
        {
            float val = NumericLimits<float>::Lowest();
            return type_convert<f8_t>(val);
        }
        if constexpr(is_same_v<T, half_t>)
        {
            float val = NumericLimits<float>::Lowest();
            return type_convert<half_t>(val);
        }
        else
        {
            return NumericLimits<T>::Lowest();
        }
    };

    __host__ __device__ static constexpr bool
    IsCompatibleInMemoryDataOperation(InMemoryDataOperationEnum operation)
    {
        // ToChange: atomic_max to be added
        return operation == InMemoryDataOperationEnum::Set;
    };

    template <typename T>
    __host__ __device__ inline constexpr void operator()(T& a, T b) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, int32_t>::value || is_same<T, int8_t>::value,
                      "The data type is not supported by the Max accumulator!");

        if(a < b)
            a = b;
    }

    __host__ __device__ inline constexpr void operator()(bhalf_t& a, bhalf_t b) const
    {
        float a_ = type_convert<float>(a);
        float b_ = type_convert<float>(b);

        if(a_ < b_)
            a = b;
    }

    __host__ __device__ inline constexpr void operator()(half_t& a, half_t b) const
    {
        float a_ = type_convert<float>(a);
        float b_ = type_convert<float>(b);

        if(a_ < b_)
            a = b;
    }

    __host__ __device__ inline constexpr void operator()(f8_t& a, f8_t b) const
    {
        float a_ = type_convert<float>(a);
        float b_ = type_convert<float>(b);

        if(a_ < b_)
            a = b;
    }

    template <typename T>
    __host__ __device__ inline constexpr void operator()(T& a, T b, bool& changed) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, int32_t>::value || is_same<T, int8_t>::value,
                      "The data type is not supported by the Max accumulator!");

        if(a < b)
        {
            a       = b;
            changed = true;
        }
    }

    __host__ __device__ inline constexpr void operator()(bhalf_t& a, bhalf_t b, bool& changed) const
    {
        float a_ = type_convert<float>(a);
        float b_ = type_convert<float>(b);

        if(a_ < b_)
        {
            a       = b;
            changed = true;
        }
    }

    __host__ __device__ inline constexpr void operator()(half_t& a, half_t b, bool& changed) const
    {
        float a_ = type_convert<float>(a);
        float b_ = type_convert<float>(b);

        if(a_ < b_)
        {
            a       = b;
            changed = true;
        }
    }

    __host__ __device__ inline constexpr void operator()(f8_t& a, f8_t b, bool& changed) const
    {
        float a_ = type_convert<float>(a);
        float b_ = type_convert<float>(b);

        if(a_ < b_)
        {
            a       = b;
            changed = true;
        }
    }
};

struct Min
{
    template <typename T>
    __host__ __device__ static constexpr T GetIdentityValue()
    {
        if constexpr(is_same_v<T, bhalf_t>)
        {
            float val = NumericLimits<float>::Max();
            return type_convert<bhalf_t>(val);
        }
        else if constexpr(is_same_v<T, half_t>)
        {
            float val = NumericLimits<float>::Max();
            return type_convert<half_t>(val);
        }
        else if constexpr(is_same_v<T, f8_t>)
        {
            float val = NumericLimits<float>::Max();
            return type_convert<f8_t>(val);
        }
        else
        {
            return NumericLimits<T>::Max();
        }
        return NumericLimits<T>::Max();
    };

    __host__ __device__ static constexpr bool
    IsCompatibleInMemoryDataOperation(InMemoryDataOperationEnum operation)
    {
        // ToChange: atomic_min to be added
        return operation == InMemoryDataOperationEnum::Set;
    };

    template <typename T>
    __host__ __device__ inline constexpr void operator()(T& a, T b) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, int32_t>::value || is_same<T, int8_t>::value,
                      "The data type is not supported by the Min accumulator!");

        if(a > b)
            a = b;
    }

    __host__ __device__ inline constexpr void operator()(bhalf_t& a, bhalf_t b) const
    {
        float a_ = type_convert<float>(a);
        float b_ = type_convert<float>(b);

        if(a_ > b_)
            a = b;
    }

    __host__ __device__ inline constexpr void operator()(half_t& a, half_t b) const
    {
        float a_ = type_convert<float>(a);
        float b_ = type_convert<float>(b);

        if(a_ > b_)
            a = b;
    }

    __host__ __device__ inline constexpr void operator()(f8_t& a, f8_t b) const
    {
        float a_ = type_convert<float>(a);
        float b_ = type_convert<float>(b);

        if(a_ > b_)
            a = b;
    }

    template <typename T>
    __host__ __device__ inline constexpr void operator()(T& a, T b, bool& changed) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, half_t>::value || is_same<T, int32_t>::value ||
                          is_same<T, int8_t>::value,
                      "The data type is not supported by the Min accumulator!");

        if(a > b)
        {
            a       = b;
            changed = true;
        }
    }

    __host__ __device__ inline constexpr void operator()(bhalf_t& a, bhalf_t b, bool& changed) const
    {
        float a_ = type_convert<float>(a);
        float b_ = type_convert<float>(b);

        if(a_ > b_)
        {
            a       = b;
            changed = true;
        }
    }

    __host__ __device__ inline constexpr void operator()(half_t& a, half_t b, bool& changed) const
    {
        float a_ = type_convert<float>(a);
        float b_ = type_convert<float>(b);

        if(a_ > b_)
        {
            a       = b;
            changed = true;
        }
    }

    __host__ __device__ inline constexpr void operator()(f8_t& a, f8_t b, bool& changed) const
    {
        float a_ = type_convert<float>(a);
        float b_ = type_convert<float>(b);

        if(a_ > b_)
        {
            a       = b;
            changed = true;
        }
    }
};

struct AMax
{
    template <typename T>
    __host__ __device__ static constexpr T GetIdentityValue()
    {
        return type_convert<T>(0.0f);
    };

    __host__ __device__ static constexpr bool
    IsCompatibleInMemoryDataOperation(InMemoryDataOperationEnum operation)
    {
        // ToChange: atomic_max to be added
        return operation == InMemoryDataOperationEnum::Set;
    };

    template <typename T>
    __host__ __device__ inline constexpr void operator()(T& a, T b) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, half_t>::value || is_same<T, int32_t>::value ||
                          is_same<T, int8_t>::value,
                      "The data type is not supported by the AMax accumulator!");

        if(a < b)
            a = b;
    }

    __host__ __device__ inline constexpr void operator()(f8_t& a, f8_t b) const
    {
        float a_ = type_convert<float>(a);
        float b_ = type_convert<float>(b);

        if(a_ < b_)
            a = b;
    }

    template <typename T>
    __host__ __device__ inline constexpr void operator()(T& a, T b, bool& changed) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, half_t>::value || is_same<T, int32_t>::value ||
                          is_same<T, int8_t>::value,
                      "The data type is not supported by the AMax accumulator!");

        if(a < b)
        {
            a       = b;
            changed = true;
        }
    }

    __host__ __device__ inline constexpr void operator()(f8_t& a, f8_t b, bool& changed) const
    {
        float a_ = type_convert<float>(a);
        float b_ = type_convert<float>(b);

        if(a_ < b_)
        {
            a       = b;
            changed = true;
        }
    }
};

template <typename T>
constexpr T GetIdentityValueForInMemoryDataOperation(InMemoryDataOperationEnum operation)
{
    T result = ck::type_convert<T>(0.0f);

    if(operation == InMemoryDataOperationEnum::AtomicMax)
        result = ck::NumericLimits<T>::Lowest();

    return (result);
};

template <InMemoryDataOperationEnum Operation, typename DataType>
struct InMemoryDataOperationSupportedOnDataType
{
    static constexpr bool value = false;
};

template <typename DataType>
struct InMemoryDataOperationSupportedOnDataType<InMemoryDataOperationEnum::AtomicAdd, DataType>
{
    static constexpr bool value =
        is_same<DataType, float>::value || is_same<DataType, double>::value;
};

template <typename DataType>
struct InMemoryDataOperationSupportedOnDataType<InMemoryDataOperationEnum::AtomicMax, DataType>
{
    static constexpr bool value =
        is_same<DataType, float>::value || is_same<DataType, double>::value;
};

template <typename DataType>
struct InMemoryDataOperationSupportedOnDataType<InMemoryDataOperationEnum::Set, DataType>
{
    static constexpr bool value =
        is_same<DataType, float>::value || is_same<DataType, double>::value ||
        is_same<DataType, half_t>::value || is_same<DataType, bhalf_t>::value ||
        is_same<DataType, int8_t>::value || is_same<DataType, int32_t>::value ||
        is_same<DataType, f8_t>::value;
};

template <typename DataType>
struct InMemoryDataOperationSupportedOnDataType<InMemoryDataOperationEnum::Add, DataType>
{
    static constexpr bool value =
        is_same<DataType, float>::value || is_same<DataType, double>::value ||
        is_same<DataType, half_t>::value || is_same<DataType, int8_t>::value ||
        is_same<DataType, int32_t>::value;
};

} // namespace reduce
} // namespace ck
