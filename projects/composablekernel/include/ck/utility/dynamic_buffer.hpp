// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/ck.hpp"
#include "ck/utility/data_type.hpp"
#include "enable_if.hpp"
#include "c_style_pointer_cast.hpp"
#if __clang_major__ >= 20
#include "amd_buffer_addressing_builtins.hpp"
#else
#include "amd_buffer_addressing.hpp"
#endif
#include "amd_transpose_load.hpp"
#include "generic_memory_space_atomic.hpp"

namespace ck {

// T may be scalar or vector
// X may be scalar or vector
// T and X have same scalar type
// X contains multiple T
template <AddressSpaceEnum BufferAddressSpace,
          typename T,
          typename ElementSpaceSize,
          bool InvalidElementUseNumericalZeroValue,
          AmdBufferCoherenceEnum coherence = AmdBufferCoherenceEnum::DefaultCoherence,
          typename IndexType               = index_t>
struct DynamicBuffer
{
    using type = T;

    T* p_data_;
    ElementSpaceSize element_space_size_;
    T invalid_element_value_ = T{0};

    // XXX: PackedSize semantics for pk_i4_t is different from the other packed types.
    // Objects of f4x2_pk_t and f6_pk_t are counted as 1 element, while
    // objects of pk_i4_t are counted as 2 elements. Therefore, element_space_size_ for pk_i4_t must
    // be divided by 2 to correctly represent the number of addressable elements.
    static constexpr index_t PackedSize = []() {
        if constexpr(is_same_v<remove_cvref_t<T>, pk_i4_t>)
            return 2;
        else
            return 1;
    }();

    __host__ __device__ constexpr DynamicBuffer(T* p_data, ElementSpaceSize element_space_size)
        : p_data_{p_data}, element_space_size_{element_space_size}
    {
    }

    __host__ __device__ constexpr DynamicBuffer(T* p_data,
                                                ElementSpaceSize element_space_size,
                                                T invalid_element_value)
        : p_data_{p_data},
          element_space_size_{element_space_size},
          invalid_element_value_{invalid_element_value}
    {
    }

    __host__ __device__ static constexpr AddressSpaceEnum GetAddressSpace()
    {
        return BufferAddressSpace;
    }

    __host__ __device__ constexpr const T& operator[](IndexType i) const { return p_data_[i]; }

    __host__ __device__ constexpr T& operator()(IndexType i) { return p_data_[i]; }

    template <typename X,
              bool DoTranspose               = false,
              typename enable_if<is_same<typename scalar_type<remove_cvref_t<X>>::type,
                                         typename scalar_type<remove_cvref_t<T>>::type>::value ||
                                     !is_native_type<X>(),
                                 bool>::type = false>
    __host__ __device__ constexpr auto Get(IndexType i, bool is_valid_element) const
    {
        // X contains multiple T
        constexpr index_t scalar_per_t_vector = scalar_type<remove_cvref_t<T>>::vector_size;

        constexpr index_t scalar_per_x_vector = scalar_type<remove_cvref_t<X>>::vector_size;

        static_assert(scalar_per_x_vector % scalar_per_t_vector == 0,
                      "wrong! X should contain multiple T");

#if CK_USE_AMD_BUFFER_LOAD
        bool constexpr use_amd_buffer_addressing = sizeof(IndexType) <= sizeof(int32_t);
#else
        bool constexpr use_amd_buffer_addressing = false;
#endif

        if constexpr(GetAddressSpace() == AddressSpaceEnum::Global && use_amd_buffer_addressing &&
                     !DoTranspose)
        {
            constexpr index_t t_per_x = scalar_per_x_vector / scalar_per_t_vector;

            if constexpr(InvalidElementUseNumericalZeroValue)
            {
                return amd_buffer_load_invalid_element_return_zero<remove_cvref_t<T>,
                                                                   t_per_x,
                                                                   coherence>(
                    p_data_, i, is_valid_element, element_space_size_ / PackedSize);
            }
            else
            {
                return amd_buffer_load_invalid_element_return_customized_value<remove_cvref_t<T>,
                                                                               t_per_x,
                                                                               coherence>(
                    p_data_,
                    i,
                    is_valid_element,
                    element_space_size_ / PackedSize,
                    invalid_element_value_);
            }
        }
        else if constexpr(GetAddressSpace() == AddressSpaceEnum::Global && DoTranspose)
        {
#ifdef __gfx12__
            return amd_global_load_transpose_to_vgpr(p_data_ + i);
#else
            static_assert(!DoTranspose, "load-with-transpose only supported on gfx12+");
#endif
        }
        else
        {
            if(is_valid_element)
            {
#if CK_EXPERIMENTAL_USE_MEMCPY_FOR_VECTOR_ACCESS
                X tmp;

                __builtin_memcpy(&tmp, &(p_data_[i]), sizeof(X));

                return tmp;
#else
                return *c_style_pointer_cast<const X*>(&p_data_[i]);
#endif
            }
            else
            {
                if constexpr(InvalidElementUseNumericalZeroValue)
                {
                    return X{0};
                }
                else
                {
                    return X{invalid_element_value_};
                }
            }
        }
    }

    template <InMemoryDataOperationEnum Op,
              typename X,
              typename enable_if<is_same<typename scalar_type<remove_cvref_t<X>>::type,
                                         typename scalar_type<remove_cvref_t<T>>::type>::value,
                                 bool>::type = false>
    __host__ __device__ void Update(IndexType i, bool is_valid_element, const X& x)
    {
        if constexpr(Op == InMemoryDataOperationEnum::Set)
        {
            this->template Set<X>(i, is_valid_element, x);
        }
        else if constexpr(Op == InMemoryDataOperationEnum::AtomicAdd)
        {
            this->template AtomicAdd<X>(i, is_valid_element, x);
        }
        else if constexpr(Op == InMemoryDataOperationEnum::AtomicMax)
        {
            this->template AtomicMax<X>(i, is_valid_element, x);
        }
        else if constexpr(Op == InMemoryDataOperationEnum::Add)
        {
            auto tmp       = this->template Get<X>(i, is_valid_element);
            using scalar_t = typename scalar_type<remove_cvref_t<T>>::type;
            // handle bfloat addition
            if constexpr(is_same_v<scalar_t, bhalf_t>)
            {
                if constexpr(is_scalar_type<X>::value)
                {
                    // Scalar type
                    auto result =
                        type_convert<X>(type_convert<float>(x) + type_convert<float>(tmp));
                    this->template Set<X>(i, is_valid_element, result);
                }
                else
                {
                    // Vector type
                    constexpr auto vector_size = scalar_type<remove_cvref_t<X>>::vector_size;
                    const vector_type<scalar_t, vector_size> a_vector{tmp};
                    const vector_type<scalar_t, vector_size> b_vector{x};
                    static_for<0, vector_size, 1>{}([&](auto idx) {
                        auto result = type_convert<scalar_t>(
                            type_convert<float>(a_vector.template AsType<scalar_t>()[idx]) +
                            type_convert<float>(b_vector.template AsType<scalar_t>()[idx]));
                        this->template Set<scalar_t>(i + idx, is_valid_element, result);
                    });
                }
            }
            else
            {
                this->template Set<X>(i, is_valid_element, x + tmp);
            }
        }
    }

    template <typename DstBuffer, index_t NumElemsPerThread>
    __host__ __device__ void DirectCopyToLds(DstBuffer& dst_buf,
                                             IndexType src_offset,
                                             IndexType dst_offset,
                                             bool is_valid_element) const
    {
        // Copy data from global to LDS memory using direct loads.
        static_assert(GetAddressSpace() == AddressSpaceEnum::Global,
                      "Source data must come from a global memory buffer.");
        static_assert(DstBuffer::GetAddressSpace() == AddressSpaceEnum::Lds,
                      "Destination data must be stored in an LDS memory buffer.");

        amd_direct_load_global_to_lds<T, NumElemsPerThread>(p_data_,
                                                            src_offset,
                                                            dst_buf.p_data_,
                                                            dst_offset,
                                                            is_valid_element,
                                                            element_space_size_ / PackedSize);
    }

    template <typename X,
              typename enable_if<is_same<typename scalar_type<remove_cvref_t<X>>::type,
                                         typename scalar_type<remove_cvref_t<T>>::type>::value ||
                                     !is_native_type<X>(),
                                 bool>::type = false>
    __host__ __device__ void Set(IndexType i, bool is_valid_element, const X& x)
    {
        // X contains multiple T
        constexpr index_t scalar_per_t_vector = scalar_type<remove_cvref_t<T>>::vector_size;

        constexpr index_t scalar_per_x_vector = scalar_type<remove_cvref_t<X>>::vector_size;

        static_assert(scalar_per_x_vector % scalar_per_t_vector == 0,
                      "wrong! X should contain multiple T");

#if CK_USE_AMD_BUFFER_LOAD
        bool constexpr use_amd_buffer_addressing = sizeof(IndexType) <= sizeof(int32_t);
#else
        bool constexpr use_amd_buffer_addressing = false;
#endif

#if CK_WORKAROUND_SWDEV_XXXXXX_INT8_DS_WRITE_ISSUE
        bool constexpr workaround_int8_ds_write_issue = true;
#else
        bool constexpr workaround_int8_ds_write_issue = false;
#endif

        if constexpr(GetAddressSpace() == AddressSpaceEnum::Global && use_amd_buffer_addressing)
        {
            constexpr index_t t_per_x = scalar_per_x_vector / scalar_per_t_vector;

            amd_buffer_store<remove_cvref_t<T>, t_per_x, coherence>(
                x, p_data_, i, is_valid_element, element_space_size_ / PackedSize);
        }
        else if constexpr(GetAddressSpace() == AddressSpaceEnum::Lds &&
                          is_same<typename scalar_type<remove_cvref_t<T>>::type, int8_t>::value &&
                          workaround_int8_ds_write_issue)
        {
            if(is_valid_element)
            {
                // HACK: compiler would lower IR "store<i8, 16> address_space(3)" into inefficient
                // ISA, so I try to let compiler emit IR "store<i32, 4>" which would be lower to
                // ds_write_b128
                // TODO: remove this after compiler fix
                static_assert((is_same<remove_cvref_t<T>, int8_t>::value &&
                               is_same<remove_cvref_t<X>, int8_t>::value) ||
                                  (is_same<remove_cvref_t<T>, int8_t>::value &&
                                   is_same<remove_cvref_t<X>, int8x2_t>::value) ||
                                  (is_same<remove_cvref_t<T>, int8_t>::value &&
                                   is_same<remove_cvref_t<X>, int8x4_t>::value) ||
                                  (is_same<remove_cvref_t<T>, int8_t>::value &&
                                   is_same<remove_cvref_t<X>, int8x8_t>::value) ||
                                  (is_same<remove_cvref_t<T>, int8_t>::value &&
                                   is_same<remove_cvref_t<X>, int8x16_t>::value) ||
                                  (is_same<remove_cvref_t<T>, int8x4_t>::value &&
                                   is_same<remove_cvref_t<X>, int8x4_t>::value) ||
                                  (is_same<remove_cvref_t<T>, int8x8_t>::value &&
                                   is_same<remove_cvref_t<X>, int8x8_t>::value) ||
                                  (is_same<remove_cvref_t<T>, int8x16_t>::value &&
                                   is_same<remove_cvref_t<X>, int8x16_t>::value),
                              "wrong! not implemented for this combination, please add "
                              "implementation");

                if constexpr(is_same<remove_cvref_t<T>, int8_t>::value &&
                             is_same<remove_cvref_t<X>, int8_t>::value)
                {
                    // HACK: cast pointer of x is bad
                    // TODO: remove this after compiler fix
                    *c_style_pointer_cast<int8_t*>(&p_data_[i]) =
                        *c_style_pointer_cast<const int8_t*>(&x);
                }
                else if constexpr(is_same<remove_cvref_t<T>, int8_t>::value &&
                                  is_same<remove_cvref_t<X>, int8x2_t>::value)
                {
                    // HACK: cast pointer of x is bad
                    // TODO: remove this after compiler fix
                    *c_style_pointer_cast<int16_t*>(&p_data_[i]) =
                        *c_style_pointer_cast<const int16_t*>(&x);
                }
                else if constexpr(is_same<remove_cvref_t<T>, int8_t>::value &&
                                  is_same<remove_cvref_t<X>, int8x4_t>::value)
                {
                    // HACK: cast pointer of x is bad
                    // TODO: remove this after compiler fix
                    *c_style_pointer_cast<int32_t*>(&p_data_[i]) =
                        *c_style_pointer_cast<const int32_t*>(&x);
                }
                else if constexpr(is_same<remove_cvref_t<T>, int8_t>::value &&
                                  is_same<remove_cvref_t<X>, int8x8_t>::value)
                {
                    // HACK: cast pointer of x is bad
                    // TODO: remove this after compiler fix
                    *c_style_pointer_cast<int32x2_t*>(&p_data_[i]) =
                        *c_style_pointer_cast<const int32x2_t*>(&x);
                }
                else if constexpr(is_same<remove_cvref_t<T>, int8_t>::value &&
                                  is_same<remove_cvref_t<X>, int8x16_t>::value)
                {
                    // HACK: cast pointer of x is bad
                    // TODO: remove this after compiler fix
                    *c_style_pointer_cast<int32x4_t*>(&p_data_[i]) =
                        *c_style_pointer_cast<const int32x4_t*>(&x);
                }
                else if constexpr(is_same<remove_cvref_t<T>, int8x4_t>::value &&
                                  is_same<remove_cvref_t<X>, int8x4_t>::value)
                {
                    // HACK: cast pointer of x is bad
                    // TODO: remove this after compiler fix
                    *c_style_pointer_cast<int32_t*>(&p_data_[i]) =
                        *c_style_pointer_cast<const int32_t*>(&x);
                }
                else if constexpr(is_same<remove_cvref_t<T>, int8x8_t>::value &&
                                  is_same<remove_cvref_t<X>, int8x8_t>::value)
                {
                    // HACK: cast pointer of x is bad
                    // TODO: remove this after compiler fix
                    *c_style_pointer_cast<int32x2_t*>(&p_data_[i]) =
                        *c_style_pointer_cast<const int32x2_t*>(&x);
                }
                else if constexpr(is_same<remove_cvref_t<T>, int8x16_t>::value &&
                                  is_same<remove_cvref_t<X>, int8x16_t>::value)
                {
                    // HACK: cast pointer of x is bad
                    // TODO: remove this after compiler fix
                    *c_style_pointer_cast<int32x4_t*>(&p_data_[i]) =
                        *c_style_pointer_cast<const int32x4_t*>(&x);
                }
            }
        }
        else
        {
            if(is_valid_element)
            {
#if 0
                X tmp = x;

                __builtin_memcpy(&(p_data_[i]), &tmp, sizeof(X));
#else
                // if(i >= 2169041600)
                *c_style_pointer_cast<X*>(&p_data_[i]) = x;
#endif
            }
        }
    }

    template <typename X,
              typename enable_if<is_same<typename scalar_type<remove_cvref_t<X>>::type,
                                         typename scalar_type<remove_cvref_t<T>>::type>::value,
                                 bool>::type = false>
    __host__ __device__ void AtomicAdd(IndexType i, bool is_valid_element, const X& x)
    {
        using scalar_t = typename scalar_type<remove_cvref_t<T>>::type;

        // X contains multiple T
        constexpr index_t scalar_per_t_vector = scalar_type<remove_cvref_t<T>>::vector_size;

        constexpr index_t scalar_per_x_vector = scalar_type<remove_cvref_t<X>>::vector_size;

        static_assert(scalar_per_x_vector % scalar_per_t_vector == 0,
                      "wrong! X should contain multiple T");

        static_assert(GetAddressSpace() == AddressSpaceEnum::Global, "only support global mem");

#if CK_USE_AMD_BUFFER_ATOMIC_ADD_INTEGER && CK_USE_AMD_BUFFER_ATOMIC_ADD_FLOAT
        bool constexpr use_amd_buffer_addressing =
            is_same_v<remove_cvref_t<scalar_t>, int32_t> ||
            is_same_v<remove_cvref_t<scalar_t>, float> ||
            (is_same_v<remove_cvref_t<scalar_t>, half_t> && scalar_per_x_vector % 2 == 0) ||
            (is_same_v<remove_cvref_t<scalar_t>, bhalf_t> && scalar_per_x_vector % 2 == 0);
#elif CK_USE_AMD_BUFFER_ATOMIC_ADD_INTEGER && (!CK_USE_AMD_BUFFER_ATOMIC_ADD_FLOAT)
        bool constexpr use_amd_buffer_addressing =
            sizeof(IndexType) <= sizeof(int32_t) && is_same_v<remove_cvref_t<scalar_t>, int32_t>;
#elif(!CK_USE_AMD_BUFFER_ATOMIC_ADD_INTEGER) && CK_USE_AMD_BUFFER_ATOMIC_ADD_FLOAT
        bool constexpr use_amd_buffer_addressing =
            sizeof(IndexType) <= sizeof(int32_t) &&
            (is_same_v<remove_cvref_t<scalar_t>, float> ||
             (is_same_v<remove_cvref_t<scalar_t>, half_t> && scalar_per_x_vector % 2 == 0) ||
             (is_same_v<remove_cvref_t<scalar_t>, bhalf_t> && scalar_per_x_vector % 2 == 0));
#else
        bool constexpr use_amd_buffer_addressing = false;
#endif

        if constexpr(use_amd_buffer_addressing)
        {
            constexpr index_t t_per_x = scalar_per_x_vector / scalar_per_t_vector;

            amd_buffer_atomic_add<remove_cvref_t<T>, t_per_x>(
                x, p_data_, i, is_valid_element, element_space_size_ / PackedSize);
        }
        else
        {
            if(is_valid_element)
            {
                atomic_add<X>(c_style_pointer_cast<X*>(&p_data_[i]), x);
            }
        }
    }

    template <typename X,
              typename enable_if<is_same<typename scalar_type<remove_cvref_t<X>>::type,
                                         typename scalar_type<remove_cvref_t<T>>::type>::value,
                                 bool>::type = false>
    __host__ __device__ void AtomicMax(IndexType i, bool is_valid_element, const X& x)
    {
        // X contains multiple T
        constexpr IndexType scalar_per_t_vector = scalar_type<remove_cvref_t<T>>::vector_size;

        constexpr IndexType scalar_per_x_vector = scalar_type<remove_cvref_t<X>>::vector_size;

        static_assert(scalar_per_x_vector % scalar_per_t_vector == 0,
                      "wrong! X should contain multiple T");

        static_assert(GetAddressSpace() == AddressSpaceEnum::Global, "only support global mem");

#if CK_USE_AMD_BUFFER_ATOMIC_MAX_FLOAT64
        using scalar_t = typename scalar_type<remove_cvref_t<T>>::type;
        bool constexpr use_amd_buffer_addressing =
            sizeof(IndexType) <= sizeof(int32_t) && is_same_v<remove_cvref_t<scalar_t>, double>;
#else
        bool constexpr use_amd_buffer_addressing = false;
#endif

        if constexpr(use_amd_buffer_addressing)
        {
            constexpr index_t t_per_x = scalar_per_x_vector / scalar_per_t_vector;

            amd_buffer_atomic_max<remove_cvref_t<T>, t_per_x>(
                x, p_data_, i, is_valid_element, element_space_size_ / PackedSize);
        }
        else if(is_valid_element)
        {
            atomic_max<X>(c_style_pointer_cast<X*>(&p_data_[i]), x);
        }
    }

    __host__ __device__ static constexpr bool IsStaticBuffer() { return false; }

    __host__ __device__ static constexpr bool IsDynamicBuffer() { return true; }
};

template <AddressSpaceEnum BufferAddressSpace,
          AmdBufferCoherenceEnum coherence = AmdBufferCoherenceEnum::DefaultCoherence,
          typename T,
          typename ElementSpaceSize>
__host__ __device__ constexpr auto make_dynamic_buffer(T* p, ElementSpaceSize element_space_size)
{
    return DynamicBuffer<BufferAddressSpace, T, ElementSpaceSize, true, coherence>{
        p, element_space_size};
}

template <AddressSpaceEnum BufferAddressSpace,
          AmdBufferCoherenceEnum coherence = AmdBufferCoherenceEnum::DefaultCoherence,
          typename T,
          typename ElementSpaceSize>
__host__ __device__ constexpr auto make_long_dynamic_buffer(T* p,
                                                            ElementSpaceSize element_space_size)
{
    return DynamicBuffer<BufferAddressSpace, T, ElementSpaceSize, true, coherence, long_index_t>{
        p, element_space_size};
}

template <
    AddressSpaceEnum BufferAddressSpace,
    AmdBufferCoherenceEnum coherence = AmdBufferCoherenceEnum::DefaultCoherence,
    typename T,
    typename ElementSpaceSize,
    typename X,
    typename enable_if<is_same<remove_cvref_t<T>, remove_cvref_t<X>>::value, bool>::type = false>
__host__ __device__ constexpr auto
make_dynamic_buffer(T* p, ElementSpaceSize element_space_size, X invalid_element_value)
{
    return DynamicBuffer<BufferAddressSpace, T, ElementSpaceSize, false, coherence>{
        p, element_space_size, invalid_element_value};
}

} // namespace ck
