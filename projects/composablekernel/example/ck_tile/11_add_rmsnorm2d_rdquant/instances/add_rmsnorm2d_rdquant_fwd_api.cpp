// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <ck_tile/core.hpp>
#include "add_rmsnorm2d_rdquant_fwd.hpp"

template <typename InputDataType_,
          typename QuantizedDataType_,
          ck_tile::index_t Repeat_M_,         // each thread repeat along M
          ck_tile::index_t Repeat_N_,         // each thread repeat along N
          ck_tile::index_t ThreadPerBlock_M_, // num threads along M
          ck_tile::index_t ThreadPerBlock_N_, // num threads along N
          ck_tile::index_t Vector_N_,         // vector size along N
          bool kPadN_,
          bool kSaveX_,
          bool kThreePass_>
using trait_ = add_rmsnorm2d_rdquant_fwd_traits_<InputDataType_,
                                                 QuantizedDataType_,
                                                 Repeat_M_,
                                                 Repeat_N_,
                                                 ThreadPerBlock_M_,
                                                 ThreadPerBlock_N_,
                                                 Vector_N_,
                                                 kPadN_,
                                                 kSaveX_,
                                                 kThreePass_>;

template <typename input_data_type, typename quantized_data_type>
float add_rmsnorm2d_rdquant_fwd_b16_(add_rmsnorm2d_rdquant_fwd_traits t,
                                     add_rmsnorm2d_rdquant_fwd_args a,
                                     const ck_tile::stream_config& s)
{
    float r = -1;
    // clang-format off
    //                                                      rm  rn  tm   tn  vn   pd     x      3p
    if(a.n <= 64) {
            r = add_rmsnorm2d_rdquant_fwd_<trait_<input_data_type, quantized_data_type, 1,  1,  4,  64, 1,  true,  true, false>>(s, a);
    }
    else if(a.n <= 128) {
        if (a.n % 2 == 0)
            r = add_rmsnorm2d_rdquant_fwd_<trait_<input_data_type, quantized_data_type, 1,  1,  4,  64, 2,  true,  true, false>>(s, a);
        else
            r = add_rmsnorm2d_rdquant_fwd_<trait_<input_data_type, quantized_data_type, 1,  2,  4,  64, 1,  true,  true, false>>(s, a);
    }
    else if(a.n <= 256) {
        if (a.n % 4 == 0)
            r = add_rmsnorm2d_rdquant_fwd_<trait_<input_data_type, quantized_data_type,  1, 1,  4,  64, 4,  true,  true, false>>(s, a);
        else if (a.n % 2 == 0)
            r = add_rmsnorm2d_rdquant_fwd_<trait_<input_data_type, quantized_data_type,  1, 2,  4,  64, 2,  true,  true, false>>(s, a);
        else
            r = add_rmsnorm2d_rdquant_fwd_<trait_<input_data_type, quantized_data_type,  1, 4,  4,  64, 1,  true,  true, false>>(s, a);
    }
    else if(a.n <= 512) {
        if (a.n % 8 == 0)
            r = add_rmsnorm2d_rdquant_fwd_<trait_<input_data_type, quantized_data_type,  1, 1,  4,  64, 8,  true,  true, false>>(s, a);
        else if (a.n % 4 == 0)
            r = add_rmsnorm2d_rdquant_fwd_<trait_<input_data_type, quantized_data_type,  1, 2,  4,  64, 4,  true,  true, false>>(s, a);
        else if (a.n % 2 == 0)
            r = add_rmsnorm2d_rdquant_fwd_<trait_<input_data_type, quantized_data_type,  1, 4,  4,  64, 2,  true,  true, false>>(s, a);
        else
            r = add_rmsnorm2d_rdquant_fwd_<trait_<input_data_type, quantized_data_type,  1, 8,  4,  64, 1,  true,  true, false>>(s, a);
    }
    else if(a.n <= 768) {
        if (a.n % 4 == 0)
            r = add_rmsnorm2d_rdquant_fwd_<trait_<input_data_type, quantized_data_type,  1, 3,  4,  64, 4,  true,  true, false>>(s, a);
        else if (a.n % 2 == 0)
            r = add_rmsnorm2d_rdquant_fwd_<trait_<input_data_type, quantized_data_type,  1, 6,  4,  64, 2,  true,  true, false>>(s, a);
        else
            r = add_rmsnorm2d_rdquant_fwd_<trait_<input_data_type, quantized_data_type,  1,12,  4,  64, 1,  true,  true, false>>(s, a);
    }
    else if(a.n <= 1024) {
        if (a.n % 8 == 0)
            r = add_rmsnorm2d_rdquant_fwd_<trait_<input_data_type, quantized_data_type,  1, 1, 2,  128, 8,  true,  true, false>>(s, a);
        else if (a.n % 4 == 0)
            r = add_rmsnorm2d_rdquant_fwd_<trait_<input_data_type, quantized_data_type,  1, 2, 2,  128, 4,  true,  true, false>>(s, a);
        else if (a.n % 2 == 0)
            r = add_rmsnorm2d_rdquant_fwd_<trait_<input_data_type, quantized_data_type,  1, 4, 2,  128, 2,  true,  true, false>>(s, a);
        else
            r = add_rmsnorm2d_rdquant_fwd_<trait_<input_data_type, quantized_data_type,  1, 4, 1,  256, 1,  true,  true, false>>(s, a);
    }
    else if(a.n <= 1536) {
        if (a.n % 8 == 0)
            r = add_rmsnorm2d_rdquant_fwd_<trait_<input_data_type, quantized_data_type,  1, 3, 4,   64, 8,  true,  true, false>>(s, a);
        else if (a.n % 4 == 0)
            r = add_rmsnorm2d_rdquant_fwd_<trait_<input_data_type, quantized_data_type,  1, 3, 2,  128, 4,  true,  true, false>>(s, a);
        else if (a.n % 2 == 0)
            r = add_rmsnorm2d_rdquant_fwd_<trait_<input_data_type, quantized_data_type,  1, 3, 1,  256, 2,  true,  true, false>>(s, a);
        else
            r = add_rmsnorm2d_rdquant_fwd_<trait_<input_data_type, quantized_data_type,  1, 6, 1,  256, 1,  true,  true, false>>(s, a);
    }
    else if(a.n <= 2048) {
        if (a.n % 8 == 0)
            r = add_rmsnorm2d_rdquant_fwd_<trait_<input_data_type, quantized_data_type,  1, 1, 1,  256, 8,  true,  true, false>>(s, a);
        else if (a.n % 4 == 0)
            r = add_rmsnorm2d_rdquant_fwd_<trait_<input_data_type, quantized_data_type,  1, 2, 1,  256, 4,  true,  true, false>>(s, a);
        else if (a.n % 2 == 0)
            r = add_rmsnorm2d_rdquant_fwd_<trait_<input_data_type, quantized_data_type,  1, 4, 1,  256, 2,  true,  true, false>>(s, a);
        else
            r = add_rmsnorm2d_rdquant_fwd_<trait_<input_data_type, quantized_data_type,  1, 8, 1,  256, 1,  true,  true, false>>(s, a);
    }
    else if(a.n <= 3072) {
        if (a.n % 8 == 0)
            r = add_rmsnorm2d_rdquant_fwd_<trait_<input_data_type, quantized_data_type,  1, 3, 1,  128, 8,  true,  true, false>>(s, a);
        else if (a.n % 4 == 0)
            r = add_rmsnorm2d_rdquant_fwd_<trait_<input_data_type, quantized_data_type,  1, 3, 1,  256, 4,  true,  true, false>>(s, a);
        else if (a.n % 2 == 0)
            r = add_rmsnorm2d_rdquant_fwd_<trait_<input_data_type, quantized_data_type,  1, 6, 1,  256, 2,  true,  true, false>>(s, a);
        else
            r = add_rmsnorm2d_rdquant_fwd_<trait_<input_data_type, quantized_data_type,  1, 3, 1, 1024, 1,  true,  true, false>>(s, a);
    }
    else if(a.n <= 4096) {
        if (a.n % 8 == 0)
            r = add_rmsnorm2d_rdquant_fwd_<trait_<input_data_type, quantized_data_type,  1, 2, 1,  256, 8,  true,  true, false>>(s, a);
        else if (a.n % 4 == 0)
            r = add_rmsnorm2d_rdquant_fwd_<trait_<input_data_type, quantized_data_type,  1, 4, 1,  256, 4,  true,  true, false>>(s, a);
        else if (a.n % 2 == 0)
            r = add_rmsnorm2d_rdquant_fwd_<trait_<input_data_type, quantized_data_type,  1, 2, 1, 1024, 2,  true,  true, false>>(s, a);
        else
            r = add_rmsnorm2d_rdquant_fwd_<trait_<input_data_type, quantized_data_type,  1, 4, 1, 1024, 1,  true,  true, false>>(s, a);
    }
    else if(a.n <= 8192) {
        if(a.n<8192){
            if(t.save_x){
                if (a.n % 8 == 0)
                    r = add_rmsnorm2d_rdquant_fwd_<trait_<input_data_type, quantized_data_type,  1, 2, 1,  512, 8,  true,  true, false>>(s, a);
                else if (a.n % 4 == 0)
                    r = add_rmsnorm2d_rdquant_fwd_<trait_<input_data_type, quantized_data_type,  1, 4, 1,  512, 4,  true,  true, false>>(s, a);
                else if (a.n % 2 == 0)
                    r = add_rmsnorm2d_rdquant_fwd_<trait_<input_data_type, quantized_data_type,  1, 4, 1, 1024, 2,  true,  true, false>>(s, a);
                else
                    r = add_rmsnorm2d_rdquant_fwd_<trait_<input_data_type, quantized_data_type,  1, 8, 1, 1024, 1,  true,  true, false>>(s, a);
            }
            else{
                if (a.n % 8 == 0)
                    r = add_rmsnorm2d_rdquant_fwd_<trait_<input_data_type, quantized_data_type,  1, 2, 1,  512, 8,  true,  false, false>>(s, a);
                else if (a.n % 4 == 0)
                    r = add_rmsnorm2d_rdquant_fwd_<trait_<input_data_type, quantized_data_type,  1, 4, 1,  512, 4,  true,  false, false>>(s, a);
                else if (a.n % 2 == 0)
                    r = add_rmsnorm2d_rdquant_fwd_<trait_<input_data_type, quantized_data_type,  1, 4, 1, 1024, 2,  true,  false, false>>(s, a);
                else
                    r = add_rmsnorm2d_rdquant_fwd_<trait_<input_data_type, quantized_data_type,  1, 8, 1, 1024, 1,  true,  false, false>>(s, a);
            }
        }
        else{
            if(t.save_x){
                if (a.n % 8 == 0)
                    r = add_rmsnorm2d_rdquant_fwd_<trait_<input_data_type, quantized_data_type,  1, 2, 1,  512, 8,  false,  true, false>>(s, a);
                else if (a.n % 4 == 0)
                    r = add_rmsnorm2d_rdquant_fwd_<trait_<input_data_type, quantized_data_type,  1, 4, 1,  512, 4,  false,  true, false>>(s, a);
                else if (a.n % 2 == 0)
                    r = add_rmsnorm2d_rdquant_fwd_<trait_<input_data_type, quantized_data_type,  1, 4, 1, 1024, 2,  false,  true, false>>(s, a);
                else
                    r = add_rmsnorm2d_rdquant_fwd_<trait_<input_data_type, quantized_data_type,  1, 8, 1, 1024, 1,  false,  true, false>>(s, a);
            }
            else{
                if (a.n % 8 == 0)
                    r = add_rmsnorm2d_rdquant_fwd_<trait_<input_data_type, quantized_data_type,  1, 2, 1,  512, 8,  false,  false, false>>(s, a);
                else if (a.n % 4 == 0)
                    r = add_rmsnorm2d_rdquant_fwd_<trait_<input_data_type, quantized_data_type,  1, 4, 1,  512, 4,  false,  false, false>>(s, a);
                else if (a.n % 2 == 0)
                    r = add_rmsnorm2d_rdquant_fwd_<trait_<input_data_type, quantized_data_type,  1, 4, 1, 1024, 2,  false,  false, false>>(s, a);
                else
                    r = add_rmsnorm2d_rdquant_fwd_<trait_<input_data_type, quantized_data_type,  1, 8, 1, 1024, 1,  false,  false, false>>(s, a);
            }
        }
    }
    else if(a.n > 8192) {
        if (a.n % 8 == 0)
            r = add_rmsnorm2d_rdquant_fwd_<trait_<input_data_type, quantized_data_type,  1, 2, 1,  512, 8,  true,  true, true>>(s, a);
        else if (a.n % 4 == 0)
            r = add_rmsnorm2d_rdquant_fwd_<trait_<input_data_type, quantized_data_type,  1, 4, 1,  512, 4,  true,  true, true>>(s, a);
        else if (a.n % 2 == 0)
            r = add_rmsnorm2d_rdquant_fwd_<trait_<input_data_type, quantized_data_type,  1, 4, 1, 1024, 2,  true,  true, true>>(s, a);
        else
            r = add_rmsnorm2d_rdquant_fwd_<trait_<input_data_type, quantized_data_type,  1, 8, 1, 1024, 1,  true,  true, true>>(s, a);
    }
    return r;
    // clang-format on
}

float add_rmsnorm2d_rdquant_fwd(add_rmsnorm2d_rdquant_fwd_traits t,
                                add_rmsnorm2d_rdquant_fwd_args a,
                                const ck_tile::stream_config& s)
{
    if(t.input_data_type.compare("fp16") == 0 && t.quantized_data_type.compare("int8") == 0 &&
       t.save_x)
    {
        return add_rmsnorm2d_rdquant_fwd_b16_<ck_tile::fp16_t, ck_tile::int8_t>(t, a, s);
    }
    else if(t.input_data_type.compare("fp16") == 0 && t.quantized_data_type.compare("int8") == 0 &&
            !t.save_x)
    {
        return add_rmsnorm2d_rdquant_fwd_b16_<ck_tile::fp16_t, ck_tile::int8_t>(t, a, s);
    }
    else if(t.input_data_type.compare("bf16") == 0 && t.quantized_data_type.compare("int8") == 0 &&
            t.save_x)
    {
        return add_rmsnorm2d_rdquant_fwd_b16_<ck_tile::bf16_t, ck_tile::int8_t>(t, a, s);
    }
    else if(t.input_data_type.compare("bf16") == 0 && t.quantized_data_type.compare("int8") == 0 &&
            !t.save_x)
    {
        return add_rmsnorm2d_rdquant_fwd_b16_<ck_tile::bf16_t, ck_tile::int8_t>(t, a, s);
    }
    else if(t.input_data_type.compare("fp16") == 0 && t.quantized_data_type.compare("fp8") == 0 &&
            t.save_x)
    {
        return add_rmsnorm2d_rdquant_fwd_b16_<ck_tile::fp16_t, ck_tile::fp8_t>(t, a, s);
    }
    else if(t.input_data_type.compare("fp16") == 0 && t.quantized_data_type.compare("fp8") == 0 &&
            !t.save_x)
    {
        return add_rmsnorm2d_rdquant_fwd_b16_<ck_tile::fp16_t, ck_tile::fp8_t>(t, a, s);
    }
    else if(t.input_data_type.compare("bf16") == 0 && t.quantized_data_type.compare("fp8") == 0 &&
            t.save_x)
    {
        return add_rmsnorm2d_rdquant_fwd_b16_<ck_tile::bf16_t, ck_tile::fp8_t>(t, a, s);
    }
    else if(t.input_data_type.compare("bf16") == 0 && t.quantized_data_type.compare("fp8") == 0 &&
            !t.save_x)
    {
        return add_rmsnorm2d_rdquant_fwd_b16_<ck_tile::bf16_t, ck_tile::fp8_t>(t, a, s);
    }
    else
        throw std::runtime_error("Without supported instances!");
}
