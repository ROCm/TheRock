/* ************************************************************************
 * Copyright (C) 2018-2025 Advanced Micro Devices, Inc. All rights reserved.
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
 * ************************************************************************ */

#ifndef _HIPBLAS_TYPE_DISPATCH_
#define _HIPBLAS_TYPE_DISPATCH_
#include "hipblas.hpp"
#include "utility.h"

// ----------------------------------------------------------------------------
// Calls TEST template based on the argument types. TEST<> is expected to
// return a functor which takes a const Arguments& argument. If the types do
// not match a recognized type combination, then TEST<void> is called.  This
// function returns the same type as TEST<...>{}(arg), usually bool or void.
// ----------------------------------------------------------------------------

// Simple functions which take only one datatype
//
// Even if the function can take mixed datatypes, this function can handle the
// cases where the types are uniform, in which case one template type argument
// is passed to TEST, and the rest are assumed to match the first.
template <template <typename...> class TEST>
auto hipblas_simple_dispatch(const Arguments& arg)
{
    switch(arg.a_type)
    {
    case HIP_R_16F:
        return TEST<hipblasHalf>{}(arg);
    case HIP_R_16BF:
        return TEST<hipblasBfloat16>{}(arg);
    case HIP_R_32F:
        return TEST<float>{}(arg);
    case HIP_R_64F:
        return TEST<double>{}(arg);
    case HIP_C_32F:
        return TEST<std::complex<float>>{}(arg);
    case HIP_C_64F:
        return TEST<std::complex<double>>{}(arg);
    default:
        return TEST<void>{}(arg);
    }
}

// BLAS1 functions
template <template <typename...> class TEST>
auto hipblas_blas1_dispatch(const Arguments& arg)
{
    const auto Ti = arg.a_type, Tb = arg.b_type, To = arg.d_type;
    if(Ti == To)
    {
        if(Tb == Ti)
            return hipblas_simple_dispatch<TEST>(arg);
        else
        { // for csscal and zdscal and complex rotg only
            if(Ti == HIP_C_32F && Tb == HIP_R_32F)
                return TEST<std::complex<float>, float>{}(arg);
            else if(Ti == HIP_C_64F && Tb == HIP_R_64F)
                return TEST<std::complex<double>, double>{}(arg);
        }
    }
    else if(Ti == HIP_C_32F && Tb == HIP_R_32F)
        return TEST<std::complex<float>, float>{}(arg);
    else if(Ti == HIP_C_64F && Tb == HIP_R_64F)
        return TEST<std::complex<double>, double>{}(arg);
    else if(Ti == HIP_R_32F && Tb == HIP_R_32F)
        return TEST<float, float>{}(arg);
    else if(Ti == HIP_R_64F && Tb == HIP_R_64F)
        return TEST<double, double>{}(arg);
    //  else if(Ti == hipblas_datatype_f16_c && To == HIP_R_16F)
    //      return TEST<hipblas_half_complex, hipblasHalf>{}(arg);

    return TEST<void>{}(arg);
}

// BLAS1_ex functions
// TODO: Update this when adding these functions to hipblas-bench
template <template <typename...> class TEST>
auto hipblas_blas1_ex_dispatch(const Arguments& arg)
{
    const auto        Ta = arg.a_type, Tx = arg.b_type, Ty = arg.c_type, Tex = arg.compute_type;
    const std::string function = arg.function;
    const bool        is_axpy  = function == "axpy_ex" || function == "axpy_batched_ex"
                         || function == "axpy_strided_batched_ex";
    const bool is_dot = function == "dot_ex" || function == "dot_batched_ex"
                        || function == "dot_strided_batched_ex" || function == "dotc_ex"
                        || function == "dotc_batched_ex" || function == "dotc_strided_batched_ex";
    const bool is_nrm2 = function == "nrm2_ex" || function == "nrm2_batched_ex"
                         || function == "nrm2_strided_batched_ex";
    const bool is_rot = function == "rot_ex" || function == "rot_batched_ex"
                        || function == "rot_strided_batched_ex";
    const bool is_scal = function == "scal_ex" || function == "scal_batched_ex"
                         || function == "scal_strided_batched_ex";

    if(Ta == Tx && Tx == Ty && Ty == Tex)
    {
        return hipblas_simple_dispatch<TEST>(arg); // Ta == Tx == Ty == Tex
    }
    else if(is_scal && Ta == Tx && Tx == Tex)
    {
        // hscal with f16_r compute (scal doesn't care about Ty)
        return hipblas_simple_dispatch<TEST>(arg);
    }
    else if((is_rot || is_dot || is_axpy) && Ta == Tx && Tx == Ty && Ta == HIP_R_16F
            && Tex == HIP_R_32F)
    {
        return TEST<hipblasHalf, hipblasHalf, hipblasHalf, float>{}(arg);
    }
    else if((is_rot || is_dot || is_axpy) && Ta == Tx && Tx == Ty && Ta == HIP_R_16BF
            && Tex == HIP_R_32F)
    {
        return TEST<hipblasBfloat16, hipblasBfloat16, hipblasBfloat16, float>{}(arg);
    }
    else if(is_axpy && Ta == Tex && Tx == Ty && Tx == HIP_R_16F && Tex == HIP_R_32F)
    {
        return TEST<float, hipblasHalf, hipblasHalf, float>{}(arg);
    }
    else if((is_scal || is_nrm2) && Ta == Tx && Ta == HIP_R_16F && Tex == HIP_R_32F)
    {
        // half scal, nrm2, axpy
        return TEST<hipblasHalf, hipblasHalf, float>{}(arg);
    }
    else if((is_scal || is_nrm2) && Ta == Tx && Ta == HIP_R_16BF && Tex == HIP_R_32F)
    {
        // bfloat16 scal, nrm2
        return TEST<hipblasBfloat16, hipblasBfloat16, float>{}(arg);
    }
    else if(is_axpy && Ta == Tex && Tx == Ty && (Tx == HIP_R_16BF || Tx == HIP_R_16F)
            && Tex == HIP_R_32F)
    {
        // axpy bfloat16 with float alpha
        return TEST<float, hipblasBfloat16, hipblasBfloat16, float>{}(arg);
    }
    // exclusive functions cases
    else if(is_scal)
    {
        // scal_ex ordering: <alphaType, dataType, exType> opposite order of scal test
        if(Ta == Tex && Tx == HIP_R_16BF && Tex == HIP_R_32F)
        {
            // scal bfloat16 with float alpha
            return TEST<float, hipblasBfloat16, float>{}(arg);
        }
        else if(Ta == HIP_R_32F && Tx == HIP_R_16F && Tex == HIP_R_32F)
        {
            // scal half with float alpha
            return TEST<float, hipblasHalf, float>{}(arg);
        }
        else if(Ta == HIP_R_32F && Tx == HIP_C_32F && Tex == HIP_C_32F)
        {
            // csscal-like
            return TEST<float, std::complex<float>, std::complex<float>>{}(arg);
        }
        else if(Ta == HIP_R_64F && Tx == HIP_C_64F && Tex == HIP_C_64F)
        {
            // zdscal-like
            return TEST<double, std::complex<double>, std::complex<double>>{}(arg);
        }
    }
    else if(is_nrm2)
    {
        if(Ta == HIP_C_32F && Tx == HIP_R_32F && Tex == HIP_R_32F)
        {
            // scnrm2
            return TEST<std::complex<float>, float, float>{}(arg);
        }
        else if(Ta == HIP_C_64F && Tx == HIP_R_64F && Tex == HIP_R_64F)
        {
            // dznrm2
            return TEST<std::complex<double>, double, double>{}(arg);
        }
    }
    else if(is_rot)
    {
        if(Ta == HIP_C_32F && Tx == HIP_C_32F && Ty == HIP_R_32F && Tex == HIP_C_32F)
        {
            // rot with complex x/y/compute and real cs
            return TEST<std::complex<float>, std::complex<float>, float, std::complex<float>>{}(
                arg);
        }
        else if(Ta == HIP_C_64F && Tx == HIP_C_64F && Ty == HIP_R_64F && Tex == HIP_C_64F)
        {
            // rot with complex x/y/compute and real cs
            return TEST<std::complex<double>, std::complex<double>, double, std::complex<double>>{}(
                arg);
        }
    }

    return TEST<void>{}(arg);
}

// rot
// giving rot it's own dispatch function so the code is easier to follow
template <template <typename...> class TEST>
auto hipblas_rot_dispatch(const Arguments& arg)
{
    const auto Ta = arg.a_type, Tb = arg.b_type, Tc = arg.c_type;
    if(Ta == Tb && Tb == Tc)
    {
        // srot, drot
        return hipblas_simple_dispatch<TEST>(arg);
    }
    else if(Ta == HIP_C_32F && Tb == HIP_R_32F && Tc == Tb)
    {
        // csrot
        return TEST<std::complex<float>, float, float>{}(arg);
    }
    else if(Ta == HIP_C_64F && Tb == HIP_R_64F && Tc == Tb)
    {
        // zdrot
        return TEST<std::complex<double>, double, double>{}(arg);
    }
    else if(Ta == HIP_C_32F && Tb == HIP_R_32F && Tc == Ta)
    {
        // crot
        return TEST<std::complex<float>, float, std::complex<float>>{}(arg);
    }
    else if(Ta == HIP_C_64F && Tb == HIP_R_64F && Tc == Ta)
    {
        // zrot
        return TEST<std::complex<double>, double, std::complex<double>>{}(arg);
    }

    return TEST<void>{}(arg);
}

// gemm functions
template <template <typename...> class TEST>
auto hipblas_gemm_dispatch(const Arguments& arg)
{
    const auto Ti = arg.a_type, To = arg.c_type, Tc = arg.compute_type;

    if(arg.b_type == Ti && arg.d_type == To)
    {
        if(Ti != To)
        {
            if(Ti == HIP_R_8I && To == HIP_R_32I && Tc == To)
                return TEST<int8_t, int32_t, int32_t>{}(arg);
        }
        else if(Tc != To)
        {
            if(To == HIP_R_16F && Tc == HIP_R_32F)
            {
                return TEST<hipblasHalf, hipblasHalf, float>{}(arg);
            }
            else if(To == HIP_R_16BF && Tc == HIP_R_32F)
            {
                return TEST<hipblasBfloat16, hipblasBfloat16, float>{}(arg);
            }
        }
        else
        {
            return hipblas_simple_dispatch<TEST>(arg); // Ti = To = Tc
        }
    }
    return TEST<void>{}(arg);
}

// syrk_ex
template <template <typename...> class TEST>
auto hipblas_syrk_ex_dispatch(const Arguments& arg)
{
    const auto Ti = arg.a_type, To = arg.c_type, Tex = arg.compute_type;

    if(Ti != Tex && (Ti == HIP_R_16F || Ti == HIP_R_16BF) && (To == Ti || To == HIP_R_32F))
    {
        if(Ti == HIP_R_16F && To == Ti && Tex == HIP_R_32F)
        {
            return TEST<hipblasHalf, hipblasHalf, float>{}(arg);
        }
        else if(Ti == HIP_R_16F && To == Tex && Tex == HIP_R_32F)
        {
            return TEST<hipblasHalf, float, float>{}(arg);
        }
        else if(Ti == HIP_R_16BF && To == Ti && Tex == HIP_R_32F)
        {
            return TEST<hipblasBfloat16, hipblasBfloat16, float>{}(arg);
        }
        else if(Ti == HIP_R_16BF && To == Tex && Tex == HIP_R_32F)
        {
            return TEST<hipblasBfloat16, float, float>{}(arg);
        }
    }
    else if(Ti != Tex && (Ti == HIP_R_32F && Tex == HIP_R_64F))
    {
        if(To == HIP_R_32F && To == Ti && Tex == HIP_R_64F)
        {
            return TEST<float, float, double>{}(arg);
        }
        else if(To == Tex && Tex == HIP_R_64F)
        {
            return TEST<float, double, double>{}(arg);
        }
    }

    return TEST<void>{}(arg);
}

// herk_ex
template <template <typename...> class TEST>
auto hipblas_herk_ex_dispatch(const Arguments& arg)
{
    const auto Ti = arg.a_type, To = arg.c_type, Tex = arg.compute_type;

    if(Ti != Tex && (Ti == HIP_C_32F && Tex == HIP_C_64F))
    {
        if(To == HIP_C_32F && To == Ti && Tex == HIP_C_64F)
        {
            return TEST<std::complex<float>, std::complex<float>, std::complex<double>>{}(arg);
        }
        else if(To == Tex)
        {
            return TEST<std::complex<float>, std::complex<double>, std::complex<double>>{}(arg);
        }
    }

    return TEST<void>{}(arg);
}

#endif
