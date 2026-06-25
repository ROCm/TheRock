/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc.
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
#include <algorithm>
#include <cstdlib>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hipblaslt/hipblaslt.h>
#include <numeric>
#include <tuple>
#include <utility>
#include <vector>

namespace
{
    struct MatrixTransformIO
    {
        MatrixTransformIO()                 = default;
        virtual ~MatrixTransformIO()        = default;
        virtual void*  getBuf(size_t i)     = 0;
        virtual size_t elemNumBytes() const = 0;

    private:
        MatrixTransformIO(const MatrixTransformIO&)            = delete;
        MatrixTransformIO(MatrixTransformIO&&)                 = delete;
        MatrixTransformIO& operator=(const MatrixTransformIO&) = delete;
        MatrixTransformIO& operator=(MatrixTransformIO&&)      = delete;
    };

    template <typename DType>
    struct TypedMatrixTransformIO : public MatrixTransformIO
    {
        TypedMatrixTransformIO(int64_t m, int64_t n, int64_t b)
        {
            constexpr std::size_t alignment = 2 * 1024 * 1024;
            const auto            bufSize   = m * n * b * sizeof(DType);
            const auto            res       = bufSize % alignment;
            const auto            allocSize = bufSize + (res ? (alignment - res) : 0);
            auto                  err       = hipMalloc(&this->aBase, allocSize);
            EXPECT_EQ(err, hipSuccess);
            err = hipMalloc(&this->bBase, allocSize);
            EXPECT_EQ(err, hipSuccess);
            err = hipMalloc(&this->cBase, allocSize);
            EXPECT_EQ(err, hipSuccess);
            this->a = reinterpret_cast<DType*>(aBase + (allocSize - bufSize));
            this->b = reinterpret_cast<DType*>(bBase + (allocSize - bufSize));
            this->c = reinterpret_cast<DType*>(cBase + (allocSize - bufSize));
            init(this->a, m * n * b);
            init(this->b, m * n * b);
        }

        ~TypedMatrixTransformIO() override
        {
            auto err = hipFree(aBase);
            err      = hipFree(bBase);
            err      = hipFree(cBase);
            aBase    = nullptr;
            bBase    = nullptr;
            cBase    = nullptr;
            EXPECT_EQ(err, hipSuccess);
        }

        void* getBuf(size_t i) override
        {
            void* buf[] = {a, b, c};
            return buf[i];
        }

        size_t elemNumBytes() const override
        {
            return sizeof(DType);
        }

    private:
        void init(DType* buf, size_t len)
        {
            srand(time(nullptr));
            std::vector<DType> ref(len);

            for(auto& i : ref)
            {
                i = DType(rand() % 7 - 3);
            }

            auto err = hipMemcpy(buf, ref.data(), len * sizeof(DType), hipMemcpyHostToDevice);
            ASSERT_EQ(err, hipSuccess);
        }

    private:
        DType* a{};
        DType* b{};
        DType* c{};
        char*  aBase{};
        char*  bBase{};
        char*  cBase{};
    };

    using MatrixTransformIOPtr = std::unique_ptr<MatrixTransformIO>;
    MatrixTransformIOPtr
        makeMatrixTransformIOPtr(hipDataType datatype, int64_t m, int64_t n, int64_t b)
    {
        if(datatype == HIP_R_32F)
        {
            return std::make_unique<TypedMatrixTransformIO<hipblasLtFloat>>(m, n, b);
        }
        else if(datatype == HIP_R_16F)
        {
            return std::make_unique<TypedMatrixTransformIO<hipblasLtHalf>>(m, n, b);
        }
        else if(datatype == HIP_R_16BF)
        {
            return std::make_unique<TypedMatrixTransformIO<hipblasLtBfloat16>>(m, n, b);
        }
        else if(datatype == HIP_R_8I)
        {
            return std::make_unique<TypedMatrixTransformIO<int8_t>>(m, n, b);
        }
        else if(datatype == HIP_R_32I)
        {
            return std::make_unique<TypedMatrixTransformIO<int32_t>>(m, n, b);
        }
        return nullptr;
    }

    template <bool RowMaj>
    int64_t getLeadingDimSize(int64_t numRows, int64_t numCols)
    {
        return RowMaj ? numCols : numRows;
    }

    template <bool RowMaj>
    uint32_t getOffset(uint32_t row, uint32_t col, uint32_t ld)
    {
        if constexpr(RowMaj)
        {
            return ld * row + col;
        }
        else
        {
            return ld * col + row;
        }
    }

    template <typename DType, typename ScaleType, bool RowMajA, bool RowMajB, bool RowMajC>
    void cpuTransform(DType*       c,
                      const DType* a,
                      const DType* b,
                      ScaleType    alpha,
                      ScaleType    beta,
                      bool         transA,
                      bool         transB,
                      uint32_t     m,
                      uint32_t     n,
                      uint32_t     ldA,
                      uint32_t     ldB,
                      uint32_t     ldC,
                      uint32_t     batchSize,
                      uint32_t     batchStride)
    {
        for(uint32_t k = 0; k < batchSize; ++k)
        {
            const int64_t batchOffset = k * int64_t(batchStride);

            for(uint32_t i = 0; i < m; ++i)
            {
                for(uint32_t j = 0; j < n; ++j)
                {
                    const auto offsetA
                        = transA ? getOffset<RowMajA>(j, i, ldA) : getOffset<RowMajA>(i, j, ldA);
                    const auto offsetB
                        = transB ? getOffset<RowMajB>(j, i, ldB) : getOffset<RowMajB>(i, j, ldB);
                    const auto offsetC = getOffset<RowMajC>(i, j, ldC);
                    c[batchOffset + offsetC]
                        = a[batchOffset + offsetA] * alpha + b[batchOffset + offsetB] * beta;
                }
            }
        }
    }

    template <typename DType>
    void validation(void*    c,
                    void*    a,
                    void*    b,
                    float    alpha,
                    float    beta,
                    uint32_t m,
                    uint32_t n,
                    uint32_t ldA,
                    uint32_t ldB,
                    uint32_t ldC,
                    uint32_t batchSize,
                    uint32_t batchStride,
                    bool     rowMajA,
                    bool     rowMajB,
                    bool     rowMajC,
                    bool     transA,
                    bool     transB)
    {
        using std::begin;
        using std::end;
        std::vector<float> hC(m * n * batchSize, 0);
        std::vector<float> hA(m * n * batchSize, 0);
        std::vector<float> hB(m * n * batchSize, 0);
        std::vector<float> cpuRef(m * n * batchSize, 0);
        std::vector<DType> dA(m * n * batchSize);
        std::vector<DType> dB(m * n * batchSize);
        std::vector<DType> dC(m * n * batchSize);
        auto               err = hipSuccess;

        if(a)
        {
            err = hipMemcpyDtoH(dA.data(), a, m * n * batchSize * sizeof(DType));
        }

        if(b)
        {
            err = hipMemcpyDtoH(dB.data(), b, m * n * batchSize * sizeof(DType));
        }

        err = hipMemcpyDtoH(dC.data(), c, m * n * batchSize * sizeof(DType));

        ASSERT_EQ(err, hipSuccess);

        std::transform(begin(dC), end(dC), begin(hC), [](auto i) { return float(i); });

        if(a)
        {
            std::transform(begin(dA), end(dA), begin(hA), [](auto i) { return float(i); });
        }

        if(b)
        {
            std::transform(begin(dB), end(dB), begin(hB), [](auto i) { return float(i); });
        }

        if(rowMajA && rowMajB && rowMajC)
        {
            cpuTransform<float, float, true, true, true>(cpuRef.data(),
                                                         hA.data(),
                                                         hB.data(),
                                                         alpha,
                                                         beta,
                                                         transA,
                                                         transB,
                                                         m,
                                                         n,
                                                         ldA,
                                                         ldB,
                                                         ldC,
                                                         batchSize,
                                                         batchStride);
        }
        else if(!rowMajA && rowMajB && rowMajC)
        {
            cpuTransform<float, float, false, true, true>(cpuRef.data(),
                                                          hA.data(),
                                                          hB.data(),
                                                          alpha,
                                                          beta,
                                                          transA,
                                                          transB,
                                                          m,
                                                          n,
                                                          ldA,
                                                          ldB,
                                                          ldC,
                                                          batchSize,
                                                          batchStride);
        }
        else if(rowMajA && !rowMajB && rowMajC)
        {
            cpuTransform<float, float, true, false, true>(cpuRef.data(),
                                                          hA.data(),
                                                          hB.data(),
                                                          alpha,
                                                          beta,
                                                          transA,
                                                          transB,
                                                          m,
                                                          n,
                                                          ldA,
                                                          ldB,
                                                          ldC,
                                                          batchSize,
                                                          batchStride);
        }
        else if(rowMajA && rowMajB && !rowMajC)
        {
            cpuTransform<float, float, true, true, false>(cpuRef.data(),
                                                          hA.data(),
                                                          hB.data(),
                                                          alpha,
                                                          beta,
                                                          transA,
                                                          transB,
                                                          m,
                                                          n,
                                                          ldA,
                                                          ldB,
                                                          ldC,
                                                          batchSize,
                                                          batchStride);
        }
        else if(!rowMajA && !rowMajB && rowMajC)
        {
            cpuTransform<float, float, false, false, true>(cpuRef.data(),
                                                           hA.data(),
                                                           hB.data(),
                                                           alpha,
                                                           beta,
                                                           transA,
                                                           transB,
                                                           m,
                                                           n,
                                                           ldA,
                                                           ldB,
                                                           ldC,
                                                           batchSize,
                                                           batchStride);
        }
        else if(!rowMajA && rowMajB && !rowMajC)
        {
            cpuTransform<float, float, false, true, false>(cpuRef.data(),
                                                           hA.data(),
                                                           hB.data(),
                                                           alpha,
                                                           beta,
                                                           transA,
                                                           transB,
                                                           m,
                                                           n,
                                                           ldA,
                                                           ldB,
                                                           ldC,
                                                           batchSize,
                                                           batchStride);
        }
        else if(rowMajA && !rowMajB && !rowMajC)
        {
            cpuTransform<float, float, true, false, false>(cpuRef.data(),
                                                           hA.data(),
                                                           hB.data(),
                                                           alpha,
                                                           beta,
                                                           transA,
                                                           transB,
                                                           m,
                                                           n,
                                                           ldA,
                                                           ldB,
                                                           ldC,
                                                           batchSize,
                                                           batchStride);
        }
        else if(!rowMajA && !rowMajB && !rowMajC)
        {
            cpuTransform<float, float, false, false, false>(cpuRef.data(),
                                                            hA.data(),
                                                            hB.data(),
                                                            alpha,
                                                            beta,
                                                            transA,
                                                            transB,
                                                            m,
                                                            n,
                                                            ldA,
                                                            ldB,
                                                            ldC,
                                                            batchSize,
                                                            batchStride);
        }

        ASSERT_THAT(hC, ::testing::Pointwise(::testing::FloatNear(1e-5), cpuRef));
    }
}

class MatrixTransformTest : public ::testing::TestWithParam<std::tuple<int64_t,
                                                                       int64_t,
                                                                       hipDataType,
                                                                       hipDataType,
                                                                       hipblasOperation_t,
                                                                       hipblasOperation_t,
                                                                       hipblasLtOrder_t,
                                                                       hipblasLtOrder_t,
                                                                       hipblasLtOrder_t>>
{
};

TEST_P(MatrixTransformTest, Basic)
{
    int64_t                     m             = std::get<0>(GetParam());
    int64_t                     n             = std::get<1>(GetParam());
    int32_t                     batchSize     = 1;
    auto                        datatype      = std::get<2>(GetParam());
    auto                        scaleDatatype = std::get<3>(GetParam());
    auto                        opA           = std::get<4>(GetParam());
    auto                        opB           = std::get<5>(GetParam());
    auto                        orderA        = std::get<6>(GetParam());
    auto                        orderB        = std::get<7>(GetParam());
    auto                        orderC        = std::get<8>(GetParam());
    float                       alpha         = 1;
    float                       beta          = 1;
    int64_t                     batchStride   = m * n;
    std::pair<int64_t, int64_t> shapeA;
    std::pair<int64_t, int64_t> shapeB;
    shapeA.first  = opA == HIPBLAS_OP_N ? m : n;
    shapeA.second = opA == HIPBLAS_OP_N ? n : m;
    shapeB.first  = opB == HIPBLAS_OP_N ? m : n;
    shapeB.second = opB == HIPBLAS_OP_N ? n : m;
    uint32_t ldA  = (orderA == HIPBLASLT_ORDER_ROW)
                        ? getLeadingDimSize<true>(shapeA.first, shapeA.second)
                        : getLeadingDimSize<false>(shapeA.first, shapeA.second);
    uint32_t ldB  = (orderB == HIPBLASLT_ORDER_ROW)
                        ? getLeadingDimSize<true>(shapeB.first, shapeB.second)
                        : getLeadingDimSize<false>(shapeB.first, shapeB.second);
    uint32_t ldC  = (orderC == HIPBLASLT_ORDER_ROW) ? getLeadingDimSize<true>(m, n)
                                                    : getLeadingDimSize<false>(m, n);

    auto  inputs = makeMatrixTransformIOPtr(datatype, m, n, batchSize);
    void* dA     = inputs->getBuf(0);
    void* dB     = inputs->getBuf(1);
    void* dC     = inputs->getBuf(2);

    hipblasLtMatrixTransformDesc_t desc;
    auto                   hipblasLtErr = hipblasLtMatrixTransformDescCreate(&desc, scaleDatatype);
    hipblasLtPointerMode_t pMode        = HIPBLASLT_POINTER_MODE_HOST;
    hipblasLtErr                        = hipblasLtMatrixTransformDescSetAttribute(
        desc,
        hipblasLtMatrixTransformDescAttributes_t::HIPBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE,
        &pMode,
        sizeof(pMode));

    ASSERT_EQ(hipblasLtErr, HIPBLAS_STATUS_SUCCESS);

    hipblasLtErr = hipblasLtMatrixTransformDescSetAttribute(
        desc, HIPBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &opA, sizeof(opA));
    hipblasLtErr = hipblasLtMatrixTransformDescSetAttribute(
        desc, HIPBLASLT_MATRIX_TRANSFORM_DESC_TRANSB, &opB, sizeof(opB));
    hipblasLtMatrixLayout_t layoutA, layoutB, layoutC;
    hipblasLtErr
        = hipblasLtMatrixLayoutCreate(&layoutA, datatype, shapeA.first, shapeA.second, ldA);
    hipblasLtErr
        = hipblasLtMatrixLayoutCreate(&layoutB, datatype, shapeB.first, shapeB.second, ldB);
    hipblasLtErr = hipblasLtMatrixLayoutCreate(&layoutC, datatype, m, n, ldC);
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutA,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_ORDER,
        &orderA,
        sizeof(orderA));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutB,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_ORDER,
        &orderB,
        sizeof(orderB));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutC,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_ORDER,
        &orderC,
        sizeof(orderC));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutA,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batchSize,
        sizeof(batchSize));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutB,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batchSize,
        sizeof(batchSize));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutC,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batchSize,
        sizeof(batchSize));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutA,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &batchStride,
        sizeof(batchStride));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutB,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &batchStride,
        sizeof(batchStride));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutC,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &batchStride,
        sizeof(batchStride));
    hipblasLtHandle_t handle{};
    hipblasLtErr = hipblasLtCreate(&handle);
    hipblasLtErr = hipblasLtMatrixTransform(
        handle, desc, &alpha, dA, layoutA, &beta, dB, layoutB, dC, layoutC, nullptr);
    ASSERT_EQ(hipblasLtErr, HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    auto rowMajA = (orderA == HIPBLASLT_ORDER_ROW);
    auto rowMajB = (orderB == HIPBLASLT_ORDER_ROW);
    auto rowMajC = (orderC == HIPBLASLT_ORDER_ROW);
    auto transA  = (opA != HIPBLAS_OP_N);
    auto transB  = (opB != HIPBLAS_OP_N);

    if(datatype == HIP_R_32F)
    {
        validation<float>(dC,
                          dA,
                          dB,
                          alpha,
                          beta,
                          m,
                          n,
                          ldA,
                          ldB,
                          ldC,
                          batchSize,
                          batchStride,
                          rowMajA,
                          rowMajB,
                          rowMajC,
                          transA,
                          transB);
    }
    else if(datatype == HIP_R_16F)
    {
        validation<hipblasLtHalf>(dC,
                                  dA,
                                  dB,
                                  alpha,
                                  beta,
                                  m,
                                  n,
                                  ldA,
                                  ldB,
                                  ldC,
                                  batchSize,
                                  batchStride,
                                  rowMajA,
                                  rowMajB,
                                  rowMajC,
                                  transA,
                                  transB);
    }
    else if(datatype == HIP_R_16BF)
    {
        validation<hipblasLtBfloat16>(dC,
                                      dA,
                                      dB,
                                      alpha,
                                      beta,
                                      m,
                                      n,
                                      ldA,
                                      ldB,
                                      ldC,
                                      batchSize,
                                      batchStride,
                                      rowMajA,
                                      rowMajB,
                                      rowMajC,
                                      transA,
                                      transB);
    }
    else if(datatype == HIP_R_8I)
    {
        validation<int8_t>(dC,
                           dA,
                           dB,
                           alpha,
                           beta,
                           m,
                           n,
                           ldA,
                           ldB,
                           ldC,
                           batchSize,
                           batchStride,
                           rowMajA,
                           rowMajB,
                           rowMajC,
                           transA,
                           transB);
    }
    else if(datatype == HIP_R_32I)
    {
        validation<int32_t>(dC,
                            dA,
                            dB,
                            alpha,
                            beta,
                            m,
                            n,
                            ldA,
                            ldB,
                            ldC,
                            batchSize,
                            batchStride,
                            rowMajA,
                            rowMajB,
                            rowMajC,
                            transA,
                            transB);
    }

    hipblasLtErr = hipblasLtMatrixTransformDescDestroy(desc);
    hipblasLtErr = hipblasLtDestroy(handle);
    hipblasLtErr = hipblasLtMatrixLayoutDestroy(layoutA);
    hipblasLtErr = hipblasLtMatrixLayoutDestroy(layoutB);
    hipblasLtErr = hipblasLtMatrixLayoutDestroy(layoutC);
}

TEST(MatrixTransformTest, InvalidConfigurations)
{
    int64_t                     m             = 1024;
    int64_t                     n             = 1024;
    int32_t                     batchSize     = 1;
    auto                        datatype      = HIP_R_32F;
    auto                        scaleDatatype = HIP_R_32F;
    auto                        opA           = HIPBLAS_OP_N;
    auto                        opB           = HIPBLAS_OP_N;
    auto                        orderA        = HIPBLASLT_ORDER_ROW;
    auto                        orderB        = HIPBLASLT_ORDER_ROW;
    auto                        orderC        = HIPBLASLT_ORDER_COL;
    float                       alpha         = 1;
    float                       beta          = 1;
    int64_t                     batchStride   = m * n;
    std::pair<int64_t, int64_t> shapeA;
    std::pair<int64_t, int64_t> shapeB;
    shapeA.first  = opA == HIPBLAS_OP_N ? m : n;
    shapeA.second = opA == HIPBLAS_OP_N ? n : m;
    shapeB.first  = opB == HIPBLAS_OP_N ? m : n;
    shapeB.second = opB == HIPBLAS_OP_N ? n : m;
    uint32_t ldA  = (orderA == HIPBLASLT_ORDER_ROW)
                        ? getLeadingDimSize<true>(shapeA.first, shapeA.second)
                        : getLeadingDimSize<false>(shapeA.first, shapeA.second);
    uint32_t ldB  = (orderB == HIPBLASLT_ORDER_ROW)
                        ? getLeadingDimSize<true>(shapeB.first, shapeB.second)
                        : getLeadingDimSize<false>(shapeB.first, shapeB.second);
    uint32_t ldC  = (orderC == HIPBLASLT_ORDER_ROW) ? getLeadingDimSize<true>(m, n)
                                                    : getLeadingDimSize<false>(m, n);

    auto  inputs = makeMatrixTransformIOPtr(datatype, m, n, batchSize);
    void* dA     = inputs->getBuf(0);
    void* dB     = inputs->getBuf(1);
    void* dC     = inputs->getBuf(2);

    hipblasLtMatrixTransformDesc_t desc;
    auto                   hipblasLtErr = hipblasLtMatrixTransformDescCreate(&desc, scaleDatatype);
    hipblasLtPointerMode_t pMode        = HIPBLASLT_POINTER_MODE_HOST;
    hipblasLtErr                        = hipblasLtMatrixTransformDescSetAttribute(
        desc,
        hipblasLtMatrixTransformDescAttributes_t::HIPBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE,
        &pMode,
        sizeof(pMode));

    ASSERT_EQ(hipblasLtErr, HIPBLAS_STATUS_SUCCESS);

    hipblasLtErr = hipblasLtMatrixTransformDescSetAttribute(
        desc, HIPBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &opA, sizeof(opA));
    hipblasLtErr = hipblasLtMatrixTransformDescSetAttribute(
        desc, HIPBLASLT_MATRIX_TRANSFORM_DESC_TRANSB, &opB, sizeof(opB));
    hipblasLtMatrixLayout_t layoutA, layoutB, layoutC;
    hipblasLtErr
        = hipblasLtMatrixLayoutCreate(&layoutA, datatype, shapeA.first, shapeA.second, ldA);
    hipblasLtErr
        = hipblasLtMatrixLayoutCreate(&layoutB, datatype, shapeB.first, shapeB.second, ldB);
    hipblasLtErr = hipblasLtMatrixLayoutCreate(&layoutC, datatype, m, n, ldC);
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutA,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_ORDER,
        &orderA,
        sizeof(orderA));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutB,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_ORDER,
        &orderB,
        sizeof(orderB));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutC,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_ORDER,
        &orderC,
        sizeof(orderC));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutA,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batchSize,
        sizeof(batchSize));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutB,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batchSize,
        sizeof(batchSize));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutC,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batchSize,
        sizeof(batchSize));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutA,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &batchStride,
        sizeof(batchStride));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutB,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &batchStride,
        sizeof(batchStride));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutC,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &batchStride,
        sizeof(batchStride));
    hipblasLtHandle_t handle{};

    hipblasLtErr = hipblasLtMatrixTransform(
        handle, desc, &alpha, dA, layoutA, &beta, dB, layoutB, dC, layoutC, nullptr);
    ASSERT_EQ(hipblasLtErr, HIPBLAS_STATUS_NOT_INITIALIZED);

    hipblasLtErr = hipblasLtCreate(&handle);

    hipblasLtErr = hipblasLtMatrixTransform(
        handle, desc, &alpha, dA, layoutA, &beta, dB, layoutB, dC, nullptr, nullptr);
    ASSERT_EQ(hipblasLtErr, HIPBLAS_STATUS_INVALID_VALUE);

    hipblasLtErr = hipblasLtMatrixTransform(
        handle, desc, &alpha, dA, layoutA, &beta, dB, nullptr, dC, layoutC, nullptr);
    ASSERT_EQ(hipblasLtErr, HIPBLAS_STATUS_INVALID_VALUE);

    hipblasLtErr = hipblasLtMatrixTransform(
        handle, desc, &alpha, dA, nullptr, &beta, dB, layoutB, dC, layoutC, nullptr);
    ASSERT_EQ(hipblasLtErr, HIPBLAS_STATUS_INVALID_VALUE);

    hipblasLtErr = hipblasLtMatrixTransform(
        handle, desc, &alpha, dA, layoutA, &beta, dB, nullptr, dC, nullptr, nullptr);
    ASSERT_EQ(hipblasLtErr, HIPBLAS_STATUS_INVALID_VALUE);

    hipblasLtErr = hipblasLtMatrixTransform(
        handle, desc, &alpha, dA, nullptr, &beta, dB, nullptr, dC, layoutC, nullptr);
    ASSERT_EQ(hipblasLtErr, HIPBLAS_STATUS_INVALID_VALUE);

    hipblasLtErr = hipblasLtMatrixTransform(
        handle, desc, &alpha, dA, nullptr, &beta, dB, layoutB, dC, nullptr, nullptr);
    ASSERT_EQ(hipblasLtErr, HIPBLAS_STATUS_INVALID_VALUE);

    hipblasLtErr = hipblasLtMatrixTransform(
        handle, desc, &alpha, dA, layoutA, &beta, dB, layoutB, dC, layoutC, nullptr);
    ASSERT_EQ(hipblasLtErr, HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    hipblasLtErr = hipblasLtMatrixTransformDescDestroy(desc);
    hipblasLtErr = hipblasLtDestroy(handle);
    hipblasLtErr = hipblasLtMatrixLayoutDestroy(layoutA);
    hipblasLtErr = hipblasLtMatrixLayoutDestroy(layoutB);
    hipblasLtErr = hipblasLtMatrixLayoutDestroy(layoutC);
}

TEST(MatrixTransformTest, NullA)
{
    int64_t                     m             = 1024;
    int64_t                     n             = 1024;
    int32_t                     batchSize     = 1;
    auto                        datatype      = HIP_R_32F;
    auto                        scaleDatatype = HIP_R_32F;
    auto                        opA           = HIPBLAS_OP_N;
    auto                        opB           = HIPBLAS_OP_N;
    auto                        orderA        = HIPBLASLT_ORDER_ROW;
    auto                        orderB        = HIPBLASLT_ORDER_ROW;
    auto                        orderC        = HIPBLASLT_ORDER_COL;
    float                       alpha         = 0;
    float                       beta          = 1;
    int64_t                     batchStride   = m * n;
    std::pair<int64_t, int64_t> shapeA;
    std::pair<int64_t, int64_t> shapeB;
    shapeA.first  = opA == HIPBLAS_OP_N ? m : n;
    shapeA.second = opA == HIPBLAS_OP_N ? n : m;
    shapeB.first  = opB == HIPBLAS_OP_N ? m : n;
    shapeB.second = opB == HIPBLAS_OP_N ? n : m;
    uint32_t ldB  = (orderB == HIPBLASLT_ORDER_ROW)
                        ? getLeadingDimSize<true>(shapeB.first, shapeB.second)
                        : getLeadingDimSize<false>(shapeB.first, shapeB.second);
    uint32_t ldC  = (orderC == HIPBLASLT_ORDER_ROW) ? getLeadingDimSize<true>(m, n)
                                                    : getLeadingDimSize<false>(m, n);

    auto  inputs = makeMatrixTransformIOPtr(datatype, m, n, batchSize);
    void* dB     = inputs->getBuf(1);
    void* dC     = inputs->getBuf(2);

    hipblasLtMatrixTransformDesc_t desc;
    auto                   hipblasLtErr = hipblasLtMatrixTransformDescCreate(&desc, scaleDatatype);
    hipblasLtPointerMode_t pMode        = HIPBLASLT_POINTER_MODE_HOST;
    hipblasLtErr                        = hipblasLtMatrixTransformDescSetAttribute(
        desc,
        hipblasLtMatrixTransformDescAttributes_t::HIPBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE,
        &pMode,
        sizeof(pMode));

    ASSERT_EQ(hipblasLtErr, HIPBLAS_STATUS_SUCCESS);

    hipblasLtErr = hipblasLtMatrixTransformDescSetAttribute(
        desc, HIPBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &opA, sizeof(opA));
    hipblasLtErr = hipblasLtMatrixTransformDescSetAttribute(
        desc, HIPBLASLT_MATRIX_TRANSFORM_DESC_TRANSB, &opB, sizeof(opB));
    hipblasLtMatrixLayout_t layoutB, layoutC;
    hipblasLtErr
        = hipblasLtMatrixLayoutCreate(&layoutB, datatype, shapeB.first, shapeB.second, ldB);
    hipblasLtErr = hipblasLtMatrixLayoutCreate(&layoutC, datatype, m, n, ldC);
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutB,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_ORDER,
        &orderB,
        sizeof(orderB));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutC,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_ORDER,
        &orderC,
        sizeof(orderC));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutB,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batchSize,
        sizeof(batchSize));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutC,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batchSize,
        sizeof(batchSize));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutB,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &batchStride,
        sizeof(batchStride));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutC,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &batchStride,
        sizeof(batchStride));
    hipblasLtHandle_t handle{};
    hipblasLtErr = hipblasLtCreate(&handle);
    hipblasLtErr = hipblasLtMatrixTransform(
        handle, desc, nullptr, nullptr, nullptr, &beta, dB, layoutB, dC, layoutC, nullptr);
    ASSERT_EQ(hipblasLtErr, HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    hipblasLtErr = hipblasLtMatrixTransformDescDestroy(desc);
    hipblasLtErr = hipblasLtDestroy(handle);
    hipblasLtErr = hipblasLtMatrixLayoutDestroy(layoutB);
    hipblasLtErr = hipblasLtMatrixLayoutDestroy(layoutC);
    auto rowMajA = (orderA == HIPBLASLT_ORDER_ROW);
    auto rowMajB = (orderB == HIPBLASLT_ORDER_ROW);
    auto rowMajC = (orderC == HIPBLASLT_ORDER_ROW);
    auto transA  = (opA != HIPBLAS_OP_N);
    auto transB  = (opB != HIPBLAS_OP_N);
    validation<float>(dC,
                      nullptr,
                      dB,
                      alpha,
                      beta,
                      m,
                      n,
                      0,
                      ldB,
                      ldC,
                      batchSize,
                      batchStride,
                      rowMajA,
                      rowMajB,
                      rowMajC,
                      transA,
                      transB);
}

TEST(MatrixTransformTest, NullB)
{
    int64_t                     m             = 1024;
    int64_t                     n             = 1024;
    int32_t                     batchSize     = 1;
    auto                        datatype      = HIP_R_32F;
    auto                        scaleDatatype = HIP_R_32F;
    auto                        opA           = HIPBLAS_OP_N;
    auto                        opB           = HIPBLAS_OP_N;
    auto                        orderA        = HIPBLASLT_ORDER_ROW;
    auto                        orderB        = HIPBLASLT_ORDER_ROW;
    auto                        orderC        = HIPBLASLT_ORDER_COL;
    float                       alpha         = 1;
    float                       beta          = 0;
    int64_t                     batchStride   = m * n;
    std::pair<int64_t, int64_t> shapeA;
    std::pair<int64_t, int64_t> shapeB;
    shapeA.first  = opA == HIPBLAS_OP_N ? m : n;
    shapeA.second = opA == HIPBLAS_OP_N ? n : m;
    shapeB.first  = opB == HIPBLAS_OP_N ? m : n;
    shapeB.second = opB == HIPBLAS_OP_N ? n : m;
    uint32_t ldA  = (orderA == HIPBLASLT_ORDER_ROW)
                        ? getLeadingDimSize<true>(shapeA.first, shapeA.second)
                        : getLeadingDimSize<false>(shapeA.first, shapeA.second);
    uint32_t ldC  = (orderC == HIPBLASLT_ORDER_ROW) ? getLeadingDimSize<true>(m, n)
                                                    : getLeadingDimSize<false>(m, n);

    auto  inputs = makeMatrixTransformIOPtr(datatype, m, n, batchSize);
    void* dA     = inputs->getBuf(0);
    void* dC     = inputs->getBuf(2);

    hipblasLtMatrixTransformDesc_t desc;
    auto                   hipblasLtErr = hipblasLtMatrixTransformDescCreate(&desc, scaleDatatype);
    hipblasLtPointerMode_t pMode        = HIPBLASLT_POINTER_MODE_HOST;
    hipblasLtErr                        = hipblasLtMatrixTransformDescSetAttribute(
        desc,
        hipblasLtMatrixTransformDescAttributes_t::HIPBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE,
        &pMode,
        sizeof(pMode));

    ASSERT_EQ(hipblasLtErr, HIPBLAS_STATUS_SUCCESS);

    hipblasLtErr = hipblasLtMatrixTransformDescSetAttribute(
        desc, HIPBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &opA, sizeof(opA));
    hipblasLtErr = hipblasLtMatrixTransformDescSetAttribute(
        desc, HIPBLASLT_MATRIX_TRANSFORM_DESC_TRANSB, &opB, sizeof(opB));
    hipblasLtMatrixLayout_t layoutA, layoutC;
    hipblasLtErr
        = hipblasLtMatrixLayoutCreate(&layoutA, datatype, shapeA.first, shapeA.second, ldA);
    hipblasLtErr = hipblasLtMatrixLayoutCreate(&layoutC, datatype, m, n, ldC);
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutA,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_ORDER,
        &orderA,
        sizeof(orderA));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutC,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_ORDER,
        &orderC,
        sizeof(orderC));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutA,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batchSize,
        sizeof(batchSize));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutC,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batchSize,
        sizeof(batchSize));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutA,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &batchStride,
        sizeof(batchStride));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutC,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &batchStride,
        sizeof(batchStride));
    hipblasLtHandle_t handle{};
    hipblasLtErr = hipblasLtCreate(&handle);
    hipblasLtErr = hipblasLtMatrixTransform(
        handle, desc, &alpha, dA, layoutA, nullptr, nullptr, nullptr, dC, layoutC, nullptr);
    ASSERT_EQ(hipblasLtErr, HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    hipblasLtErr = hipblasLtMatrixTransformDescDestroy(desc);
    hipblasLtErr = hipblasLtDestroy(handle);
    hipblasLtErr = hipblasLtMatrixLayoutDestroy(layoutA);
    hipblasLtErr = hipblasLtMatrixLayoutDestroy(layoutC);
    auto rowMajA = (orderA == HIPBLASLT_ORDER_ROW);
    auto rowMajB = (orderB == HIPBLASLT_ORDER_ROW);
    auto rowMajC = (orderC == HIPBLASLT_ORDER_ROW);
    auto transA  = (opA != HIPBLAS_OP_N);
    auto transB  = (opB != HIPBLAS_OP_N);
    validation<float>(dC,
                      dA,
                      nullptr,
                      alpha,
                      beta,
                      m,
                      n,
                      ldA,
                      0,
                      ldC,
                      batchSize,
                      batchStride,
                      rowMajA,
                      rowMajB,
                      rowMajC,
                      transA,
                      transB);
}

TEST(MatrixTransformTest, ScalarsOnDevice)
{
    int64_t m             = 1024;
    int64_t n             = 1024;
    int32_t batchSize     = 1;
    auto    datatype      = HIP_R_32F;
    auto    scaleDatatype = HIP_R_32F;
    auto    opA           = HIPBLAS_OP_N;
    auto    opB           = HIPBLAS_OP_N;
    auto    orderA        = HIPBLASLT_ORDER_ROW;
    auto    orderB        = HIPBLASLT_ORDER_ROW;
    auto    orderC        = HIPBLASLT_ORDER_COL;
    float   alpha         = 1;
    float   beta          = 1;
    float*  deviceAlpha{};
    float*  deviceBeta{};
    auto    hipErr                          = hipMalloc(&deviceAlpha, sizeof(deviceAlpha));
    hipErr                                  = hipMalloc(&deviceBeta, sizeof(deviceBeta));
    hipErr                                  = hipMemcpyHtoD(deviceAlpha, &alpha, sizeof(alpha));
    hipErr                                  = hipMemcpyHtoD(deviceBeta, &beta, sizeof(beta));
    int64_t                     batchStride = m * n;
    std::pair<int64_t, int64_t> shapeA;
    std::pair<int64_t, int64_t> shapeB;
    shapeA.first  = opA == HIPBLAS_OP_N ? m : n;
    shapeA.second = opA == HIPBLAS_OP_N ? n : m;
    shapeB.first  = opB == HIPBLAS_OP_N ? m : n;
    shapeB.second = opB == HIPBLAS_OP_N ? n : m;
    uint32_t ldA  = (orderA == HIPBLASLT_ORDER_ROW)
                        ? getLeadingDimSize<true>(shapeA.first, shapeA.second)
                        : getLeadingDimSize<false>(shapeA.first, shapeA.second);
    uint32_t ldB  = (orderB == HIPBLASLT_ORDER_ROW)
                        ? getLeadingDimSize<true>(shapeB.first, shapeB.second)
                        : getLeadingDimSize<false>(shapeB.first, shapeB.second);
    uint32_t ldC  = (orderC == HIPBLASLT_ORDER_ROW) ? getLeadingDimSize<true>(m, n)
                                                    : getLeadingDimSize<false>(m, n);

    auto  inputs = makeMatrixTransformIOPtr(datatype, m, n, batchSize);
    void* dA     = inputs->getBuf(0);
    void* dB     = inputs->getBuf(1);
    void* dC     = inputs->getBuf(2);

    hipblasLtMatrixTransformDesc_t desc;
    auto                   hipblasLtErr = hipblasLtMatrixTransformDescCreate(&desc, scaleDatatype);
    hipblasLtPointerMode_t pMode        = HIPBLASLT_POINTER_MODE_DEVICE;
    hipblasLtErr                        = hipblasLtMatrixTransformDescSetAttribute(
        desc,
        hipblasLtMatrixTransformDescAttributes_t::HIPBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE,
        &pMode,
        sizeof(pMode));

    ASSERT_EQ(hipblasLtErr, HIPBLAS_STATUS_SUCCESS);

    hipblasLtErr = hipblasLtMatrixTransformDescSetAttribute(
        desc, HIPBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &opA, sizeof(opA));
    hipblasLtErr = hipblasLtMatrixTransformDescSetAttribute(
        desc, HIPBLASLT_MATRIX_TRANSFORM_DESC_TRANSB, &opB, sizeof(opB));
    hipblasLtMatrixLayout_t layoutA, layoutB, layoutC;
    hipblasLtErr
        = hipblasLtMatrixLayoutCreate(&layoutA, datatype, shapeA.first, shapeA.second, ldA);
    hipblasLtErr
        = hipblasLtMatrixLayoutCreate(&layoutB, datatype, shapeB.first, shapeB.second, ldB);
    hipblasLtErr = hipblasLtMatrixLayoutCreate(&layoutC, datatype, m, n, ldC);
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutA,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_ORDER,
        &orderA,
        sizeof(orderA));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutB,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_ORDER,
        &orderB,
        sizeof(orderB));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutC,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_ORDER,
        &orderC,
        sizeof(orderC));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutA,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batchSize,
        sizeof(batchSize));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutB,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batchSize,
        sizeof(batchSize));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutC,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batchSize,
        sizeof(batchSize));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutA,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &batchStride,
        sizeof(batchStride));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutB,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &batchStride,
        sizeof(batchStride));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutC,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &batchStride,
        sizeof(batchStride));
    hipblasLtHandle_t handle{};
    hipblasLtErr = hipblasLtCreate(&handle);
    hipblasLtErr = hipblasLtMatrixTransform(
        handle, desc, deviceAlpha, dA, layoutA, deviceBeta, dB, layoutB, dC, layoutC, nullptr);
    ASSERT_EQ(hipblasLtErr, HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    hipblasLtErr = hipblasLtMatrixTransformDescDestroy(desc);
    hipblasLtErr = hipblasLtDestroy(handle);
    hipblasLtErr = hipblasLtMatrixLayoutDestroy(layoutA);
    hipblasLtErr = hipblasLtMatrixLayoutDestroy(layoutB);
    hipblasLtErr = hipblasLtMatrixLayoutDestroy(layoutC);
    hipErr       = hipFree(deviceAlpha);
    hipErr       = hipFree(deviceBeta);
    auto rowMajA = (orderA == HIPBLASLT_ORDER_ROW);
    auto rowMajB = (orderB == HIPBLASLT_ORDER_ROW);
    auto rowMajC = (orderC == HIPBLASLT_ORDER_ROW);
    auto transA  = (opA != HIPBLAS_OP_N);
    auto transB  = (opB != HIPBLAS_OP_N);
    validation<float>(dC,
                      dA,
                      dB,
                      alpha,
                      beta,
                      m,
                      n,
                      ldA,
                      ldB,
                      ldC,
                      batchSize,
                      batchStride,
                      rowMajA,
                      rowMajB,
                      rowMajC,
                      transA,
                      transB);
}

TEST(MatrixTransformTest, MultipleDevices)
{
    int numDevices{};
    int curDevice{};
    auto hipErr = hipGetDeviceCount(&numDevices);
    EXPECT_EQ(hipErr, hipSuccess);
    hipErr = hipGetDevice(&curDevice);
    EXPECT_EQ(hipErr, hipSuccess);
    // acquire at most 2 devices
    numDevices = std::min<int>(numDevices, 2);

    for (int deviceId = 0; deviceId < numDevices; ++deviceId)
    {
        hipErr = hipSetDevice(deviceId);
        EXPECT_EQ(hipErr, hipSuccess);
        int64_t                     m             = 1024;
        int64_t                     n             = 1024;
        int32_t                     batchSize     = 1;
        auto                        datatype      = HIP_R_32F;
        auto                        scaleDatatype = HIP_R_32F;
        auto                        opA           = HIPBLAS_OP_N;
        auto                        opB           = HIPBLAS_OP_N;
        auto                        orderA        = HIPBLASLT_ORDER_ROW;
        auto                        orderB        = HIPBLASLT_ORDER_ROW;
        auto                        orderC        = HIPBLASLT_ORDER_COL;
        float                       alpha         = 1;
        float                       beta          = 1;
        int64_t                     batchStride   = m * n;
        std::pair<int64_t, int64_t> shapeA;
        std::pair<int64_t, int64_t> shapeB;
        shapeA.first  = opA == HIPBLAS_OP_T ? n : m;
        shapeA.second = opA == HIPBLAS_OP_T ? m : n;
        shapeB.first  = opB == HIPBLAS_OP_T ? n : m;
        shapeB.second = opB == HIPBLAS_OP_T ? m : n;
        uint32_t ldA  = (orderA == HIPBLASLT_ORDER_ROW)
                            ? getLeadingDimSize<true>(shapeA.first, shapeA.second)
                            : getLeadingDimSize<false>(shapeA.first, shapeA.second);
        uint32_t ldB  = (orderB == HIPBLASLT_ORDER_ROW)
                            ? getLeadingDimSize<true>(shapeB.first, shapeB.second)
                            : getLeadingDimSize<false>(shapeB.first, shapeB.second);
        uint32_t ldC  = (orderC == HIPBLASLT_ORDER_ROW) ? getLeadingDimSize<true>(m, n)
                                                        : getLeadingDimSize<false>(m, n);

        auto  inputs = makeMatrixTransformIOPtr(datatype, m, n, batchSize);
        void* dA     = inputs->getBuf(0);
        void* dB     = inputs->getBuf(1);
        void* dC     = inputs->getBuf(2);

        hipblasLtMatrixTransformDesc_t desc;
        auto                   hipblasLtErr = hipblasLtMatrixTransformDescCreate(&desc, scaleDatatype);
        hipblasLtPointerMode_t pMode        = HIPBLASLT_POINTER_MODE_HOST;
        hipblasLtErr                        = hipblasLtMatrixTransformDescSetAttribute(
            desc,
            hipblasLtMatrixTransformDescAttributes_t::HIPBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE,
            &pMode,
            sizeof(pMode));

        ASSERT_EQ(hipblasLtErr, HIPBLAS_STATUS_SUCCESS);

        hipblasLtErr = hipblasLtMatrixTransformDescSetAttribute(
            desc, HIPBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &opA, sizeof(opA));
        hipblasLtErr = hipblasLtMatrixTransformDescSetAttribute(
            desc, HIPBLASLT_MATRIX_TRANSFORM_DESC_TRANSB, &opB, sizeof(opB));
        hipblasLtMatrixLayout_t layoutA, layoutB, layoutC;
        hipblasLtErr
            = hipblasLtMatrixLayoutCreate(&layoutA, datatype, shapeA.first, shapeA.second, ldA);
        hipblasLtErr
            = hipblasLtMatrixLayoutCreate(&layoutB, datatype, shapeB.first, shapeB.second, ldB);
        hipblasLtErr = hipblasLtMatrixLayoutCreate(&layoutC, datatype, m, n, ldC);
        hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
            layoutA,
            hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_ORDER,
            &orderA,
            sizeof(orderA));
        hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
            layoutB,
            hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_ORDER,
            &orderB,
            sizeof(orderB));
        hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
            layoutC,
            hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_ORDER,
            &orderC,
            sizeof(orderC));
        hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
            layoutA,
            hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
            &batchSize,
            sizeof(batchSize));
        hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
            layoutB,
            hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
            &batchSize,
            sizeof(batchSize));
        hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
            layoutC,
            hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
            &batchSize,
            sizeof(batchSize));
        hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
            layoutA,
            hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
            &batchStride,
            sizeof(batchStride));
        hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
            layoutB,
            hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
            &batchStride,
            sizeof(batchStride));
        hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
            layoutC,
            hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
            &batchStride,
            sizeof(batchStride));
        hipblasLtHandle_t handle{};
        hipblasLtErr = hipblasLtCreate(&handle);
        hipblasLtErr = hipblasLtMatrixTransform(
            handle, desc, &alpha, dA, layoutA, &beta, dB, layoutB, dC, layoutC, nullptr);
        ASSERT_EQ(hipblasLtErr, HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        auto rowMajA = (orderA == HIPBLASLT_ORDER_ROW);
        auto rowMajB = (orderB == HIPBLASLT_ORDER_ROW);
        auto rowMajC = (orderC == HIPBLASLT_ORDER_ROW);
        auto transA  = (opA == HIPBLAS_OP_T);
        auto transB  = (opB == HIPBLAS_OP_T);

        if(datatype == HIP_R_32F)
        {
            validation<float>(dC,
                            dA,
                            dB,
                            alpha,
                            beta,
                            m,
                            n,
                            ldA,
                            ldB,
                            ldC,
                            batchSize,
                            batchStride,
                            rowMajA,
                            rowMajB,
                            rowMajC,
                            transA,
                            transB);
        }
        else if(datatype == HIP_R_16F)
        {
            validation<hipblasLtHalf>(dC,
                                    dA,
                                    dB,
                                    alpha,
                                    beta,
                                    m,
                                    n,
                                    ldA,
                                    ldB,
                                    ldC,
                                    batchSize,
                                    batchStride,
                                    rowMajA,
                                    rowMajB,
                                    rowMajC,
                                    transA,
                                    transB);
        }
        else if(datatype == HIP_R_16BF)
        {
            validation<hipblasLtBfloat16>(dC,
                                        dA,
                                        dB,
                                        alpha,
                                        beta,
                                        m,
                                        n,
                                        ldA,
                                        ldB,
                                        ldC,
                                        batchSize,
                                        batchStride,
                                        rowMajA,
                                        rowMajB,
                                        rowMajC,
                                        transA,
                                        transB);
        }
        else if(datatype == HIP_R_8I)
        {
            validation<int8_t>(dC,
                            dA,
                            dB,
                            alpha,
                            beta,
                            m,
                            n,
                            ldA,
                            ldB,
                            ldC,
                            batchSize,
                            batchStride,
                            rowMajA,
                            rowMajB,
                            rowMajC,
                            transA,
                            transB);
        }
        else if(datatype == HIP_R_32I)
        {
            validation<int32_t>(dC,
                                dA,
                                dB,
                                alpha,
                                beta,
                                m,
                                n,
                                ldA,
                                ldB,
                                ldC,
                                batchSize,
                                batchStride,
                                rowMajA,
                                rowMajB,
                                rowMajC,
                                transA,
                                transB);
        }

        hipblasLtErr = hipblasLtMatrixTransformDescDestroy(desc);
        hipblasLtErr = hipblasLtDestroy(handle);
        hipblasLtErr = hipblasLtMatrixLayoutDestroy(layoutA);
        hipblasLtErr = hipblasLtMatrixLayoutDestroy(layoutB);
        hipblasLtErr = hipblasLtMatrixLayoutDestroy(layoutC);
    }

    hipErr = hipSetDevice(curDevice);
    EXPECT_EQ(hipErr, hipSuccess);
}

INSTANTIATE_TEST_SUITE_P(
    AllCombinations,
    MatrixTransformTest,
    ::testing::Combine(::testing::ValuesIn({int64_t(1), int64_t(127), int64_t(1024)}),
                       ::testing::ValuesIn({int64_t(1), int64_t(127), int64_t(1024)}),
                       ::testing::ValuesIn({HIP_R_32F, HIP_R_16F, HIP_R_16BF, HIP_R_8I, HIP_R_32I}),
                       ::testing::ValuesIn({HIP_R_32F}),
                       ::testing::ValuesIn({HIPBLAS_OP_N, HIPBLAS_OP_T}),
                       ::testing::ValuesIn({HIPBLAS_OP_N, HIPBLAS_OP_T}),
                       ::testing::ValuesIn({HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_COL}),
                       ::testing::ValuesIn({HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_COL}),
                       ::testing::ValuesIn({HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_COL})));
