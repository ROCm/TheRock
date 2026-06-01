// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iomanip>
#include <vector>
#include <iostream>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/utility/data_type.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_mx.hpp"
#include "ck/library/tensor_operation_instance/gpu/gemm_mx.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_v3_mx.hpp"

using F16 = ck::half_t;
using F32 = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using AElementOp = PassThrough;
using BElementOp = PassThrough;
using CElementOp = PassThrough;

using ADataType = ck::f8_t;
using BDataType = ck::f8_t;
using CDataType = ck::half_t;

using XDataType       = ck::e8m0_bexp_t;
using XPackedDataType = int32_t;
template <typename X, typename Y>
inline constexpr bool is_same_v = ck::is_same<X, Y>::value;

using ALayout = Row;
using BLayout = Col;
using CLayout = Row;

using AScaleLayout = Row;
using BScaleLayout = Col;

template <bool KLast>
void preShuffleScaleBuffer(ck::e8m0_bexp_t* src, ck::e8m0_bexp_t* dst, int MN, int K)
{
    int MNXdlPack = 2;
    int KXdlPack  = 2;

    int XdlMNThread = 16;
    int XdlKThread  = 64 / XdlMNThread;

    int K0 = K / KXdlPack / XdlKThread; // KRepeat

    // The 4 16x128 building blocks will be packed into 1 32x256 for F4
    // The 8 16x16x128 mfma will be packed into 1 32x32x256 for F4

    // unfold the MN32xK(256/32) scale buffer
    //    4            16             2           2
    // To XdlKThread-> XdlMNThread -> KXdlPack -> MNXdlPack
    // Then, MNRepeat->KRepeat

    for(int n = 0; n < MN; ++n)
    {
        for(int k = 0; k < K; ++k)
        {
            int n0    = n / (XdlMNThread * MNXdlPack); // i MNRepeat
            int tempn = n % (XdlMNThread * MNXdlPack);
            int n1    = tempn % XdlMNThread; // i XdlMNThread
            int n2    = tempn / XdlMNThread; // i MNXdlPack

            int k0    = k / (XdlKThread * KXdlPack); // i KRepeat
            int tempk = k % (XdlKThread * KXdlPack);
            int k1    = tempk % XdlKThread; // i XdlKThread
            int k2    = tempk / XdlKThread; // i KXdlPack

            int outputIndex = n0 * MNXdlPack * KXdlPack * XdlMNThread * XdlKThread * K0 +
                              k0 * MNXdlPack * KXdlPack * XdlMNThread * XdlKThread +
                              k1 * MNXdlPack * KXdlPack * XdlMNThread + n1 * MNXdlPack * KXdlPack +
                              k2 * MNXdlPack + n2;
            // src[n * K + k] = ck::type_convert<ck::e8m0_bexp_t>(static_cast<float>(powf(2.0f, n2 +
            // k2 * MNXdlPack)));
            if constexpr(KLast)
                dst[outputIndex] = src[n * K + k];
            else
                dst[outputIndex] = src[k * MN + n];
        }
    }
}

struct SimpleDeviceMem
{
    SimpleDeviceMem() = delete;

    SimpleDeviceMem(std::size_t mem_size) : p_mem_{}
    {
        mem_size_ = mem_size;
        (void)hipMalloc(static_cast<void**>(&p_mem_), mem_size);
    }

    void* GetDeviceBuffer() { return p_mem_; }

    ~SimpleDeviceMem() { (void)hipFree(p_mem_); }

    void* p_mem_;
    std::size_t mem_size_;
};

int main(int argc, char* argv[])
{
    // GEMM shape
    ck::index_t M = 3840;
    ck::index_t N = 4096;
    ck::index_t K = 4096;

    ck::index_t StrideA = 4096;
    ck::index_t StrideB = 4096;
    ck::index_t StrideC = 4096;

    ck::index_t KBatch = 1;

    /* Require by mx type*/
    constexpr ck::index_t ScaleBlockSize = 32; // scaling block size

    if(argc == 1)
    {
        // use default case
    }
    else if(argc == 7)
    {
        M = std::stoi(argv[1]);
        N = std::stoi(argv[2]);
        K = std::stoi(argv[3]);

        StrideA = std::stoi(argv[4]);
        StrideB = std::stoi(argv[5]);
        StrideC = std::stoi(argv[6]);
    }
    else
    {
        printf("arg1 to 6: M, N, K, StrideA, StrideB, StrideC\n");
        exit(0);
    }

    auto f_matrix_space_size =
        [](std::size_t nRow, std::size_t nCol, std::size_t stride, auto layout) {
            using Layout = decltype(layout);

            if constexpr(std::is_same<Layout, Row>::value)
            {
                return (nRow - 1) * stride + nCol;
            }
            else
            {
                return (nCol - 1) * stride + nRow;
            }
        };

    /* Scale stride Calculation */
    auto f_get_default_stride =
        [](ck::index_t row, ck::index_t col, ck::index_t stride, auto layout) {
            if(stride == -1)
            {
                // give a chance if stride is -1, return a default packed stride
                if constexpr(std::is_same_v<decltype(layout), ck::tensor_layout::gemm::RowMajor>)
                    return static_cast<ck::index_t>(col);
                else
                    return static_cast<ck::index_t>(row);
            }
            else
                return static_cast<ck::index_t>(stride);
        };

    if(K % ScaleBlockSize != 0)
    {
        throw std::runtime_error("wrong! K must be multiple of ScaleBlockSize.");
    };
    auto Scale_Padded_M = (M + ScaleBlockSize - 1) / ScaleBlockSize * ScaleBlockSize;
    auto Scale_Stride_AM =
        f_get_default_stride(Scale_Padded_M, K / ScaleBlockSize, -1, AScaleLayout{});
    auto Scale_Stride_BN = f_get_default_stride(K / ScaleBlockSize, N, -1, BScaleLayout{});

    SimpleDeviceMem a_device_buf(sizeof(ADataType) * f_matrix_space_size(M, K, StrideA, ALayout{}));
    SimpleDeviceMem b_device_buf(sizeof(BDataType) * f_matrix_space_size(K, N, StrideB, BLayout{}));
    SimpleDeviceMem c_device_buf(sizeof(CDataType) * f_matrix_space_size(M, N, StrideC, CLayout{}));
    SimpleDeviceMem a_scale_device_buf(
        sizeof(XDataType) *
        f_matrix_space_size(Scale_Padded_M, K / ScaleBlockSize, Scale_Stride_AM, AScaleLayout{}));
    SimpleDeviceMem b_scale_device_buf(
        sizeof(XDataType) *
        f_matrix_space_size(K / ScaleBlockSize, N, Scale_Stride_BN, BScaleLayout{}));

    using DeviceOp =
        ck::tensor_operation::device::DeviceGemmMX<ALayout,
                                                   BLayout,
                                                   CLayout,
                                                   ADataType,
                                                   XPackedDataType,
                                                   BDataType,
                                                   XPackedDataType,
                                                   CDataType,
                                                   ScaleBlockSize,
                                                   ck::tensor_operation::element_wise::PassThrough,
                                                   ck::tensor_operation::element_wise::PassThrough,
                                                   ck::tensor_operation::element_wise::PassThrough>;

    // get device op instances
    const auto op_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp>::GetInstances();

    std::cout << "found " << op_ptrs.size() << " instances" << std::endl;

    const auto a_element_op = AElementOp{};
    const auto b_element_op = BElementOp{};
    const auto c_element_op = CElementOp{};

    std::string best_op_name;
    bool found            = false;
    int best_op_id        = -1;
    float best_ave_time   = 0;
    float best_tflops     = 0;
    float best_gb_per_sec = 0;

    // profile device operation instances
    std::cout << "Run all instances and do timing" << std::endl;

    for(int i = 0; i < op_ptrs.size(); ++i)
    {
        auto& op_ptr = op_ptrs[i];

        auto argument_ptr = op_ptr->MakeArgumentPointer(
            static_cast<ADataType*>(a_device_buf.GetDeviceBuffer()),
            static_cast<XPackedDataType*>(a_scale_device_buf.GetDeviceBuffer()),
            static_cast<BDataType*>(b_device_buf.GetDeviceBuffer()),
            static_cast<XPackedDataType*>(b_scale_device_buf.GetDeviceBuffer()),
            static_cast<CDataType*>(c_device_buf.GetDeviceBuffer()),
            M,
            N,
            K,
            StrideA,
            Scale_Stride_AM,
            StrideB,
            Scale_Stride_BN,
            StrideC,
            KBatch,
            a_element_op,
            b_element_op,
            c_element_op);

        auto invoker_ptr = op_ptr->MakeInvokerPointer();

        std::string op_name = op_ptr->GetTypeString();

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            float ave_time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, true});

            std::size_t flop =
                std::size_t(2) * M * N * K + std::size_t(2) * M * N * K / ScaleBlockSize;

            std::size_t num_btype = sizeof(ADataType) * M * K / ck::packed_size_v<ADataType> +
                                    sizeof(BDataType) * K * N / ck::packed_size_v<BDataType> +
                                    sizeof(CDataType) * M * N +
                                    sizeof(XDataType) * M * K / ScaleBlockSize +
                                    sizeof(XDataType) * N * K / ScaleBlockSize;

            float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

            float gb_per_sec = num_btype / 1.E6 / ave_time;

            std::cout << "Perf: " << std::setw(10) << ave_time << " ms, " << tflops << " TFlops, "
                      << gb_per_sec << " GB/s, " << op_name << std::endl;

            if(tflops > best_tflops)
            {
                found           = true;
                best_op_id      = i;
                best_op_name    = op_name;
                best_tflops     = tflops;
                best_ave_time   = ave_time;
                best_gb_per_sec = gb_per_sec;
            }
        }
        else
        {
            std::cout << op_name << " does not support this problem" << std::endl;
        }
    }

    std::cout << "Best Perf: " << best_ave_time << " ms, " << best_tflops << " TFlops, "
              << best_gb_per_sec << " GB/s, " << best_op_name << std::endl;

    // run the best intance
    if(found)
    {
        auto& op_ptr = op_ptrs[best_op_id];

        std::cout << "Run the best instance without timing: " << op_ptr->GetTypeString()
                  << std::endl;

        auto argument_ptr = op_ptr->MakeArgumentPointer(
            static_cast<ADataType*>(a_device_buf.GetDeviceBuffer()),
            static_cast<XPackedDataType*>(a_scale_device_buf.GetDeviceBuffer()),
            static_cast<BDataType*>(b_device_buf.GetDeviceBuffer()),
            static_cast<XPackedDataType*>(b_scale_device_buf.GetDeviceBuffer()),
            static_cast<CDataType*>(c_device_buf.GetDeviceBuffer()),
            M,
            N,
            K,
            StrideA,
            Scale_Stride_AM,
            StrideB,
            Scale_Stride_BN,
            StrideC,
            KBatch,
            a_element_op,
            b_element_op,
            c_element_op);

        auto invoker_ptr = op_ptr->MakeInvokerPointer();

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, false});
        }

        std::cout << "Done" << std::endl;
    }

    return 0;
}
