// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

int run(int argc, char* argv[])
{
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;

    // GEMM shape for A/B0/B1/C
    // C_g_m_o = A_g_m_k * B0_g_k_n * B1_g_n_o
    ck::index_t M = 113;
#ifdef CK_MHA_USE_RCCR_LAYOUT
    ck::index_t N = 480; // Must be multiple of 8 even with padding.
#else
    ck::index_t N = 477;
#endif
    ck::index_t K = 200; // Must be multiple of 8 even with padding.
    ck::index_t O = 208; // Must be multiple of 8 even with padding.
    ck::index_t G = 91;  // Batch

    float alpha = 1;

    if(argc == 1)
    {
        // use default case
    }
    else if(argc == 4)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);
    }
    else if(argc == 10)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);

        M = std::stoi(argv[4]);
        N = std::stoi(argv[5]);
        K = std::stoi(argv[6]);
        O = std::stoi(argv[7]);
        G = std::stoi(argv[8]);

        alpha = std::stof(argv[9]);
    }
    else
    {
        printf("arg1: verification (0=no, 1=yes)\n");
        printf("arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n");
        printf("arg3: time kernel (0=no, 1=yes)\n");
        printf("arg4 to 8: M, N, K, O, G\n");
        printf("arg9: scale (alpha)\n");
        exit(0);
    }

    std::vector<ck::index_t> a_g_m_k_lengths{G, M, K};
    std::vector<ck::index_t> a_g_m_k_strides{M * K, K, 1}; // A layout [G, M, K]
    std::vector<ck::index_t> b0_g_n_k_lengths{G, N, K};
    std::vector<ck::index_t> b0_g_n_k_strides{N * K, K, 1}; // B0 layout [G, N, K]
    std::vector<ck::index_t> b1_g_o_n_lengths{G, O, N};
#ifdef CK_MHA_USE_RCCR_LAYOUT
    std::vector<ck::index_t> b1_g_o_n_strides{N * O, N, 1}; // B1 layout [G, O, N]
    auto b1_layout = Row{};
#else
    std::vector<ck::index_t> b1_g_o_n_strides{N * O, 1, O}; // B1 layout [G, N, O]
    auto b1_layout = Col{};
#endif
    std::vector<ck::index_t> c_g_m_o_lengths{G, M, O};
    std::vector<ck::index_t> c_g_m_o_strides{M * O, O, 1}; // C layout [G, M, O]

    Tensor<ADataType> a_g_m_k(a_g_m_k_lengths, a_g_m_k_strides, Row{});
    Tensor<B0DataType> b0_g_n_k(b0_g_n_k_lengths, b0_g_n_k_strides, Row{});
    Tensor<B1DataType> b1_g_o_n(b1_g_o_n_lengths, b1_g_o_n_strides, b1_layout);
    Tensor<CDataType> c_g_m_o_host_result(c_g_m_o_lengths, c_g_m_o_strides, Row{});
    Tensor<CDataType> c_g_m_o_device_result(c_g_m_o_lengths, c_g_m_o_strides, Row{});

    std::cout << "a_g_m_k: " << a_g_m_k.mDesc << std::endl;
    std::cout << "b0_g_n_k: " << b0_g_n_k.mDesc << std::endl;
    std::cout << "b1_g_o_n: " << b1_g_o_n.mDesc << std::endl;
    std::cout << "c_g_m_o: " << c_g_m_o_host_result.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        a_g_m_k.GenerateTensorValue(GeneratorTensor_2<ADataType>{-2, 2});
        b0_g_n_k.GenerateTensorValue(GeneratorTensor_2<B0DataType>{-2, 2});
        b1_g_o_n.GenerateTensorValue(GeneratorTensor_2<B1DataType>{-2, 2});
        break;
    case 2:
        a_g_m_k.GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0});
        b0_g_n_k.GenerateTensorValue(GeneratorTensor_3<B0DataType>{0.0, 1.0});
        b1_g_o_n.GenerateTensorValue(GeneratorTensor_3<B1DataType>{-0.5, 0.5});
        break;
    case 3:
        a_g_m_k.GenerateTensorValue(GeneratorTensor_2<ADataType>{-2, 2});
        b0_g_n_k.GenerateTensorValue(GeneratorTensor_Diagonal<B0DataType>{});
        b1_g_o_n.GenerateTensorValue(GeneratorTensor_Diagonal<B1DataType>{});
        break;
    case 4: // A, B0, B1 1
        a_g_m_k.GenerateTensorValue(GeneratorTensor_1<ADataType>{});
        b0_g_n_k.GenerateTensorValue(GeneratorTensor_1<B0DataType>{});
        b1_g_o_n.GenerateTensorValue(GeneratorTensor_1<B1DataType>{});
        break;
    case 5: // Rand: b1 b0; unit: a
        a_g_m_k.GenerateTensorValue(GeneratorTensor_1<ADataType>{});
        b0_g_n_k.GenerateTensorValue(GeneratorTensor_2<B0DataType>{-2, 2});
        b1_g_o_n.GenerateTensorValue(GeneratorTensor_2<B1DataType>{-2, 2});
        break;
    case 6: // Rand: a b0 ; unit: B1
        a_g_m_k.GenerateTensorValue(GeneratorTensor_2<ADataType>{-2, 2});
        b0_g_n_k.GenerateTensorValue(GeneratorTensor_2<B0DataType>{-2, 2});
        b1_g_o_n.GenerateTensorValue(GeneratorTensor_1<B1DataType>{});
        break;
    case 7: // Rand: a b1 ; unit: b0
        a_g_m_k.GenerateTensorValue(GeneratorTensor_2<ADataType>{-2, 2});
        b0_g_n_k.GenerateTensorValue(GeneratorTensor_1<B0DataType>{});
        b1_g_o_n.GenerateTensorValue(GeneratorTensor_2<B1DataType>{-2, 2});
        break;
    case 8: // Rand: a ; unit: b0 b1
        a_g_m_k.GenerateTensorValue(GeneratorTensor_2<ADataType>{-2, 2});
        b0_g_n_k.GenerateTensorValue(GeneratorTensor_1<B0DataType>{});
        b1_g_o_n.GenerateTensorValue(GeneratorTensor_1<B1DataType>{});
        break;
    case 9: // Rand: b0 ; unit: a b1
        a_g_m_k.GenerateTensorValue(GeneratorTensor_1<ADataType>{});
        b0_g_n_k.GenerateTensorValue(GeneratorTensor_2<B0DataType>{-2, 2});
        b1_g_o_n.GenerateTensorValue(GeneratorTensor_1<B1DataType>{});
        break;
    case 10: // Rand: b1 ; unit: a b0
        a_g_m_k.GenerateTensorValue(GeneratorTensor_1<ADataType>{});
        b0_g_n_k.GenerateTensorValue(GeneratorTensor_1<B0DataType>{});
        b1_g_o_n.GenerateTensorValue(GeneratorTensor_2<B1DataType>{-2, 2});
        break;
    default:
        a_g_m_k.GenerateTensorValue(GeneratorTensor_Sequential<ADataType, 2>{});
        b0_g_n_k.GenerateTensorValue(GeneratorTensor_Diagonal<B0DataType>{});
        b1_g_o_n.GenerateTensorValue(GeneratorTensor_Diagonal<B1DataType>{});
    }

    DeviceMem a_device_buf(sizeof(ADataType) * a_g_m_k.mDesc.GetElementSpaceSize());
    DeviceMem b0_device_buf(sizeof(B0DataType) * b0_g_n_k.mDesc.GetElementSpaceSize());
    DeviceMem b1_device_buf(sizeof(B1DataType) * b1_g_o_n.mDesc.GetElementSpaceSize());
    DeviceMem c_device_buf(sizeof(CDataType) * c_g_m_o_device_result.mDesc.GetElementSpaceSize());

    a_device_buf.ToDevice(a_g_m_k.mData.data());
    b0_device_buf.ToDevice(b0_g_n_k.mData.data());
    b1_device_buf.ToDevice(b1_g_o_n.mData.data());

    auto a_element_op    = AElementOp{};
    auto b0_element_op   = B0ElementOp{};
    auto acc0_element_op = Acc0ElementOp{alpha};
    auto b1_element_op   = B1ElementOp{};
    auto c_element_op    = CElementOp{};

    // do GEMM
    float best_perf         = .0;
    float best_time         = .0;
    int not_pass            = 0;
    std::string best_kernel = "";
    printf("Verification: %s\n", do_verification ? "ON" : "OFF");

    ck::static_for<0, std::tuple_size_v<DeviceMHAFactory>, 1>{}([&](auto i) -> void {
        const auto device_mha_instance = std::get<i>(DeviceMHAFactory{});

        using DeviceMHAInstance = ck::remove_cvref_t<decltype(device_mha_instance)>;
        auto gemm               = DeviceMHAInstance{};
        auto invoker_ptr        = gemm.MakeInvokerPointer();
        auto argument_ptr =
            gemm.MakeArgumentPointer(static_cast<ADataType*>(a_device_buf.GetDeviceBuffer()),
                                     static_cast<B0DataType*>(b0_device_buf.GetDeviceBuffer()),
                                     static_cast<B1DataType*>(b1_device_buf.GetDeviceBuffer()),
                                     static_cast<CDataType*>(c_device_buf.GetDeviceBuffer()),
                                     M,
                                     N,
                                     K,
                                     O,
                                     G,                   // Batch,
                                     a_g_m_k_strides[1],  // StrideA,
                                     b0_g_n_k_strides[1], // StrideB0,
#ifdef CK_MHA_USE_RCCR_LAYOUT
                                     b1_g_o_n_strides[1], // StrideB1,
#else
                                     b1_g_o_n_strides[2], // StrideB1,
#endif
                                     c_g_m_o_strides[1],  // StrideC,
                                     a_g_m_k_strides[0],  // BatchStrideA
                                     b0_g_n_k_strides[0], // BatchStrideB0
                                     b1_g_o_n_strides[0], // BatchStrideB1
                                     c_g_m_o_strides[0],  // BatchStrideC
                                     a_element_op,
                                     b0_element_op,
                                     acc0_element_op,
                                     b1_element_op,
                                     c_element_op);

        if(!gemm.IsSupportedArgument(argument_ptr.get()))
        {
            std::cout << gemm.GetTypeString() << " does not support this problem" << std::endl;
            return;
        }

        float ave_time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, time_kernel});

        std::size_t flop      = (size_t(M) * N * K * 2 + size_t(M) * N * O * 2) * G;
        std::size_t num_btype = (sizeof(ADataType) * M * K + sizeof(B0DataType) * K * N +
                                 sizeof(B1DataType) * N * O + sizeof(CDataType) * M * O) *
                                G;

        float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

        float gb_per_sec = num_btype / 1.E6 / ave_time;

        std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec
                  << " GB/s, " << gemm.GetTypeString() << std::endl;
        if(tflops > best_perf)
        {
            best_perf   = tflops;
            best_time   = ave_time * 1000;
            best_kernel = gemm.GetTypeString();
        }
        if(do_verification)
        {
            c_device_buf.FromDevice(c_g_m_o_device_result.mData.data());

            Tensor<B0DataType> b0_g_k_n({G, K, N});
            Tensor<B1DataType> b1_g_n_o({G, N, O});
            Tensor<AccDataType> acc0_g_m_n({G, M, N}); // scratch object after gemm0
            Tensor<ADataType> a1_g_m_n({G, M, N});     // scratch object after conversion

            // permute
            b0_g_n_k.ForEach(
                [&](auto& self, auto idx) { b0_g_k_n(idx[0], idx[2], idx[1]) = self(idx); });
            b1_g_o_n.ForEach(
                [&](auto& self, auto idx) { b1_g_n_o(idx[0], idx[2], idx[1]) = self(idx); });

            // gemm 0
            auto ref_gemm0          = ReferenceGemm0Instance{};
            auto ref_gemm0_invoker  = ref_gemm0.MakeInvoker();
            auto ref_gemm0_argument = ref_gemm0.MakeArgument(
                a_g_m_k, b0_g_k_n, acc0_g_m_n, a_element_op, b0_element_op, acc0_element_op);

            ref_gemm0_invoker.Run(ref_gemm0_argument);

            acc0_g_m_n.ForEach([&](auto& self, auto idx) {
                // Passthrough instead of softmax, DOES involve data type conversion.
                a1_g_m_n(idx) = ck::type_convert<ADataType, AccDataType>(self(idx));
            });

            // gemm1
            auto ref_gemm1          = ReferenceGemm1Instance{};
            auto ref_gemm1_invoker  = ref_gemm1.MakeInvoker();
            auto ref_gemm1_argument = ref_gemm1.MakeArgument(a1_g_m_n,
                                                             b1_g_n_o,
                                                             c_g_m_o_host_result,
                                                             PassThrough{},
                                                             b1_element_op,
                                                             c_element_op);

            ref_gemm1_invoker.Run(ref_gemm1_argument);

            // default absolute error and relative error is 0.001
            double rtol = 1e-3;
            double atol = 1e-3;

            // when BF16 is taken, set absolute error and relative error to 0.01
            if(std::is_same_v<ADataType, ck::bhalf_t> && std::is_same_v<B0DataType, ck::bhalf_t> &&
               std::is_same_v<B1DataType, ck::bhalf_t> && std::is_same_v<CDataType, ck::bhalf_t>)
            {
                rtol = 1e-2;
                atol = 1e-2;
            }

            bool this_run_verification = ck::utils::check_err(c_g_m_o_device_result.mData,
                                                              c_g_m_o_host_result.mData,
                                                              "Error: Incorrect results!",
                                                              rtol,
                                                              atol);
            printf("Verification: %s, Pass: %s\n",
                   do_verification ? "ON" : "OFF",
                   this_run_verification ? "YES" : "NO");

            if(!this_run_verification)
            {
                not_pass = 1;
                printf("%d th MHA instance verification Failed \n", i.value);
            }
        }
    });
    std::cout << "---------------------------------------------------------------------------------"
                 "-----------"
              << std::endl;
    std::cout << "Problem Size: G: " << G << ", M: " << M << ", N: " << N << ", K: " << K
              << ", O: " << O << std::endl;
    std::cout << "---------------------------------------------------------------------------------"
                 "-----------"
              << std::endl;
    std::cout << "Best kernel: " << best_kernel << " , " << best_perf << " TFlops , " << best_time
              << " us" << std::endl;
    std::cout << "---------------------------------------------------------------------------------"
                 "-----------"
              << std::endl;
    return not_pass;
}
