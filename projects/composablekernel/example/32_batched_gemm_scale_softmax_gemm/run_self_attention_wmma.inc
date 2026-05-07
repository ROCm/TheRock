// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

using Row    = ck::tensor_layout::gemm::RowMajor;
using Col    = ck::tensor_layout::gemm::ColumnMajor;
using Bypass = ck::tensor_layout::BypassLayoutVerification;

int run(int argc, char* argv[])
{
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;

    // GEMM shape for A/B0/B1/C
    // C_g_m_o = A_g_m_k * B0_g_k_n * B1_g_n_o
    ck::index_t sequence_length = 256;
    ck::index_t head_dim        = 80;

    // Output shape C[batch_size, sequence_length, head_num, head_dim]. Batch dim, outer dim, inner
    // dim must match GEMM shape C_g0_g1_m_o = reshape(C_g_m_o, [g0, g1, m, o]) C_g0_m_g1_o =
    // permute(C_g0_g1_m_o, [0, 2, 1, 3])
    ck::index_t batch_size = 2;
    ck::index_t head_num   = 8;

    float alpha         = 1;
    bool input_permute  = true;
    bool output_permute = true;

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
    else if(argc == 9)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);

        sequence_length = std::stoi(argv[4]);
        head_dim        = std::stoi(argv[5]);
        batch_size      = std::stoi(argv[6]);
        head_num        = std::stoi(argv[7]);

        alpha = std::stof(argv[8]);
    }
    else
    {
        printf("arg1: verification (0=no, 1=yes)\n");
        printf("arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n");
        printf("arg3: time kernel (0=no, 1=yes)\n");
        printf("arg4 to 7: sequence_length, head_dim, batch_size, head_num\n");
        printf("arg8: scale (alpha)\n");
        exit(0);
    }

    std::vector<ck::index_t> a_gs_ms_ks_lengths{batch_size, head_num, sequence_length, head_dim};
    std::vector<ck::index_t> a_gs_ms_ks_strides =
        input_permute ? std::vector<ck::index_t>{sequence_length * head_num * head_dim,
                                                 head_dim,
                                                 head_num * head_dim,
                                                 1}
                      // A layout [batch_size, sequence_length, head_num, head_dim]
                      : std::vector<ck::index_t>{
                            head_num * sequence_length * head_dim,
                            sequence_length * head_dim,
                            head_dim,
                            1}; // A layout [batch_size, head_num, sequence_length, head_dim]

    std::vector<ck::index_t> b0_gs_ns_ks_lengths{batch_size, head_num, sequence_length, head_dim};
    std::vector<ck::index_t> b0_gs_ns_ks_strides =
        input_permute ? std::vector<ck::index_t>{sequence_length * head_num * head_dim,
                                                 head_dim,
                                                 head_num * head_dim,
                                                 1}
                      // B0 layout [batch_size, sequence_length, head_num, head_dim]
                      : std::vector<ck::index_t>{
                            head_num * sequence_length * head_dim,
                            sequence_length * head_dim,
                            head_dim,
                            1}; // B0 layout [batch_size, head_num, sequence_length, head_dim]

    std::vector<ck::index_t> b1_gs_os_ns_lengths{batch_size, head_num, head_dim, sequence_length};
    std::vector<ck::index_t> b1_gs_os_ns_strides =
        input_permute
            ? std::vector<ck::index_t>{sequence_length * head_num * head_dim,
                                       head_dim,
                                       1,
                                       head_num * head_dim}
            // B1 layout [batch_size, sequence_length, head_num, head_dim]
            : std::vector<ck::index_t>{
                  head_num * sequence_length * head_dim,
                  sequence_length * head_dim,
                  1,
                  head_dim}; // B1 layout [batch_size, head_num, sequence_length, head_dim]

    std::vector<ck::index_t> c_gs_ms_os_lengths{batch_size, head_num, sequence_length, head_dim};
    std::vector<ck::index_t> c_gs_ms_os_strides =
        output_permute ? std::vector<ck::index_t>{sequence_length * head_num * head_dim,
                                                  head_dim,
                                                  head_num * head_dim,
                                                  1}
                       // C layout [batch_size, sequence_length, head_num, head_dim]
                       : std::vector<ck::index_t>{
                             head_num * sequence_length * head_dim,
                             sequence_length * head_dim,
                             head_dim,
                             1}; // C layout [batch_size, head_num, sequence_length, head_dim]

    auto f_host_tensor_descriptor = [](std::vector<ck::index_t> lens,
                                       std::vector<ck::index_t> strides,
                                       bool permute,
                                       auto layout) {
        if(permute)
        {
            return HostTensorDescriptor(lens, strides, Bypass{});
        }
        else
        {
            return HostTensorDescriptor(lens, strides, layout);
        }
    };

    Tensor<ADataType> a_gs_ms_ks(
        f_host_tensor_descriptor(a_gs_ms_ks_lengths, a_gs_ms_ks_strides, input_permute, Row{}));
    Tensor<B0DataType> b0_gs_ns_ks(
        f_host_tensor_descriptor(b0_gs_ns_ks_lengths, b0_gs_ns_ks_strides, input_permute, Row{}));
    Tensor<B1DataType> b1_gs_os_ns(
        f_host_tensor_descriptor(b1_gs_os_ns_lengths, b1_gs_os_ns_strides, input_permute, Col{}));
    Tensor<CDataType> c_gs_ms_os_host_result(
        f_host_tensor_descriptor(c_gs_ms_os_lengths, c_gs_ms_os_strides, output_permute, Row{}));
    Tensor<CDataType> c_gs_ms_os_device_result(
        f_host_tensor_descriptor(c_gs_ms_os_lengths, c_gs_ms_os_strides, output_permute, Row{}));

    std::cout << "a_gs_ms_ks: " << a_gs_ms_ks.mDesc << std::endl;
    std::cout << "b0_gs_ns_ks: " << b0_gs_ns_ks.mDesc << std::endl;
    std::cout << "b1_gs_os_ns: " << b1_gs_os_ns.mDesc << std::endl;
    std::cout << "c_gs_ms_os: " << c_gs_ms_os_host_result.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        a_gs_ms_ks.GenerateTensorValue(GeneratorTensor_2<ADataType>{-2, 2});
        b0_gs_ns_ks.GenerateTensorValue(GeneratorTensor_2<B0DataType>{-2, 2});
        b1_gs_os_ns.GenerateTensorValue(GeneratorTensor_2<B1DataType>{-2, 2});
        break;
    case 2:
        a_gs_ms_ks.GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0});
        b0_gs_ns_ks.GenerateTensorValue(GeneratorTensor_3<B0DataType>{0.0, 1.0});
        b1_gs_os_ns.GenerateTensorValue(GeneratorTensor_3<B1DataType>{-0.5, 0.5});
        break;
    case 3:
        a_gs_ms_ks.GenerateTensorValue(GeneratorTensor_2<ADataType>{-2, 2});
        b0_gs_ns_ks.GenerateTensorValue(GeneratorTensor_Diagonal<B0DataType>{});
        b1_gs_os_ns.GenerateTensorValue(GeneratorTensor_Diagonal<B1DataType>{});
        break;
    case 4: // A, B0, B1 1
        a_gs_ms_ks.GenerateTensorValue(GeneratorTensor_1<ADataType>{});
        b0_gs_ns_ks.GenerateTensorValue(GeneratorTensor_1<B0DataType>{});
        b1_gs_os_ns.GenerateTensorValue(GeneratorTensor_1<B1DataType>{});
        break;
    case 5: // Rand: b1 b0; unit: a
        a_gs_ms_ks.GenerateTensorValue(GeneratorTensor_1<ADataType>{});
        b0_gs_ns_ks.GenerateTensorValue(GeneratorTensor_2<B0DataType>{-2, 2});
        b1_gs_os_ns.GenerateTensorValue(GeneratorTensor_2<B1DataType>{-2, 2});
        break;
    case 6: // Rand: a b0 ; unit: B1
        a_gs_ms_ks.GenerateTensorValue(GeneratorTensor_2<ADataType>{-2, 2});
        b0_gs_ns_ks.GenerateTensorValue(GeneratorTensor_2<B0DataType>{-2, 2});
        b1_gs_os_ns.GenerateTensorValue(GeneratorTensor_1<B1DataType>{});
        break;
    case 7: // Rand: a b1 ; unit: b0
        a_gs_ms_ks.GenerateTensorValue(GeneratorTensor_2<ADataType>{-2, 2});
        b0_gs_ns_ks.GenerateTensorValue(GeneratorTensor_1<B0DataType>{});
        b1_gs_os_ns.GenerateTensorValue(GeneratorTensor_2<B1DataType>{-2, 2});
        break;
    case 8: // Rand: a ; unit: b0 b1
        a_gs_ms_ks.GenerateTensorValue(GeneratorTensor_2<ADataType>{-2, 2});
        b0_gs_ns_ks.GenerateTensorValue(GeneratorTensor_1<B0DataType>{});
        b1_gs_os_ns.GenerateTensorValue(GeneratorTensor_1<B1DataType>{});
        break;
    case 9: // Rand: b0 ; unit: a b1
        a_gs_ms_ks.GenerateTensorValue(GeneratorTensor_1<ADataType>{});
        b0_gs_ns_ks.GenerateTensorValue(GeneratorTensor_2<B0DataType>{-2, 2});
        b1_gs_os_ns.GenerateTensorValue(GeneratorTensor_1<B1DataType>{});
        break;
    case 10: // Rand: b1 ; unit: a b0
        a_gs_ms_ks.GenerateTensorValue(GeneratorTensor_1<ADataType>{});
        b0_gs_ns_ks.GenerateTensorValue(GeneratorTensor_1<B0DataType>{});
        b1_gs_os_ns.GenerateTensorValue(GeneratorTensor_2<B1DataType>{-2, 2});
        break;
    default:
        a_gs_ms_ks.GenerateTensorValue(GeneratorTensor_Sequential<ADataType, 2>{});
        b0_gs_ns_ks.GenerateTensorValue(GeneratorTensor_Diagonal<B0DataType>{});
        b1_gs_os_ns.GenerateTensorValue(GeneratorTensor_Diagonal<B1DataType>{});
    }

    std::vector<ck::index_t> qkv_gs_ms_ks_lengths{
        batch_size, head_num, sequence_length, 3, head_dim};
    std::vector<ck::index_t> qkv_gs_ms_ks_strides = std::vector<ck::index_t>{
        sequence_length * head_num * 3 * head_dim,
        3 * head_dim,
        head_num * 3 * head_dim,
        head_dim,
        1}; // qkv layout [batch_size, sequence_length, head_num, 3, head_dim]
    Tensor<ADataType> qkv_gs_ms_ks(qkv_gs_ms_ks_lengths, qkv_gs_ms_ks_strides, Bypass{});
    // merge qkv into a packed pointer send to device
    a_gs_ms_ks.ForEach(
        [&](auto& self, auto idx) { qkv_gs_ms_ks(idx[0], idx[1], idx[2], 0, idx[3]) = self(idx); });
    b0_gs_ns_ks.ForEach(
        [&](auto& self, auto idx) { qkv_gs_ms_ks(idx[0], idx[1], idx[2], 1, idx[3]) = self(idx); });
    b1_gs_os_ns.ForEach(
        [&](auto& self, auto idx) { qkv_gs_ms_ks(idx[0], idx[1], idx[3], 2, idx[2]) = self(idx); });
    DeviceMem qkv_device_buf(sizeof(ADataType) * a_gs_ms_ks.mDesc.GetElementSpaceSize() +
                             sizeof(B0DataType) * b0_gs_ns_ks.mDesc.GetElementSpaceSize() +
                             sizeof(B1DataType) * b1_gs_os_ns.mDesc.GetElementSpaceSize());
    DeviceMem c_device_buf(sizeof(CDataType) *
                           c_gs_ms_os_device_result.mDesc.GetElementSpaceSize());
    qkv_device_buf.ToDevice(qkv_gs_ms_ks.mData.data());

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
    // TODO ANT: replace array with vector?
    ck::static_for<0, std::tuple_size_v<DeviceMHAFactory>, 1>{}([&](auto i) -> void {
        const auto device_mha_instance = std::get<i>(DeviceMHAFactory{});

        using DeviceMHAInstance = ck::remove_cvref_t<decltype(device_mha_instance)>;
        auto gemm               = DeviceMHAInstance{};
        auto invoker            = gemm.MakeSelfAttnInvoker();
        auto argument =
            gemm.MakeSelfAttnArgument(static_cast<ADataType*>(qkv_device_buf.GetDeviceBuffer()),
                                      static_cast<CDataType*>(c_device_buf.GetDeviceBuffer()),
                                      batch_size,
                                      sequence_length,
                                      head_num,
                                      head_dim,
                                      alpha);

        // if(!gemm.IsSupportedArgument(argument))
        // {
        //     std::cout << gemm.GetTypeString() << " does not support this problem" << std::endl;

        //     return 0;
        // }

        ck::index_t BatchCount = batch_size * head_num;

        float ave_time = invoker.Run(argument, StreamConfig{nullptr, time_kernel});

        std::size_t flop = (size_t(sequence_length) * sequence_length * head_dim * 2 +
                            size_t(sequence_length) * sequence_length * head_dim * 2) *
                           BatchCount;
        std::size_t num_btype = (sizeof(ADataType) * sequence_length * head_dim +
                                 sizeof(B0DataType) * head_dim * sequence_length +
                                 sizeof(B1DataType) * sequence_length * head_dim +
                                 sizeof(CDataType) * sequence_length * head_dim) *
                                BatchCount;

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
            c_device_buf.FromDevice(c_gs_ms_os_device_result.mData.data());

            Tensor<ADataType> a_g_m_k({BatchCount, sequence_length, head_dim});
            Tensor<B0DataType> b0_g_k_n({BatchCount, head_dim, sequence_length});
            Tensor<B1DataType> b1_g_n_o({BatchCount, sequence_length, head_dim});
            Tensor<Acc0DataType> acc0_g_m_n(
                {BatchCount, sequence_length, sequence_length}); // scratch object after gemm0
            Tensor<ADataType> a1_g_m_n(
                {BatchCount, sequence_length, sequence_length}); // scratch object after softmax
            Tensor<CDataType> c_g_m_o_host_result(
                {BatchCount, sequence_length, head_dim}); // scratch object after gemm1

            // permute
            a_gs_ms_ks.ForEach([&](auto& self, auto idx) {
                a_g_m_k(idx[0] * head_num + idx[1], idx[2], idx[3]) = self(idx);
            });
            b0_gs_ns_ks.ForEach([&](auto& self, auto idx) {
                b0_g_k_n(idx[0] * head_num + idx[1], idx[3], idx[2]) = self(idx);
            });
            b1_gs_os_ns.ForEach([&](auto& self, auto idx) {
                b1_g_n_o(idx[0] * head_num + idx[1], idx[3], idx[2]) = self(idx);
            });

            // gemm 0
            auto ref_gemm0          = ReferenceGemm0Instance{};
            auto ref_gemm0_invoker  = ref_gemm0.MakeInvoker();
            auto ref_gemm0_argument = ref_gemm0.MakeArgument(
                a_g_m_k, b0_g_k_n, acc0_g_m_n, a_element_op, b0_element_op, acc0_element_op);

            ref_gemm0_invoker.Run(ref_gemm0_argument);

            // masking
            const auto mask = typename DeviceMHAInstance::C0MatrixMask(sequence_length);
            acc0_g_m_n.ForEach([&](auto& self, auto idx) {
                if(mask.IsMaskedElement(idx[1], idx[2]))
                    self(idx) = -ck::NumericLimits<float>::Infinity();
            });

            // softmax
            auto ref_softmax          = ReferenceSoftmaxInstance{};
            auto ref_softmax_invoker  = ref_softmax.MakeInvoker();
            auto ref_softmax_argument = ref_softmax.MakeArgument(acc0_g_m_n, a1_g_m_n, 1, 0, {2});

            ref_softmax_invoker.Run(ref_softmax_argument);

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

            // permute
            c_gs_ms_os_host_result.ForEach([&](auto& self, auto idx) {
                const size_t& g0 = idx[0];
                const size_t& g1 = idx[1];

                const size_t g = g0 * head_num + g1;

                self(idx) = c_g_m_o_host_result(g, idx[2], idx[3]);
            });

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

            bool this_run_verification = ck::utils::check_err(c_gs_ms_os_device_result.mData,
                                                              c_gs_ms_os_host_result.mData,
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
    std::cout << "Problem Size: BatchCount: " << batch_size << ", HeadNum: " << head_num
              << ", sequence_length: " << sequence_length << ", head_dim: " << head_dim
              << std::endl;
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
