// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

int run_gemm_test(int argc, char* argv[])
{
    using Row = ck::tensor_layout::gemm::RowMajor;
    using Col = ck::tensor_layout::gemm::ColumnMajor;

    using PassThrough = ck::tensor_operation::element_wise::PassThrough;
    ck::gemm_util::GemmParams params;
    ck::index_t instance_index = -1;
    if(argc == 1)
    {
        // use default params
    }
    else if(argc == 4 || argc == 5)
    {
        params.M       = atoi(argv[1]);
        params.N       = atoi(argv[2]);
        params.K       = atoi(argv[3]);
        params.StrideA = params.M;
        params.StrideB = params.N;
        params.StrideC = params.K;

        if(argc == 5)
        {
            instance_index = atoi(argv[4]);
        }
    }
    else
    {
        std::cout << "Usage of " << argv[0] << std::endl;
        std::cout << "Arg1-4: M N K instance_index(-1 means all)" << std::endl;
    }
    std::cout << "Params (M, N, K, index) " << params.M << " " << params.N << " " << params.K << " "
              << instance_index << std::endl;
    auto test = [&](auto a_layout, auto b_layout, auto c_layout) {
        bool pass = true;

        using DeviceOp = ck::tensor_operation::device::DeviceGemm<decltype(a_layout),
                                                                  decltype(b_layout),
                                                                  decltype(c_layout),
                                                                  ADataType,
                                                                  BDataType,
                                                                  CDataType,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>;

        const auto gemmPtrs =
            ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
                DeviceOp>::GetInstances();
        ck::index_t num_instance = 0;
        for(auto& gemmPtr : gemmPtrs)
        {
            if(instance_index == -1)
            {
                pass &= ck::gemm_util::TestGemm<AccDataType>{}(gemmPtr.get(), params);
            }
            else
            {
                auto test_gemm = ck::gemm_util::TestGemm<AccDataType>{};
                if(test_gemm.IsSupportedArgument(gemmPtr.get(), params))
                {
                    if(num_instance == instance_index)
                    {
                        pass &= test_gemm(gemmPtr.get(), params);
                    }
                    num_instance++;
                }
            }
        }

        if(instance_index != -1)
        {
            std::cout << "TestGemm_instance (" << instance_index << "/" << num_instance
                      << "): " << (pass ? "Passed" : "Failed") << std::endl;
        }

        return pass;
    };

    bool pass = test(Row{}, Row{}, Row{}) && test(Row{}, Col{}, Row{}) &&
                test(Col{}, Row{}, Row{}) && test(Col{}, Col{}, Row{});

    std::cout << "TestGemm ..... " << (pass ? "SUCCESS" : "FAILURE") << std::endl;
    return pass ? 0 : 1;
}
