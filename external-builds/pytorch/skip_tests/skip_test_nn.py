skip_tests = {
    "always": [
        # external-builds/pytorch/pytorch/test/test_nn.py::TestNN::test_RNN_dropout_state MIOpen(HIP): Error [Compile] 'hiprtcCompileProgram(prog.get(), c_options.size(), c_options.data())' MIOpenDropoutHIP.cpp: HIPRTC_ERROR_COMPILATION (6)
        # MIOpen(HIP): Error [BuildHip] HIPRTC status = HIPRTC_ERROR_COMPILATION (6), source file: MIOpenDropoutHIP.cpp
        # MIOpen(HIP): Warning [BuildHip] In file included from /tmp/comgr-01c423/input/MIOpenDropoutHIP.cpp:32:
        # /tmp/comgr-01c423/include/miopen_rocrand.hpp:45:10: fatal error: 'rocrand/rocrand_xorwow.h' file not found
        # 45 | #include <rocrand/rocrand_xorwow.h>
        #     |          ^~~~~~~~~~~~~~~~~~~~~~~~~~
        # 1 error generated when compiling for gfx942.
        # MIOpen Error: /therock/src/rocm-libraries/projects/miopen/src/hipoc/hipoc_program.cpp:299: Code object build failed. Source: MIOpenDropoutHIP.cpp
        "test_RNN_dropout_state",
        "test_rnn_check_device",
    ],
    "pytorch_version": {
        "2.10": [
            # RuntimeError: miopenStatusUnknownError
            "test_side_stream_backward_overlap"
        ]
    },
    "amdgpu_family": {},
}
