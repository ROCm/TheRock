from typing import Literal, get_args

from utils import *

# ROCm software should update contents as phonebook does.
ROC_X_LIBS_TYPEHINT = Literal[
    "AMD-LLVM",
    "amd_comgr",
    "composable-kernel",
    "hip",
    "hip-lang",
    "hip-other",
    "hiprtc",
    "hipBLAS",
    "hipBLAS-common",
    "hipBLASLt",
    "hipCUB",
    "hipDNN",
    "hipIFY",
    "hipFFT",
    "hipRAND",
    "hipSOLVER",
    "hipSPARSE",
    "hipSPARSELt",
    "libhipcxx",
    "MIOpen",
    "mxDataGenerator",
    "rocm-core",
    "rocm-cmake",
    "rocm-kpack",
    "rccl",
    "rdc",
    "rocALUTION",
    "rocCLR",
    "rocBLAS",
    "rocFFT",
    "rocGDB",
    "rocRAND",
    "rocPRIM",
    "rocProfiler",
    "rocSPARSE",
    "rocSOLVER",
    "rocThrust",
    "rocTracer",
    "rocWMMA",
    "TheRock",
]

rocX_config_version_cmake_phonebook: dict[ROC_X_LIBS_TYPEHINT, str] = {
    # For NOTDEFINED/Wrong cmake versioning configure file
    #   "rocX-config-version.cmake" should fix.
    "therock": r".info/version",
    # https://github.com/ROCm/llvm-project/tree/amd-staging/amd/comgr
    "amd_comgr": r"lib/cmake/amd_comgr/amd_comgr-config-version.cmake",
    # https://github.com/ROCm/llvm-project
    "amd-llvm": r"Same as AMD-LLVM",
    # https://github.com/ROCm/llvm-project/tree/amd-staging/amd/hipcc
    "hip": r"lib/cmake/hip/hip-config-version.cmake",
    "hip-lang": r"lib/cmake/hip-lang/hip-lang-config-verson.cmake",
    "hiprtc": r"lib/cmake/hiprtc/hiprtc-config-version.cmake",
    # https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipblas
    "hipblas": r"lib/cmake/hipblas/hipblas-config-version.cmake",
    # https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipblas-common
    "hipblas-common": r"lib/cmake/hipblas-common/hipblas-common-config-version.cmake",
    # https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipblaslt
    "hipblaslt": r"lib/cmake/hipblaslt/hipblaslt-config-version.cmake",
    # https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipcub
    "hipcub": r"lib/cmake/hipcub/hipcub-config-version.cmake",
    # https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipdnn
    "hipdnn": r"lib/cmake/hipdnn_data_sdk/hipdnn_data_sdkConfigVersion.cmake",
    # https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipfft
    "hipfft": r"lib/cmake/hipfft/hipfft-config-version.cmake",
    # https://github.com/ROCm/HIPIFY
    "hipify": r"Same as AMD-LLVM",
    # https://github.com/ROCm/rocm-libraries/tree/develop/projects/hiprand
    "hiprand": r"lib/cmake/hiprand/hiprand-config-version.cmake",
    # https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipsolver
    "hipsolver": r"lib/cmake/hipsolver/hipsolver-config-version.cmake",
    # https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipsparse
    "hipsparse": r"lib/cmake/hipsparse/hipsparse-config-version.cmake",
    # https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipsparselt
    "hipsparselt": r"lib/cmake/hipsparselt/hipsparselt-config-version.cmake",
    # https://github.com/ROCm/rocm-libraries/tree/develop/projects/hiptensor
    "hiptensor": r"lib/cmake/hipsparselt/hiptensor-config-version.cmake",
    # https://github.com/ROCm/libhipcxx
    "libhipcxx": r"lib/cmake/libhipcxx/libhipcxx-config-version.cmake",
    # https://github.com/ROCm/rocm-libraries/tree/develop/projects/miopen
    "miopen": r"lib/cmake/miopen/miopen-config-version.cmake",
    # https://github.com/ROCm/rocm-libraries/tree/develop/shared/mxdatagenerator
    "mxdatagenerator": r"lib/cmake/mxDataGenerator/mxDataGeneratorConfig-version.cmake",
    # https://github.com/ROCm/rocm-systems/tree/develop/projects/rocm-core
    "rocm-core": r"lib/cmake/rocm-core/rocm-core-config-version.cmake",
    # https://github.com/ROCm/rocm-cmake
    "rocm-cmake": r"share/rocmcmakebuildtools/cmake/ROCMCMakeBuildToolsConfigVersion.cmake",
    # https://github.com/ROCm/rocm-kpack
    "rocm-kpack": r"lib/cmake/rocm-kpack/rocm-kpack-config-version.cmake",
    # https://github.com/ROCm/rocALUTION
    "rocalution": r"lib/cmake/rocalution/rocalution-config-version.cmake",
    # https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocblas
    "rocblas": r"lib/cmake/rocblas/rocblas-config-version.cmake",
    # https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocfft
    "rocfft": r"lib/cmake/rocfft/rocfft-config-version.cmake",
    # https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocprim
    "rocprim": r"lib/cmake/rocprim/rocprim-config-version.cmake",
    # https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocrand
    "rocrand": r"lib/cmake/rocrand/rocrand-config-version.cmake",
    # https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocsolver
    "rocsolver": r"lib/cmake/rocsolver/rocsolver-config-version.cmake",
    # https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocsparse
    "rocsparse": r"lib/cmake/rocsparse/rocsparse-config-version.cmake",
    # https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocthrust
    "rocthrust": r"lib/cmake/rocthrust/rocthrust-config-version.cmake",
    # https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocwmma
    "rocwmma": r"lib/cmake/rocwmma/rocwmma-config-version.cmake",
    #
    #   (@TaiXeflar: actually idk where the config file or executable is    Orz)
    #   Additional projects will/could/should add into ROCm/TheRock integration.
    #
    #       Pending for analyzing that need find from other way.
    #       Pending for integration to rocm-systems/rocm-libraries or TheRock build.
    #       Pending for examine for next commit to remove/adjust.
    #
    #   If project is not found, will return NOTDEFINED/NOTFOUND for FindROCm class uses.
    #
    # https://github.com/ROCm/rocm-libraries/tree/develop/projects/composablekernel
    "composable-kernel": None,
    # https://github.com/ROCm/rocm-systems/tree/develop/projects/hipother
    "hip-other": None,
    # https://github.com/ROCm/rocm-systems/tree/develop/projects/clr
    "rocclr": None,
    # https://github.com/ROCm/rocm-systems/tree/develop/projects/rccl,
    "rccl": None,
    # https://github.com/ROCm/rocm-systems/tree/develop/projects/rocr-runtime
    "rocr-runtime": None,
    # https://github.com/ROCm/rocm-systems/tree/develop/projects/rdc
    "rdc": None,
    # https://github.com/ROCm/ROCgdb
    "rocgdb": r"bin/rocgdb",
    # https://github.com/ROCm/rocm-systems/tree/develop/projects/rocprofiler
    "rocprofiler": r"lib/cmake/rocprofiler/rocprofiler-config-version.cmake",
    # https://github.com/ROCm/rocm-libraries/tree/develop/shared/rocroller
    "rocroller": r"lib/cmake/rocroller/rocroller-config-version.cmake",
    # https://github.com/ROCm/rocm-systems/tree/develop/projects/roctracer
    "roctracer": r"lib/cmake/roctracer/roctracer-config-version.cmake",
}


ROC_X_LIBRARIES_LIST: ROC_X_LIBS_TYPEHINT = list(get_args(ROC_X_LIBS_TYPEHINT))


def _test_phonebook():
    _test_phonebook_list = list(rocX_config_version_cmake_phonebook.keys()).sort()
    _test_typehint_list = [rocX.lower() for rocX in ROC_X_LIBRARIES_LIST].sort()

    if _test_phonebook_list != _test_typehint_list:
        message("FATAL_ERROR", f"""Found ROCm Libraries record have error.""")


_test_phonebook()
