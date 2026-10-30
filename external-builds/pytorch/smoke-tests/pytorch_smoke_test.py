# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import ctypes
import importlib
import importlib.util
import json
import logging
import math
import platform
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel


def _check_hip_error(hip: ctypes.CDLL, stage: str) -> None:
    # Peek, never clear: a failed AOTriton launch can otherwise go unnoticed.
    error = hip.hipPeekAtLastError()
    print(f"{stage}: hipPeekAtLastError={error}", flush=True)
    assert error == 0, f"HIP error {error} at {stage}"


def _run_sdpa_smoke_test(
    backend_name: str, shape: tuple[int, ...], use_bias: bool
) -> None:
    # Only needed in the isolated Windows worker, not on unrelated test hosts.
    import rocm_sdk
    from torch.utils._python_dispatch import TorchDispatchMode

    class TraceSDPA(TorchDispatchMode):
        def __init__(self) -> None:
            super().__init__()
            self.operators: set[str] = set()

        def __torch_dispatch__(
            self,
            func: Callable[..., object],
            types: tuple[type, ...],
            args: tuple = (),
            kwargs: dict | None = None,
        ) -> object:
            if "scaled_dot_product" in str(func):
                self.operators.add(str(func))
            return func(*args, **(kwargs or {}))

    assert (
        Path(torch.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    ), f"Expected installed torch in {sys.prefix}, got {torch.__file__}"
    print(
        f"torch={torch.__version__}, git={torch.version.git_version}, "
        f"HIP={torch.version.hip}, arch={torch.cuda.get_device_properties(0).gcnArchName}, "
        f"Python={sys.version}, platform={platform.platform()}, torch_file={torch.__file__}",
        flush=True,
    )
    (hip_path,) = rocm_sdk.find_libraries("amdhip64")
    hip = ctypes.CDLL(str(hip_path))
    hip.hipPeekAtLastError.argtypes = []
    hip.hipPeekAtLastError.restype = ctypes.c_int
    print(f"HIP runtime: {hip_path}", flush=True)

    generator = torch.Generator(device="cpu").manual_seed(7992)
    q, k, v = [
        torch.randn(shape, generator=generator).to(torch.bfloat16) for _ in range(3)
    ]
    bias = (
        (torch.randn((*shape[:-1], shape[-2]), generator=generator) * 0.1).to(
            torch.bfloat16
        )
        if use_bias
        else None
    )
    # Reference uses the exact BF16 inputs, independently on CPU in FP32.
    scores = (q.float() @ k.float().transpose(-2, -1)) / math.sqrt(shape[-1])
    if bias is not None:
        scores += bias.float()
    expected = torch.softmax(scores, dim=-1) @ v.float()
    q_gpu, k_gpu, v_gpu = [tensor.to("cuda") for tensor in (q, k, v)]
    bias_gpu = bias.to("cuda") if bias is not None else None
    torch.cuda.synchronize()
    _check_hip_error(hip, "before SDPA")
    with TraceSDPA() as trace, sdpa_kernel([getattr(SDPBackend, backend_name)]):
        actual = torch.nn.functional.scaled_dot_product_attention(
            q_gpu, k_gpu, v_gpu, attn_mask=bias_gpu, dropout_p=0.0
        )
    print(f"SDPA operators: {sorted(trace.operators)}", flush=True)
    _check_hip_error(hip, "after SDPA before synchronize")
    torch.cuda.synchronize()
    _check_hip_error(hip, "after SDPA synchronize")
    operator = (
        "aten._scaled_dot_product_flash_attention.default"
        if backend_name == "FLASH_ATTENTION"
        else "aten._scaled_dot_product_efficient_attention.default"
    )
    assert trace.operators == {
        operator
    }, f"Expected only {operator}, saw {trace.operators}"
    assert actual.shape == q_gpu.shape
    assert actual.dtype == torch.bfloat16
    assert actual.device == q_gpu.device
    downstream = actual + 1
    _check_hip_error(hip, "after downstream before synchronize")
    torch.cuda.synchronize()
    _check_hip_error(hip, "after downstream synchronize")
    actual_cpu = actual.cpu().float()
    assert torch.isfinite(actual_cpu).all()
    print(f"max_abs_error={(actual_cpu - expected).abs().max().item()}", flush=True)
    # BF16 tolerances validated on the historical broken/working wheel pair.
    torch.testing.assert_close(actual_cpu, expected, atol=0.03, rtol=0.03)
    torch.testing.assert_close(
        downstream.cpu().float(), expected + 1, atol=0.03, rtol=0.03
    )


@pytest.mark.skipif(platform.system() != "Windows", reason="Windows AOTriton packaging")
class TestSDPA:
    @pytest.mark.parametrize(
        "backend,shape,use_bias",
        [
            pytest.param("FLASH_ATTENTION", (1, 2, 16, 64), False, id="flash"),
            pytest.param(
                "EFFICIENT_ATTENTION", (1, 2, 16, 64), True, id="efficient-bias"
            ),
            pytest.param(
                "EFFICIENT_ATTENTION", (54, 12, 196, 64), True, id="vision-bias"
            ),
        ],
    )
    def test_fused_sdpa(
        self, backend: str, shape: tuple[int, ...], use_bias: bool
    ) -> None:
        if torch.version.hip is None:
            pytest.skip("UNSUPPORTED OUTSIDE SCOPE: requires ROCm")
        if not torch.cuda.is_available():
            pytest.skip("NOT RUN: no physical AMD GPU available")
        arch = torch.cuda.get_device_properties(0).gcnArchName
        if arch.split(":")[0] != "gfx1151":
            pytest.skip(
                f"UNSUPPORTED OUTSIDE SCOPE: Windows {arch}; validated on gfx1151"
            )
        # Sticky HIP errors must not contaminate another case. Inherit the
        # runner's device visibility and use this interpreter's installed wheel.
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import json, runpy, sys; "
                "runpy.run_path(sys.argv[1])['_run_sdpa_smoke_test']"
                "(*json.loads(sys.argv[2]))",
                str(Path(__file__).resolve()),
                json.dumps((backend, shape, use_bias)),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, (
            f"{backend} {shape}: worker exit {result.returncode}\n"
            f"{result.stdout}\n{result.stderr}"
        )
        print(result.stdout)


class TestROCmAvailability:
    def test_rocm_available(self):
        logging.basicConfig(level=logging.INFO)
        if torch.cuda.is_available():
            cnt_gpu = torch.cuda.device_count()
            logging.info("GPU count visible for pytorch: " + str(cnt_gpu))
            for ii in range(cnt_gpu):
                gpu_name = torch.cuda.get_device_name(ii)
                logging.info("GPU[" + str(ii) + "]: " + gpu_name)
        assert (
            torch.cuda.is_available()
        ), "ROCm is not available or not detected by PyTorch"


class TestMatrixOperations:
    def test_matrix_multiplication(self):
        matrix1 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="cuda")
        matrix2 = torch.tensor(
            [[7.0, 8.0, 9.0, 10.0], [11.0, 12.0, 13.0, 14.0], [15.0, 16.0, 17.0, 18.0]],
            device="cuda",
        )
        expected = torch.tensor(
            [[74.0, 80.0, 86.0, 92.0], [173.0, 188.0, 203.0, 218.0]], device="cuda"
        )
        result = torch.mm(matrix1, matrix2)
        assert torch.allclose(result, expected)
        assert result.device.type == "cuda"

    def test_batch_matrix_multiplication(self):
        batch_matrix1 = torch.ones(10, 2, 3, device="cuda")
        batch_matrix2 = torch.ones(10, 3, 4, device="cuda")
        expected = torch.full((10, 2, 4), 3.0, device="cuda")
        result = torch.bmm(batch_matrix1, batch_matrix2)
        assert torch.allclose(result, expected)
        assert result.device.type == "cuda"

    def test_matrix_multiplication_at_operator(self):
        matrix1 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="cuda")
        matrix2 = torch.tensor(
            [[7.0, 8.0, 9.0, 10.0], [11.0, 12.0, 13.0, 14.0], [15.0, 16.0, 17.0, 18.0]],
            device="cuda",
        )
        expected = torch.tensor(
            [[74.0, 80.0, 86.0, 92.0], [173.0, 188.0, 203.0, 218.0]], device="cuda"
        )
        result = matrix1 @ matrix2
        assert torch.allclose(result, expected)
        assert result.device.type == "cuda"

    def test_elementwise_multiplication(self):
        matrix1 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="cuda")
        matrix2 = torch.tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], device="cuda")
        expected = torch.tensor([[7.0, 16.0, 27.0], [40.0, 55.0, 72.0]], device="cuda")
        result = matrix1 * matrix2
        assert torch.allclose(result, expected)
        assert result.device.type == "cuda"

    def test_transpose(self):
        matrix = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="cuda")
        expected = torch.tensor([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]], device="cuda")
        transposed = torch.t(matrix)
        assert torch.allclose(transposed, expected)
        assert transposed.device.type == "cuda"

    def test_dot_product(self):
        vector1 = torch.tensor([1.0, 2.0, 3.0], device="cuda")
        vector2 = torch.tensor([4.0, 5.0, 6.0], device="cuda")
        expected = torch.tensor(32.0, device="cuda")
        result = torch.dot(vector1, vector2)
        assert torch.allclose(result, expected)
        assert result.device.type == "cuda"

    def test_matrix_vector_multiplication(self):
        matrix = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="cuda")
        vector = torch.tensor([7.0, 8.0, 9.0], device="cuda")
        expected = torch.tensor([50.0, 122.0], device="cuda")
        result = torch.mv(matrix, vector)
        assert torch.allclose(result, expected)
        assert result.device.type == "cuda"

    def test_matrix_multiplication_matmul(self):
        matrix1 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="cuda")
        matrix2 = torch.tensor(
            [[7.0, 8.0, 9.0, 10.0], [11.0, 12.0, 13.0, 14.0], [15.0, 16.0, 17.0, 18.0]],
            device="cuda",
        )
        expected = torch.tensor(
            [[74.0, 80.0, 86.0, 92.0], [173.0, 188.0, 203.0, 218.0]], device="cuda"
        )
        result = torch.matmul(matrix1, matrix2)
        assert torch.allclose(result, expected)
        assert result.device.type == "cuda"


class TestConvolutions:
    def teardown_method(self):
        # TODO(#999): fix tests stalling on exit without this
        torch.cuda.synchronize()

    def test_conv_transpose2d(self):
        inputs = torch.randn(1, 4, 5, 5, device="cuda")
        weights = torch.randn(4, 8, 3, 3, device="cuda")
        # Simply running any conv op exercises MIOpen and library loading.
        # On Windows, this may fail if `amd_comgr_3.dll` (from build output) is
        # used instead of `amd_comgr0605.dll` that is expected at runtime.
        result = torch.nn.functional.conv_transpose2d(inputs, weights, padding=1)

        # TODO: check conv output values (and don't use randn)
        assert result.device.type == "cuda"

    # Lifted from
    # https://github.com/pytorch/pytorch/blob/main/test/nn/test_convolution.py
    def test_conv_cudnn_nhwc_support(self):
        input = torch.randn(
            (1, 16, 1, 1), dtype=torch.float, device="cuda", requires_grad=True
        )
        weight = torch.randn(
            (8, 16, 3, 3), dtype=torch.float, device="cuda", requires_grad=True
        )
        weight = weight.to(memory_format=torch.channels_last)
        o = torch.conv2d(input, weight, None, (2, 1), (1, 1), (1, 1), 1)
        assert o.is_contiguous(memory_format=torch.channels_last)


class TestOpenBLASAvailability:
    def test_openblas_is_selected_blas(self):
        cfg_text = torch.__config__.show().lower()
        patterns = [
            r"blas_info=open",
        ]
        assert any(
            re.search(p, cfg_text) for p in patterns
        ), "OpenBLAS is not available or not detected by PyTorch"

    def test_config_indicates_lapack_enabled(self):
        cfg_text = torch.__config__.show().lower()
        patterns = [
            r"lapack is enabled",
        ]
        assert any(
            re.search(p, cfg_text) for p in patterns
        ), "LAPACK is not available or not detected by PyTorch"

    # Implementation of svd on CPU uses LAPACK:
    # See https://docs.pytorch.org/docs/stable/generated/torch.svd.html
    def test_lapack_available_svd(self):
        input = torch.randn(50, 50, device="cpu")
        output_u, output_s, output_vh = torch.linalg.svd(input)
        assert output_u.device == torch.device("cpu")
        assert output_s.device == torch.device("cpu")
        assert output_vh.device == torch.device("cpu")


@pytest.mark.skipif(platform.system() != "Linux", reason="Linux only")
class TestRocmSdkLibraries:
    """Verify that import torch preloads ROCm libraries with RTLD_GLOBAL on Linux.

    import torch triggers _rocm_init.py (injected into the torch wheel by TheRock)
    which calls rocm_sdk.initialize_process, loading each library in
    LINUX_LIBRARY_PRELOADS with RTLD_GLOBAL. This makes their symbols available
    via dlsym(RTLD_DEFAULT, ...) so native code can resolve them without needing
    dlopen by unversioned name (which fails in wheel installs where only the
    versioned .so exists). See ROCM-27833.

    Libraries tested:
      - rocm-sdk-core libraries (amd_smi, amdhip64): always installed with torch.
      - rocm-sdk-libraries (hipblas): installed as a torch dependency but not
        guaranteed in all configurations (e.g. narrow installs). Skipped when
        rocm_sdk_libraries is not importable.
    """

    # Symbols from rocm-sdk-core — always installed with torch, no skip needed.
    _CORE_PRELOAD_SYMBOLS = [
        (
            "amd_smi",
            "amdsmi_init",
        ),  # ROCM-27833: dlopen by name fails in wheel installs
        ("amd_smi", "amdsmi_get_socket_handles"),
        ("amd_smi", "amdsmi_get_processor_handles"),
        ("amd_smi", "amdsmi_is_P2P_accessible"),
        ("amdhip64", "hipGetDeviceCount"),  # HIP runtime — core dependency of torch
    ]

    # Symbols from rocm-sdk-libraries — installed as a torch dependency but may
    # be absent in narrow configurations (e.g. rocm[core] only). Skipped when
    # rocm_sdk_libraries is not importable so narrow installs don't fail here.
    _LIBRARIES_PRELOAD_SYMBOLS = [
        ("hipblas", "hipblasCreate"),
    ]

    @pytest.mark.parametrize("lib,symbol", _CORE_PRELOAD_SYMBOLS)
    def test_core_preloaded_symbol_resolvable_via_rtld_default(self, lib, symbol):
        # ctypes.CDLL(None) is the Python equivalent of dlsym(RTLD_DEFAULT, ...).
        # Instantiated here (not as a class attribute) so it is only evaluated at
        # test execution time — after the class-level skipif has taken effect —
        # avoiding a TypeError on Windows where ctypes.CDLL(None) returns None.
        rtld_default = ctypes.CDLL(None)
        fn = getattr(rtld_default, symbol, None)
        addr = ctypes.cast(fn, ctypes.c_void_p).value
        assert addr, (
            f"dlsym(RTLD_DEFAULT, '{symbol}') returned NULL — "
            f"'{lib}' was not preloaded with RTLD_GLOBAL by _rocm_init.py"
        )

    @pytest.mark.skipif(
        # rocm-sdk-libraries is installed as a torch dependency in normal
        # configurations, but may be absent in narrow installs (e.g. rocm[core]
        # only). Skip rather than fail so narrow configs stay green.
        importlib.util.find_spec("rocm_sdk_libraries") is None,
        reason="rocm-sdk-libraries not installed",
    )
    @pytest.mark.parametrize("lib,symbol", _LIBRARIES_PRELOAD_SYMBOLS)
    def test_libraries_preloaded_symbol_resolvable_via_rtld_default(self, lib, symbol):
        rtld_default = ctypes.CDLL(None)
        fn = getattr(rtld_default, symbol, None)
        addr = ctypes.cast(fn, ctypes.c_void_p).value
        assert addr, (
            f"dlsym(RTLD_DEFAULT, '{symbol}') returned NULL — "
            f"'{lib}' was not preloaded with RTLD_GLOBAL by _rocm_init.py"
        )
