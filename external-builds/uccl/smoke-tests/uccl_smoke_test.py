# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import logging
import pytest
import torch


class TestROCmAvailability:
    def test_rocm_available(self):
        logging.basicConfig(level=logging.INFO)
        if torch.cuda.is_available():
            cnt_gpu = torch.cuda.device_count()
            logging.info(f"GPU count visible for pytorch: {cnt_gpu}")
            for ii in range(cnt_gpu):
                gpu_name = torch.cuda.get_device_name(ii)
                logging.info(f"GPU[{ii}]: {gpu_name}")
        assert (
            torch.cuda.is_available()
        ), "ROCm is not available or not detected by PyTorch"


class TestUCCLImport:
    def test_import_uccl(self):
        import uccl  # noqa: F401

    def test_import_uccl_ep(self):
        # UCCL's upstream build.sh currently skips the EP (Expert Parallelism)
        # build for the "therock" target (see build_inner.sh: "Skipping
        # GPU-driven build on therock (no GPU-driven support yet)"). Until
        # upstream enables EP for the therock target, uccl.ep will not be
        # present in wheels built by our CI, so we skip instead of failing.
        pytest.importorskip(
            "uccl.ep",
            reason="uccl.ep is not built for the 'therock' target by upstream UCCL",
        )
        from uccl.ep import Config  # noqa: F401


class TestBasicGPUTransfer:
    def test_gpu_tensor_creation(self):
        t = torch.ones(4, 4, device="cuda")
        assert t.device.type == "cuda"
        assert torch.all(t == 1.0)

    def test_gpu_to_gpu_copy(self):
        if torch.cuda.device_count() < 2:
            pytest.skip("Need at least 2 GPUs for GPU-to-GPU copy test")
        src = torch.randn(64, 64, device="cuda:0")
        dst = src.to("cuda:1")
        assert dst.device == torch.device("cuda", 1)
        assert torch.allclose(src.cpu(), dst.cpu())
