#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GPU smoke test for the torchaudio ROCm build.

Exercises the GPU extensions enabled by ROCm/audio's HIPIFY port (ROCm/audio#17):
rnnt loss, forced_align, and loading the CUDA CTC decoder extension. Uses only
tensor inputs (no audio-file dependencies) so it runs on a bare GPU runner.

Exits non-zero on any failure.
"""
import os
import sys

import torch
import torchaudio
import torchaudio.functional as F

print(f"torch      : {torch.__version__}")
print(f"torchaudio : {torchaudio.__version__}")
print(f"torch.version.hip: {torch.version.hip}")

if not torch.cuda.is_available():
    print("ERROR: torch.cuda.is_available() is False on a GPU runner", file=sys.stderr)
    sys.exit(1)

device = torch.device("cuda")
print(f"device: {torch.cuda.get_device_name(0)}")

failures = []


def check(name, fn):
    try:
        fn()
        print(f"[PASS] {name}")
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] {name}: {type(e).__name__}: {e}", file=sys.stderr)
        failures.append(name)


def rnnt_loss_gpu():
    # logits: (B, T, U+1, D)
    B, T, U, D = 2, 5, 3, 6
    torch.manual_seed(0)
    logits = torch.rand(B, T, U + 1, D, device=device, dtype=torch.float32)
    targets = torch.randint(1, D, (B, U), device=device, dtype=torch.int32)
    logit_lengths = torch.full((B,), T, device=device, dtype=torch.int32)
    target_lengths = torch.full((B,), U, device=device, dtype=torch.int32)
    loss = F.rnnt_loss(logits, targets, logit_lengths, target_lengths, blank=0)
    assert loss.is_cuda and torch.isfinite(loss).all(), loss
    # backward to exercise the gradient kernels too
    logits.requires_grad_(True)
    F.rnnt_loss(logits, targets, logit_lengths, target_lengths, blank=0).sum().backward()


def forced_align_gpu():
    # log_probs: (B, T, C)
    B, T, C = 1, 8, 5
    torch.manual_seed(0)
    log_probs = torch.rand(B, T, C, device=device, dtype=torch.float32).log_softmax(-1)
    targets = torch.tensor([[1, 2, 3]], device=device, dtype=torch.int32)
    input_lengths = torch.tensor([T], device=device, dtype=torch.int32)
    target_lengths = torch.tensor([3], device=device, dtype=torch.int32)
    aligned, scores = F.forced_align(
        log_probs, targets, input_lengths, target_lengths, blank=0
    )
    assert aligned.is_cuda and aligned.shape[0] == B, aligned.shape


def cuda_ctc_decoder_import():
    # Loads the torchaudio_prefixctc extension (cuctc) built for ROCm.
    from torchaudio.models.decoder import cuda_ctc_decoder  # noqa: F401


check("rnnt_loss (GPU, fwd+bwd)", rnnt_loss_gpu)
check("forced_align (GPU)", forced_align_gpu)
check("cuda_ctc_decoder import", cuda_ctc_decoder_import)

if failures:
    print(f"\nSMOKE TEST FAILED: {failures}", file=sys.stderr)
else:
    print("\nAll torchaudio GPU smoke checks passed.")

# torch/ROCm can hang at interpreter shutdown on Windows; flush and hard-exit
# so a clean run doesn't stall the CI step.
sys.stdout.flush()
sys.stderr.flush()
os._exit(1 if failures else 0)
