---
description: This page lists frequently asked questions about TheRock
keywords: AMD, ROCm, TheRock, FAQ, frequently asked questions
---

# Frequently asked questions

This topic provides answers to frequently asked questions for TheRock users.

## General questions

### What is TheRock?

TheRock (The HIP Environment and ROCm Kit) is a lightweight, open-source build
platform for HIP and ROCm, designed to provide a streamlined and up-to-date
ROCm environment.

### Who is TheRock intended for?

TheRock is intended for developers, researchers, and advanced users who want
access to the latest ROCm capabilities with a simplified and flexible setup.

### How does TheRock differ from standard ROCm installations?

TheRock provides a more streamlined, frequently updated environment compared to
traditional package-based ROCm installations. It offers greater flexibility in
version management and easier access to cutting-edge features.

### Which GPU architectures are supported by TheRock?

For the most complete and up-to-date information on supported GPU architectures,
please refer to the official TheRock [roadmap](https://github.com/ROCm/TheRock/blob/main/ROADMAP.md).

## gfx1151 (Strix Halo) specific questions

### Why does PyTorch use Graphics Translation Table (GTT) instead of VRAM on gfx1151?

On Strix Halo GPUs (gfx1151) memory access is handled through GPU Virtual Memory
(GPUVM), which provides multiple GPU virtual address spaces identified by VMIDs
(Virtual Memory IDs).

On APUs like Strix Halo, where memory is physically unified, there is no
discrete VRAM. Instead:

- Some memory may be firmware-reserved and pinned for GPU use, while
- GTT-backed memory is dynamically allocated from system RAM and mapped into
  per-process GPU virtual address spaces.

AI workloads typically prefer GTT-backed allocations because they allow large,
flexible mappings without permanently reserving memory for GPU-only use.

### What is the difference between Graphics Address Remapping Table (GART) and GTT?

Within GPUVM, two commonly referenced limits exist:

- GART defines the amount of platform address space (system RAM or Memory-Mapped
  I/O) that can be mapped into the GPU virtual address space used by the kernel
  driver. It is typically kept relatively small to limit GPU page-table size and
  is mainly used for driver-internal operations.

- GTT defines the amount of platform address space (system RAM) that can be
  mapped into the GPU virtual address spaces used by user processes. This is the
  memory pool visible to applications such as PyTorch and other AI workloads.

### Why is allocating to GTT beneficial compared to VRAM?

Allocating large amounts of VRAM permanently removes that memory from general
system use. Increasing GTT allows memory to remain available to both the
operating system and the GPU as needed, providing better flexibility for mixed
workloads. This behavior is expected and intentional on unified memory
architectures.

### Can I prioritize VRAM usage over GTT?

Yes, if your VRAM is larget than GTT, applications will use VRAM instead.
You have two options to prioritize VRAM usage:

- Increase the VRAM in the BIOS settings.
- Manually reduce the GTT size so it's smaller than the VRAM allocation (by
  default, GTT is set to 50% of system RAM).

Note that on APUs, the performance difference between VRAM and GTT is generally
minimal.

## Troubleshooting

### How do I verify my GPU is recognized by TheRock?

Run `rocminfo` or `amd-smi` to verify that your GPU is detected and properly
initialized by the ROCm stack.

### What should I do if I encounter memory allocation errors?

Check your GTT configuration, ensure sufficient system memory is available, and
verify that kernel parameters are correctly set. Review system logs using
`dmesg | grep amdgpu` for specific error messages.

### Where can I get help with TheRock?

For additional support, consult the [TheRock documentation](https://github.com/ROCm/TheRock),
file issues on the [TheRock project repository](https://github.com/ROCm/TheRock/issues),
or engage with the ROCm community via
[ROCm Discussions](https://github.com/ROCm/ROCm/discussions),
the AMD ROCm section on the
[AMD Community Forums](https://community.amd.com/),
and the [AMD Developer Discord](https://discord.com/invite/amd-dev).
