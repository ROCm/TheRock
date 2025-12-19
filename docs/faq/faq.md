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

### What are the system requirements for TheRock?

TheRock requires a compatible AMD GPU, a Linux-based operating system, and
sufficient system resources depending on your workload. Specific requirements
vary based on the GPU architecture (gfx1151, gfx1201, etc.).

### Which GPU architectures are supported by TheRock?

For the most complete and up-to-date information on supported GPU architectures,
please refer to the official TheRock [roadmap](https://github.com/ROCm/TheRock/blob/main/ROADMAP.md).

## gfx1151 (Strix Halo) specific questions

### Why does PyTorch use GTT instead of VRAM on gfx1151?

On Strix Halo GPUs (gfx1151), memory is unified and split between GART (VRAM)
and GTT. GART is a fixed amount of memory reserved exclusively for the GPU via
BIOS, while GTT is dynamically allocated from system memory. PyTorch and other
AI workloads typically prefer GTT because it allows flexible, large allocations
without permanently reserving system memory for GPU-only use.

### Why is allocating to GTT beneficial compared to VRAM?

Allocating large amounts of VRAM permanently removes that memory from general
system use. Increasing GTT allows memory to remain available to both the OS and
the GPU as needed, providing better flexibility for mixed workloads. This
behavior is expected and intentional on unified memory architectures.

### Can I prioritize VRAM usage over GTT?

Yes, you have two options to prioritize VRAM usage:

* Increase the VRAM allocation in the BIOS settings.
* Manually reduce the GTT size so it's smaller than the VRAM allocation (by
  default, GTT is set to 50% of system RAM).

Note that on APUs, the performance difference between VRAM and GTT is generally
minimal, unlike on discrete GPUs where VRAM offers significantly better
performance.

### What is the difference between pages_limit and page_pool_size?

The `pages_limit` option controls the maximum number of 4 KiB pages available
for GPU memory allocation. The `page_pool_size` option pre-allocates memory
exclusively for GPU use, reducing fragmentation and potentially improving
performance. Setting `page_pool_size` equal to `pages_limit` typically yields
the best performance for AI workloads.

### How do I reduce GTT memory fragmentation?

If `page_pool_size` is smaller than `pages_limit`, increasing the allocation
granularity may help reduce fragmentation. For example, set
`amdgpu.vm_fragment_size=8` (where the default of 4 equals 64 KiB and 9 equals
2 MiB).

### Is the amdgpu.gttsize parameter still relevant?

The `amdgpu.gttsize` parameter is officially deprecated. On modern Linux
systems, GTT allocation is managed by the Translation Table Maps (TTM) memory
management subsystem using `pages_limit` and `page_pool_size`. However, some
legacy applications may still reference `gttsize`, so it's recommended to keep
it aligned with your configuration for compatibility.

## Troubleshooting

### How do I verify my GPU is recognized by TheRock?

Run `rocminfo` or `rocm-smi` to verify that your GPU is detected and properly
initialized by the ROCm stack.

### What should I do if I encounter memory allocation errors?

Check your GTT configuration, ensure sufficient system memory is available, and
verify that kernel parameters are correctly set. Review system logs using
`dmesg | grep amdgpu` for specific error messages.

### Where can I get help with TheRock?

For additional support, consult the TheRock documentation, file issues on the
project's repository, or reach out to the ROCm community forums.