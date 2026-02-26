# Frequently asked questions

This topic provides answers to frequently asked questions for TheRock users.

## General questions

### What is TheRock?

TheRock (The HIP Environment and ROCm Kit) is a lightweight, open-source build
platform for HIP and ROCm, designed to provide a streamlined and up-to-date
ROCm environment.

### What does TheRock provide compared to more traditional ROCm releases?

TheRock distributes several types of packages, built daily from the latest ROCm
code. These user-space packages are designed to be easy to install, update, and
even switch between versions.

Key offerings include:

- Nightly builds with cutting-edge features
- Multiple package formats (Python wheels and portable tarballs)
- Flexible version management without system-level dependencies

Traditional ROCm releases prioritize stability and production use, while TheRock
emphasizes rapid access to new developments for contributors and early adopters.

### Which GPU architectures are supported by TheRock?

For the most complete and up-to-date information on supported GPU architectures
and release history, please refer to the the [SUPPORTED_GPUs](https://github.com/ROCm/TheRock/blob/main/SUPPORTED_GPUS.md)
list, and the [RELEASES](https://github.com/ROCm/TheRock/blob/main/RELEASES.md)
file.

For hardware-specific notes and tuning guidance, see the [System optimization pages](https://rocm.docs.amd.com/en/latest/how-to/system-optimization/index.html)

## gfx1151 (Strix Halo) specific questions

Strix Halo specific notes and optimization guidance information are collected on
the [Strix Halo system optimization page](https://rocm.docs.amd.com/en/latest/how-to/system-optimization/strixhalo.html).

### Which OS are supported for Strix Halo?

The most current list of compatible GPU architectures is available on the
[SUPPORTED_GPUs](https://github.com/ROCm/TheRock/blob/main/SUPPORTED_GPUS.md)
page.

For Linux systems running kernel versions earlier than 6.18.4, Strix Halo
requires an additional kernel patch to operate properly. For complete details
on Linux kernel compatibility and required configurations, refer to the system
optimization guide:
https://rocm.docs.amd.com/en/latest/how-to/system-optimization/strixhalo.html#required-kernel-version

### Why does PyTorch use Graphics Translation Table (GTT) instead of VRAM on gfx1151?

On Strix Halo GPUs (gfx1151) memory access is handled through GPU Virtual Memory
(GPUVM), which provides multiple GPU virtual address spaces identified by VMIDs
(Virtual Memory IDs).

GPUVM is the GPU's memory management unit that allows the GPU to remap VRAM and
system memory into separate virtual address spaces for different applications,
providing memory protection between them. Each virtual address space has its own
page table and is identified by a VMID. VMIDs are dynamically allocated to
processes as they submit work to the GPU.

On APUs like Strix Halo, where memory is physically unified, there is no
discrete VRAM. Instead:

- Some memory may be firmware-reserved and pinned for GPU use, while
- GTT-backed memory is dynamically allocated from system RAM and mapped into
  per-process GPU virtual address spaces.

AI workloads typically prefer GTT-backed allocations because they allow large,
flexible mappings without permanently reserving memory for GPU-only use.

For more information, see the
[Strix Halo system optimization page – Memory settings](https://rocm.docs.amd.com/en/latest/how-to/system-optimization/strixhalo.html#memory-settings)

### What is the difference between Graphics Address Remapping Table (GART) and GTT?

Within GPUVM, two commonly referenced limits exist:

- GART (Graphics Address Remapping Table): Defines the amount of platform
  address space (system RAM or Memory-Mapped I/O) that can be mapped into the
  GPU virtual address space used by the kernel driver. On systems with
  physically shared CPU and GPU memory, such as Strix Halo, this mapped system
  memory effectively serves as VRAM for the GPU. GART is typically kept
  relatively small to limit GPU page-table size and is mainly used for
  driver-internal operations.

- GTT (Graphics Translation Table): Defines the amount of system RAM that can be
  mapped into GPU virtual address spaces for user processes. This is the memory
  pool used by applications such as PyTorch and other AI/compute workloads.
  GTT allocations are dynamic and are not permanently reserved, allowing the
  operating system to reclaim memory when it is not actively used by the GPU.
  By default, the GTT limit is set to approximately 50% of total system RAM.

For more information, see the
[Strix Halo system optimization page – Memory settings](https://rocm.docs.amd.com/en/latest/how-to/system-optimization/strixhalo.html#memory-settings)

### Can I prioritize VRAM usage over GTT?

Yes, if your VRAM is larget than GTT, applications will use VRAM instead.
You have two options to prioritize VRAM usage:

- Increase the VRAM in the BIOS settings.
- Manually reduce the GTT size so it's smaller than the VRAM allocation (by
  default, GTT is set to 50% of system RAM).

Note that on APUs, the performance difference between VRAM and GTT is generally
minimal.

For information on configuring GTT size, see the next question.

### How do I configure shared memory allocation on Linux?

For GPUs using unified memory (including gfx1151/Strix Halo APUs), you can
adjust the GTT size allocation. See the official ROCm documentation on
[configuring shared memory](https://rocm.docs.amd.com/en/latest/how-to/system-optimization/strixhalo.html#configuring-shared-memory-limits-on-linux).

Note: This applies to Linux systems only and is relevant for any GPU using shared
memory, not just Strix Halo.

## Troubleshooting

### How do I verify my GPU is recognized by TheRock?

See the [Verifying your installation](https://github.com/ROCm/TheRock/blob/main/RELEASES.md#verifying-your-installation)
section in RELEASES.md for platform-specific instructions.

### What should I do if I encounter memory allocation errors?

Check your GTT configuration, ensure sufficient system memory is available, and
verify that kernel parameters are correctly set. Review system logs using
`dmesg | grep amdgpu` for specific error messages.
