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

## gfx1151 (Strix Halo) specific questions

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

For practical implementation details on virtual memory management APIs, see the
[HIP Virtual Memory Management documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management/virtual_memory.html).

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

For information on configuring GTT size, see the next question.

### How do I configure shared memory allocation on Linux?

For GPUs using unified memory (including gfx1151/Strix Halo APUs), you can adjust
the Graphics Translation Table (GTT) size allocation. See the official ROCm
documentation on [configuring shared memory](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installryz/native_linux/install-ryzen.html#configure-shared-memory).

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

### Build fails with "resource exhausted" or out of memory errors

ROCm is a resource-intensive project to build. Try the following:

1. **Reduce parallelism**: Limit the number of parallel jobs

   ```bash
   ninja -C build -j4  # Use only 4 parallel jobs
   ```

1. **Add swap space**: If you have limited RAM, consider adding swap

   ```bash
   sudo fallocate -l 16G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

1. **Build specific components**: Instead of building everything, target only
   what you need

   ```bash
   cmake -B build -GNinja \
     -DTHEROCK_ENABLE_ALL=OFF \
     -DTHEROCK_ENABLE_HIPIFY=ON \
     -DTHEROCK_AMDGPU_FAMILIES=gfx1100
   ```

### CMake version errors

Different project components enforce different CMake version ranges. If you
encounter CMake version issues:

```bash
# Activate your venv
source .venv/bin/activate

# Install a specific CMake version
pip install 'cmake<4'

# On Linux, refresh the path cache
hash -r
```

### Pre-commit hooks are failing

Run pre-commit on all files to see detailed error messages:

```bash
pre-commit run --all-files
```

Common issues:

- **Formatting errors**: Let the hooks auto-fix by running them again after the
  first failure
- **clang-format version mismatch**: Ensure you have clang-format 18.1.4
  installed
- **Trailing whitespace/tabs**: These are auto-fixed by the hooks

### How do I clean and rebuild a specific component?

Use the component's `+expunge` target for a clean slate:

```bash
# Clean and rebuild just CLR
ninja -C build clr+expunge && ninja -C build clr
```

### Build succeeds but tests fail

1. Ensure your GPU is properly detected:

   ```bash
   # Check if amdgpu driver is loaded
   lsmod | grep amdgpu

   # Check GPU visibility
   ls /dev/kfd /dev/dri/render*
   ```

1. Verify library paths are correct:

   ```bash
   export LD_LIBRARY_PATH=build/dist/rocm/lib:$LD_LIBRARY_PATH
   ```

1. Check permissions:

   ```bash
   # Your user should be in the 'render' or 'video' group
   groups
   # Add if needed
   sudo usermod -a -G render $USER
   sudo usermod -a -G video $USER
   # Log out and back in for changes to take effect
   ```

## Contributing questions

### How do I find good first issues?

Look for issues labeled with `good first issue` or check the
[Contribution opportunities](../CONTRIBUTING.md#contribution-opportunities)
section in CONTRIBUTING.md for areas where help is needed.

### Where can I get help with my contribution?

- **Discord**: Join the [AMD Developer Community](https://discord.com/invite/amd-dev)
  and ask in `#therock-contributors`
- **GitHub Issues**: File an issue if you're stuck on a specific problem
- **GitHub Discussions**: For broader questions about direction or approach
