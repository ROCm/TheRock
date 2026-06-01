# Running the ROCm macOS eGPU Port (gfx1201 / RDNA4)

Bring-up of ROCm on Apple Silicon macOS driving an AMD RDNA4 GPU in a Thunderbolt
eGPU enclosure, entirely from userspace (a DriverKit DEXT for MMIO/BAR/DMA; no
amdgpu kernel module). As of 2026-05-31 this runs the full ROCm math-library
stack and the hipDNN/Fusilli/IREE path reliably.

> Status snapshot (validated on the hardware below):
> - hipFFT, hipSOLVER, hipSPARSE, hipRAND, hipBLAS (SAXPY + SGEMM) — combined
>   smoke passes repeatedly, and as separate processes (no cross-process wedge).
> - hipDNN → Fusilli → IREE-compiled gfx1201 VMFB — pointwise smoke passes
>   repeatedly.

---

## 1. The branches (this is the heart of it)

All work lives on per-repo `users/powderluv/*` branches. The **TheRock
super-project pins the exact submodule commits**, so the only branch you check
out directly is the super-project; `git submodule update` then lands the right
commit in each submodule.

| Repo | Remote (`origin`) | Branch | Tip commit | Pushed? |
|------|-------------------|--------|-----------|---------|
| **TheRock** (super-project) | `github.com/ROCm/TheRock` | `users/powderluv/egpu-build` | `5a8d9778` | ✅ |
| `compiler/amd-llvm` | `github.com/ROCm/llvm-project` | `users/powderluv/macos-egpu-driver` | `060badc3` | ✅ |
| `compiler/hipify` | `github.com/ROCm/HIPIFY` | `users/powderluv/macos-egpu` | `2da58655` | ✅ |
| `rocm-libraries` | `github.com/ROCm/rocm-libraries` | `users/powderluv/macos-egpu` | `3aa586d0` | ✅ |
| `rocm-systems` | `github.com/ROCm/rocm-systems` | `users/powderluv/macos-os-darwin` | `0f16da39` | ✅ |
| `iree-libs/fusilli` | `github.com/powderluv/fusilli` (fork) | `users/powderluv/macos-egpu` | `3103867` | ✅ |
| `iree-libs/iree` | `github.com/powderluv/iree` (fork) | `users/powderluv/macos-egpu` | `b7f7acf2` | ✅ |

**iree / fusilli note:** `iree-org` does not grant push access, so those two
branches live on the **`powderluv` forks**, and the super-project's `.gitmodules`
already points the `iree`/`fusilli` submodule URLs at those forks. A fresh
`git submodule update --init --recursive` therefore resolves all seven
submodules. (If you fork under a different account, repoint
`iree-libs/iree` and `iree-libs/fusilli` in `.gitmodules` accordingly.) **The
math-library stack does not need IREE/Fusilli** — you can disable those
components and skip them entirely.

Most of the macOS port lives in **`rocm-systems`** (ROCr runtime: the macOS
agent/driver, the `lite::` direct compute queue, clr/HIP, roctracer, rccl,
rocprofiler Darwin guards) and **`amd-llvm`** (MachO HIP offload bundling). The
GPU **bring-up** (PSP/SOS, GFX, MEC, MES, clock-gating, scheduler) is Python and
lives in the super-project under `userspace_driver/`.

---

## 2. Hardware

- AMD **RX 9070 XT** (`gfx1201`, RDNA4), device id `0x7551`.
- **Razer Core X V2** Thunderbolt eGPU enclosure.
- Apple Silicon Mac (developed on macOS 15 / Darwin 25; Xcode 26.x).
- **Resizable BAR off** → 256 MB BAR0 (the port works within this window).
- Thunderbolt path goes through the Apple DART IOMMU.

---

## 3. Clone + check out

```bash
git clone https://github.com/ROCm/TheRock.git
cd TheRock
git checkout users/powderluv/egpu-build
# rocm-systems needs git-lfs installed BEFORE submodule init
brew install git-lfs && git lfs install
git submodule update --init --recursive
#   ^ resolves amd-llvm / hipify / rocm-libraries / rocm-systems from their
#     ROCm-org branches. fusilli + iree will FAIL to fetch (see caveat above) —
#     push those branches first, or skip IREE/Fusilli.
```

---

## 4. macOS driver (DEXT)

The GPU is reached through a DriverKit system extension
(`userspace_driver/macos_driver`, bundle id `ai.rocm.gpu.driver` /
`ROCmGPUDriver`). Build it with Xcode and install/activate it (a system
extension; requires SIP/driver-extension allowances and the eGPU connected at
match time). Confirm it is active:

```bash
systemextensionsctl list | grep rocm     # expect [activated enabled]
```

(The `com.apple.developer.driverkit.transport.pci` entitlement request is the one
outstanding Apple-side item; development uses a locally-signed build.)

---

## 5. Build ROCm for gfx1201

Out-of-tree build, gfx1201 only:

```bash
cmake -B build-macos-egpu -S . -GNinja -DTHEROCK_AMDGPU_FAMILIES=gfx1201
cmake --build build-macos-egpu --target artifacts --parallel <N>
# Produces build-macos-egpu/dist/rocm
```

Notes / gotchas:
- **Xcode SDK**: if the cached `CMAKE_OSX_SYSROOT` points at a removed SDK after
  an Xcode update, build with `SDKROOT="$(xcrun --show-sdk-path)"` in the env (and
  reconfigure). On Apple `ar`, the build uses Homebrew `llvm-ar`/`llvm-ranlib`
  (Apple `ar` lacks `@response`-file support).
- It is a full TheRock build (Tensile/hipBLASLt and Composable Kernel are the long
  poles). Disable `THEROCK_ENABLE_*` for components you don't need.
- To rebuild just the runtime after a change:
  `SDKROOT=$(xcrun --show-sdk-path) cmake --build build-macos-egpu/core/ROCR-Runtime/build --target install`
  then copy the staged `libhsa-runtime64.1.21.0.dylib` to the dist trees.

---

## 6. Firmware

gfx1201 firmware from linux-firmware (default path `~/firmware/linux-firmware/amdgpu`),
including `gc_12_0_1_uni_mes.bin` (MES/MES-KIQ), `gc_12_0_1_mec.bin`,
`smu_14_0_3.bin`, PSP/SOS/TOC, RLC, etc.

---

## 7. Bring up the GPU (Python, once per power-cycle)

The C++ runtime attaches to an already-brought-up GPU; bring-up is the Python
phase-9 script (PSP SOS → GFX → MEC → MES → gfx12 clock-gating → MES scheduler).
Run it from a cold (freshly power-cycled) eGPU:

```bash
PHASE9_SKIP_NOP=1 PHASE9_SEND_SET_HW_RSRC=1 PHASE9_SEND_SET_HW_RSRC_1=1 \
PHASE9_MAP_SCHED=1 PYTHONPATH=userspace_driver/python \
python3 -u userspace_driver/python/try_phase9_doorbell.py
```

Success looks like `GFX bring-up complete: BOOTLOAD=0x8000003f`,
`clock gating configured`, and all six MES/scheduler fences signaled.

---

## 8. Run the validation smokes

```bash
# Math stack (one combined process by default):
./validation/math_smoke/run_macos_egpu_smokes.sh          # -> "ROCm combined stack smoke passed"
ROCR_MACOS_COMBINED_STACK_SMOKE=0 ./validation/math_smoke/run_macos_egpu_smokes.sh  # 6 separate processes
```

hipDNN/Fusilli/IREE pointwise (needs the IREE/Fusilli build):

```bash
rocm=$PWD/build-macos-egpu/dist/rocm
AMD_GPU_MACOS_FORCE_DIRECT_COMPUTE=1 ROCR_MACOS_HOST_BLIT_ONLY=1 \
ROCR_MACOS_DIRECT_QUEUE_PQ_CONTROL=userspace \
ROCR_MACOS_DIRECT_QUEUE_DEQUEUE_AFTER_SUBMIT=1 \
ROCR_MACOS_DIRECT_QUEUE_ROTATE_BACKING_AFTER_DEQUEUE=1 \
FUSILLI_EXTERNAL_IREE_COMPILE="$rocm/bin/iree-compile" \
DYLD_LIBRARY_PATH="$rocm/lib:$rocm/lib/llvm/lib" ROCM_PATH="$rocm" HIP_PATH="$rocm" \
build-macos-egpu/validation/hipdnn_smoke/build/hipdnn_pointwise_smoke \
  "$rocm/lib/hipdnn_plugins/engines"
```

The math runner already exports the proven direct-queue discipline. Key knobs:
- `AMD_GPU_MACOS_FORCE_DIRECT_COMPUTE=1` — use the direct compute HQD path.
- `ROCR_MACOS_HOST_BLIT_ONLY=1` — H2D/D2H via host blit.
- `ROCR_MACOS_DIRECT_QUEUE_{DEQUEUE_AFTER_SUBMIT,ROTATE_BACKING_AFTER_DEQUEUE}=1`,
  `..._PQ_CONTROL=userspace` — the validated queue recipe.
- `ROCR_MACOS_AQL_ENABLE_HOST_COPYBACK=1` — opt-in copy-back for raw host-pointer
  kernargs (off by default; not needed for hipMalloc workloads).

---

## 9. What the macOS port actually changed (so reviewers know where to look)

- **rocm-systems** (`users/powderluv/macos-os-darwin`): ROCr macOS agent +
  `MacOsDriver` + the shared `lite::` direct compute queue
  (`core/driver/lite/amd_lite_direct_queue.cpp`) and the macOS AQL queue
  (`core/runtime/amd_macos_aql_queue.cpp`); Darwin build guards across
  clr/roctracer/rocshmem/rocprofiler/rccl; trap/blit-shader Darwin stub headers.
  The fixes that made it reliable: gfx12 clock-gating in bring-up, `RESET_WAVES` +
  `SPI_COMPUTE_QUEUE_RESET` dequeue, default-on post-dispatch GL2 writeback,
  zero-size kernarg fallback for Tensile UserArgs kernels, and raw host-pointer
  kernarg staging/copy-back.
- **amd-llvm** (`users/powderluv/macos-egpu-driver`): MachO host-object support in
  the HIP offload bundler.
- **TheRock super-project** (`users/powderluv/egpu-build`): darwin build
  enablement (topology, toolchain/SDK/AR propagation), the `userspace_driver/`
  Python bring-up + DEXT, and the `validation/` smokes.

---

## 10. Gotchas

- **Bring-up is per power-cycle.** Re-running phase-9 on an already-initialized
  card times out at PSP firmware re-load (benign — the GPU is already up). A real
  reset needs a physical power-cycle of the Core X enclosure.
- **Wedge recovery**: if a run aborts (`HSA_STATUS_ERROR ... 0x1000`), power-cycle
  the enclosure and re-run phase-9. (The known correctness/wedge causes are fixed;
  a hard PSP wedge still needs a physical replug — software FLR does not recover
  it.)
- Don't run the forced-HQD smokes in parallel — they share the compute HQD.
