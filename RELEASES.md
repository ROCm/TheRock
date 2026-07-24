# Releases

This page describes how to install and use our release artifacts for ROCm and
external builds like PyTorch and JAX. We produce build artifacts as part of our
Continuous Integration (CI) build/test workflows as well as release artifacts as
part of Continuous Delivery (CD) nightly releases.

<!-- TODO: mention other release channels, link to release notes, etc.
       stable: https://repo.amd.com/rocm/
       https://rocm.docs.amd.com/en/7.12.0-preview/index.html
       https://rocm.docs.amd.com/en/7.13.0-preview/index.html
-->

For the development status of GPU architecture support in TheRock, please see
[SUPPORTED_GPUS.md](./SUPPORTED_GPUS.md) which tracks release readiness for each
AMD GPU architecture.

> [!IMPORTANT]
> These instructions assume familiarity with how to use ROCm.
> Please see https://rocm.docs.amd.com/ for general information about the ROCm software
> platform.
>
> Prerequisites:
>
> - We recommend installing the latest [AMDGPU driver](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html#amdgpu-driver-installation) on Linux and [Adrenalin driver](https://www.amd.com/en/products/software/adrenalin.html) on Windows
> - Linux users, please be aware of [Configuring permissions for GPU access](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/prerequisites.html#configuring-permissions-for-gpu-access) needed for ROCm

Table of contents:

- [About multi-arch releases](#about-multi-arch-releases)
- [Installing multi-arch releases](#installing-multi-arch-releases)
  - [Installing multi-arch Python packages](#installing-multi-arch-python-packages)
    - [Installing multi-arch ROCm Python packages](#installing-multi-arch-rocm-python-packages)
      - [Using multi-arch ROCm Python packages](#using-multi-arch-rocm-python-packages)
      - [Supported Python `[device-*]` install extras](#supported-python-device--install-extras)
    - [Installing multi-arch PyTorch Python packages](#installing-multi-arch-pytorch-python-packages)
    - [Installing multi-arch JAX Python packages](#installing-multi-arch-jax-python-packages)
  - [Installing multi-arch native packages](#installing-multi-arch-native-packages)
    - [Installing multi-arch native Linux packages](#installing-multi-arch-native-linux-packages)
      <!-- - [Using ROCm from WSL](#using-rocm-from-wsl) -->
    <!-- TODO: native Windows packages -->
  - [Installing multi-arch tarballs](#installing-multi-arch-tarballs)
  - [Installing ASan-instrumented libraries](#installing-asan-instrumented-libraries)
- [Verifying your installation](#verifying-your-installation)

## About multi-arch releases

> [!IMPORTANT]
> We introduced multi-arch releases with
> [#3323](https://github.com/ROCm/TheRock/issues/3323). Rather than build
> ROCm for GPU family subsets like the
> [legacy per-family releases](/docs/packaging/legacy_per_family_releases.md),
> these multi-arch releases build all GPU architectures together and split
> GPU-specific code (kernel packs) from architecture-neutral host code as a
> packaging step.
>
> This new setup streamlines package installation, so please note the
> differences in the install instructions.

Key differences from
[legacy per-family releases](/docs/packaging/legacy_per_family_releases.md):

- **One index URL for all GPUs**: select your target with a pip extra like
  `[device-gfx942]` instead of finding a per-family index URL
- **Broader GPU support**: adding support for a new GPU target is just one
  more device package, so more GPUs can be supported without impacting build
  times or download sizes for other targets
- **Smaller downloads**: kernels downloaded can be scoped to a single GPU
  instead of always being scoped to a family or "all"

### Multi-arch release status

> [!WARNING]
> Nightly packages are built from the latest ROCm code and may be unstable.
>
> If you encounter issues, check
>
> - https://therock-hud.amd.com/ for current test status
> - https://github.com/ROCm/TheRock/issues for known issues

| Job description                        | Status                                                                                                                                                                                                                                                     |
| -------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Build ROCm artifacts/tarballs/packages | [![Multi-Arch Release](https://github.com/ROCm/rockrel/actions/workflows/multi_arch_release.yml/badge.svg)](https://github.com/ROCm/rockrel/actions/workflows/multi_arch_release.yml)                                                                      |
| ASan instrumented build                | [![Multi-Arch Release ASan](https://github.com/ROCm/rockrel/actions/workflows/multi_arch_release_asan.yml/badge.svg)](https://github.com/ROCm/rockrel/actions/workflows/multi_arch_release_asan.yml)                                                       |
| Test ROCm artifacts                    | [![Test Artifacts](https://github.com/ROCm/rockrel/actions/workflows/test_artifacts.yml/badge.svg)](https://github.com/ROCm/rockrel/actions/workflows/test_artifacts.yml)                                                                                  |
| Test ROCm native Linux packages        | [![Test Native Linux Packages Install](https://github.com/ROCm/rockrel/actions/workflows/test_native_linux_packages_install.yml/badge.svg)](https://github.com/ROCm/rockrel/actions/workflows/test_native_linux_packages_install.yml)                      |
| PyTorch packages - Linux build/test    | [![Multi-Arch Release Linux PyTorch Wheels](https://github.com/ROCm/rockrel/actions/workflows/multi_arch_release_linux_pytorch_wheels.yml/badge.svg)](https://github.com/ROCm/rockrel/actions/workflows/multi_arch_release_linux_pytorch_wheels.yml)       |
| PyTorch packages - Windows build/test  | [![Multi-Arch Release Windows PyTorch Wheels](https://github.com/ROCm/rockrel/actions/workflows/multi_arch_release_windows_pytorch_wheels.yml/badge.svg)](https://github.com/ROCm/rockrel/actions/workflows/multi_arch_release_windows_pytorch_wheels.yml) |
| PyTorch packages - full tests          | [![Test PyTorch Wheels (Full Suite)](https://github.com/ROCm/rockrel/actions/workflows/test_pytorch_wheels_full.yml/badge.svg)](https://github.com/ROCm/rockrel/actions/workflows/test_pytorch_wheels_full.yml)                                            |
| JAX packages - Linux build/test        | [![Multi-Arch Release Linux JAX Wheels](https://github.com/ROCm/rockrel/actions/workflows/multi_arch_release_linux_jax_wheels.yml/badge.svg)](https://github.com/ROCm/rockrel/actions/workflows/multi_arch_release_linux_jax_wheels.yml)                   |

**Package availability:**

| Package type            | Linux        | Windows                                                           |
| ----------------------- | ------------ | ----------------------------------------------------------------- |
| ROCm Python packages    | ✅ Available | ✅ Available                                                      |
| PyTorch Python packages | ✅ Available | ✅ Available                                                      |
| JAX Python packages     | ✅ Available | -                                                                 |
| ROCm tarballs           | ✅ Available | ✅ Available                                                      |
| Native packages         | ✅ Available | 🟠 Planned ([#1987](https://github.com/ROCm/TheRock/issues/1987)) |

## Installing multi-arch releases

### Installing multi-arch Python packages

Nightly releases of ROCm and framework Python packages are published to a
unified index at https://rocm.nightlies.amd.com/whl-multi-arch/.

<!-- TODO: mention https://repo.amd.com/rocm/whl-multi-arch/ release channel too -->

> [!TIP]
> We highly recommend working within a [Python virtual environment](https://docs.python.org/3/library/venv.html):
>
> ```bash
> python -m venv .venv
> source .venv/bin/activate
> ```
>
> Multiple virtual environments can be present on a system at a time, allowing you to switch between them at will.

> [!WARNING]
> If you _really_ want a system-wide install, you can pass `--break-system-packages` to `pip` outside a virtual environment.
> In this case, command-line interface shims for executables are installed to `/usr/local/bin`, which normally has precedence over `/usr/bin` and might therefore conflict with a previous installation of ROCm.

#### Installing multi-arch ROCm Python packages

We provide several Python packages which together form the complete ROCm SDK.
In multi-arch releases, GPU-specific device code is split into separate
`rocm-sdk-device-{target}` packages.

- See [ROCm Python Packaging via TheRock](./docs/packaging/python_packaging.md)
  for information about each package.
- The packages are defined in the
  [`build_tools/packaging/python/templates/`](https://github.com/ROCm/TheRock/tree/main/build_tools/packaging/python/templates)
  directory.

| Install extra (if any)     | Package name               | Description                                                        |
| -------------------------- | -------------------------- | ------------------------------------------------------------------ |
| `rocm`                     | `rocm`                     | Primary sdist meta package that dynamically determines other deps  |
| `rocm`                     | `rocm-sdk-core`            | OS-specific core of the ROCm SDK (e.g. compiler and utility tools) |
| `rocm[libraries]`          | `rocm-sdk-libraries`       | OS-specific libraries (architecture-neutral host code)             |
| `rocm[device-gfx{target}]` | `rocm-sdk-device-{target}` | GPU-specific device code (e.g. `rocm-sdk-device-gfx942`)           |
| `rocm[devel]`              | `rocm-sdk-devel`           | OS-specific development tools                                      |

Install ROCm with device support for your GPU by looking up the appropriate
`[device-*]` extras from the
[Supported Python `[device-*]` install extras](#supported-python-device--install-extras) table below
and then running an install command such as:

```bash
# Single device (replace device-gfx942 with your GPU):
pip install --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ \
    "rocm[libraries,device-gfx942]"

# Also install the 'devel' package:
pip install --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ \
    "rocm[libraries,devel,device-gfx942]"

# Multiple devices (e.g. for a Dockerfile used by both MI300X and MI355X):
pip install --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ \
    "rocm[libraries,device-gfx942,device-gfx950]"

# All supported devices:
pip install --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ \
    "rocm[libraries,device-all]"
```

> [!WARNING]
> A `device-*` extra (or a single-family per-architecture index) being
> installable does **not** mean the runtime is functional on that target.
> Targets without ✅ in **Sanity Tested** in
> [SUPPORTED_GPUS.md](SUPPORTED_GPUS.md) are unverified. `pip install` will
> succeed, but device enumeration, kernel launch, or library loads may fail at
> runtime. Please file an issue if you hit one.

<!-- TODO: Advertise wheel variants / WheelNext once available  -->

<!-- TODO(#5289): Advertise backwards-compatible index once available -->

##### Using multi-arch ROCm Python packages

After installing the ROCm Python packages, you should see them in your
environment:

```bash
pip freeze | grep rocm
# rocm==7.15.0a20260630
# rocm-sdk-core==7.15.0a20260630
# rocm-sdk-device-gfx1100==7.15.0a20260630
# rocm-sdk-libraries==7.15.0a20260630
```

You should also see various tools on your `PATH` and in the `bin` directory:

```bash
which rocm-sdk
# .../.venv/bin/rocm-sdk

ls .venv/bin
# activate       amdclang++    hipcc      python                 rocm-sdk
# activate.csh   amdclang-cl   hipconfig  python3                rocm-smi
# activate.fish  amdclang-cpp  pip        python3.12             roc-obj
# Activate.ps1   amdflang      pip3       rocm_agent_enumerator  roc-obj-extract
# amdclang       amdlld        pip3.12    rocminfo               roc-obj-ls
```

The `rocm-sdk` tool can be used to inspect and test the installation:

```console
$ rocm-sdk --help
usage: rocm-sdk {command} ...

ROCm SDK Python CLI

positional arguments:
  {path,test,version,targets,init}
    path                Print various paths to ROCm installation
    test                Run installation tests to verify integrity
    version             Print version information
    targets             Print information about the GPU targets that are supported
    init                Expand devel contents to initialize rocm[devel]

$ rocm-sdk test
...
Ran 22 tests in 8.284s
OK

$ rocm-sdk targets
gfx1100;gfx1101;gfx1102;gfx1103;gfx1151;gfx1200;gfx1201;...
```

If you also installed the `rocm-sdk-devel` development package using the
`rocm[devel]` extra and want to use it outside of Python, you can _eagerly_
expand its contents using `rocm-sdk init`:

```console
$ rocm-sdk init
Devel contents expanded to '.venv/lib/python3.12/site-packages/_rocm_sdk_devel'
```

The paths in the devel package can be used like so:

```console
$ rocm-sdk path --root
.venv/Lib/site-packages/_rocm_sdk_devel
$ rocm-sdk path --bin
.venv/Lib/site-packages/_rocm_sdk_devel/bin
$ rocm-sdk path --cmake
.venv/Lib/site-packages/_rocm_sdk_devel/lib/cmake
```

```bash
-DCMAKE_PREFIX_PATH=$(rocm-sdk path --cmake)
-DROCM_HOME=$(rocm-sdk path --root)
export PATH="$(rocm-sdk path --bin):$PATH"
```

For more details on using the `rocm-sdk-devel` package to build projects, see
[Using Packages from Frameworks in `docs/packaging/python_packaging.md`](https://github.com/ROCm/TheRock/blob/main/docs/packaging/python_packaging.md#using-packages-from-frameworks).

> [!TIP]
> The devel tree is expanded - and its device files linked from the installed
> `rocm-sdk-device-*` wheels - only once: on the first `rocm-sdk init` /
> `rocm-sdk test`, or the first use of a devel tool such as `hipcc`. If you
> install or remove a `rocm-sdk-device-*` wheel (for example, adding a second GPU
> target) **after** that first expansion, re-run `rocm-sdk init` or `rocm-sdk test`
> to link the new device files. The compiler tools do not re-scan on their own,
> so a device wheel added later is not picked up until you run one of those again.
> Uninstalling a `rocm-sdk-device-*` wheel removes its devel files automatically
> via `pip`. If the devel tree ever ends up in a bad state, recreate the virtual
> environment.

##### Supported Python `[device-*]` install extras

For packages which include device-specific code (such as `rocm`, `torch`, and
`torchvision`), select your GPU using a `[device-*]` install extra from the
table below. See also the
[GPU architecture specs](https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html)
for a full list of supported AMD GPUs.

| Product Name                                         | GFX Target | Device Extra     |
| ---------------------------------------------------- | ---------- | ---------------- |
| *All supported GPUs*                                 | (all)      | `device-all`     |
| AMD Instinct MI355X / MI350X                         | gfx950     | `device-gfx950`  |
| AMD Instinct MI325X / MI300X / MI300A                | gfx942     | `device-gfx942`  |
| AMD Instinct MI250X / MI250 / MI210                  | gfx90a     | `device-gfx90a`  |
| AMD Instinct MI100                                   | gfx908     | `device-gfx908`  |
| AMD Instinct MI60 / MI50, Radeon Pro VII, Radeon VII | gfx906     | `device-gfx906`  |
| AMD Instinct MI25                                    | gfx900     | `device-gfx900`  |
| AMD Radeon RX 9070 / XT, AI PRO R9700 / R9600D       | gfx1201    | `device-gfx1201` |
| AMD Radeon RX 9060 / XT                              | gfx1200    | `device-gfx1200` |
| AMD Radeon 820M iGPU                                 | gfx1153    | `device-gfx1153` |
| AMD Ryzen AI 7 350                                   | gfx1152    | `device-gfx1152` |
| AMD Ryzen AI Max+ PRO 395                            | gfx1151    | `device-gfx1151` |
| AMD Ryzen AI 9 HX 375                                | gfx1150    | `device-gfx1150` |
| AMD Ryzen 7 7840U / Ryzen 9 270                      | gfx1103    | `device-gfx1103` |
| AMD Radeon RX 7600                                   | gfx1102    | `device-gfx1102` |
| AMD Radeon RX 7800 XT / 7700 XT, PRO V710 / W7700    | gfx1101    | `device-gfx1101` |
| AMD Radeon RX 7900 XTX / 7900 XT, PRO W7900 / W7800  | gfx1100    | `device-gfx1100` |
| AMD Radeon RX 6900 XT / 6800 XT, PRO W6800 / V620    | gfx1030    | `device-gfx1030` |
| AMD Radeon RX 6750 XT / 6700 XT                      | gfx1031    | `device-gfx1031` |
| AMD Radeon RX 6600 XT / 6600, PRO W6600              | gfx1032    | `device-gfx1032` |
| AMD Van Gogh iGPU                                    | gfx1033    | `device-gfx1033` |
| AMD Radeon RX 6500 XT                                | gfx1034    | `device-gfx1034` |
| AMD Radeon 680M iGPU                                 | gfx1035    | `device-gfx1035` |
| AMD Raphael iGPU                                     | gfx1036    | `device-gfx1036` |
| AMD Radeon RX 5700 / XT                              | gfx1010    | `device-gfx1010` |
| AMD Radeon Pro V520                                  | gfx1011    | `device-gfx1011` |
| AMD Radeon Pro W5500                                 | gfx1012    | `device-gfx1012` |

#### Installing multi-arch PyTorch Python packages

Install PyTorch with ROCm support using the unified multi-arch index.
Select your GPU target using the `[device-*]` extras from the
[table above](#supported-python-device--install-extras):

> [!NOTE]
> By default, pip will install the latest stable versions of each package.
>
> - If you want to allow installing prerelease versions, use the `--pre`
>
> - If you want to install other versions, take note of the compatibility
>   matrix:
>
>   | torch version | torchaudio version | torchvision version | apex version |
>   | ------------- | ------------------ | ------------------- | ------------ |
>   | 2.14          | 2.11               | 0.29                | 1.14         |
>   | 2.13          | 2.11               | 0.28                | 1.13         |
>   | 2.12          | 2.11               | 0.27                | 1.12         |
>   | 2.11          | 2.11               | 0.26                | 1.11         |
>   | 2.10          | 2.10               | 0.25                | 1.10         |
>
>   For example, `torch` 2.11 and compatible wheels can be installed by specifying
>
>   ```
>   torch==2.11 torchaudio==2.11 torchvision==0.26 apex==1.11.0
>   ```
>
>   See also
>
>   - [Supported PyTorch versions in TheRock](https://github.com/ROCm/TheRock/tree/main/external-builds/pytorch#supported-pytorch-versions)
>   - [Installing previous versions of PyTorch](https://pytorch.org/get-started/previous-versions/)
>   - [torchvision installation - compatibility matrix](https://github.com/pytorch/vision?tab=readme-ov-file#installation)
>   - [torchaudio installation - compatibility matrix](https://docs.pytorch.org/audio/main/installation.html#compatibility-matrix)
>   - [apex installation - compatibility matrix](https://github.com/ROCm/apex/tree/master?tab=readme-ov-file#supported-versions)

> [!WARNING]
> The `torch` packages depend on `rocm[libraries]`, so the compatible ROCm packages
> should be installed automatically for you and you do not need to explicitly install
> ROCm first. If ROCm is already installed this may result in a downgrade if the
> `torch` wheel to be installed requires a different version.

```bash
# Single device (replace device-gfx942 with your GPU):
pip install --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ \
    "torch[device-gfx942]" "torchvision[device-gfx942]" torchaudio

# Multiple devices (e.g. for a Dockerfile used by both MI300X and MI355X):
pip install --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ \
    "torch[device-gfx942,device-gfx950]" \
    "torchvision[device-gfx942,device-gfx950]" \
    torchaudio

# All supported devices:
pip install --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ \
    "torch[device-all]" "torchvision[device-all]" torchaudio

# Optional additional packages on Linux:
#   apex
```

> [!TIP]
> The device extras install GPU-specific packages like `amd-torch-device-gfx1100`
> which contain GPU-specific kernels and depend on `rocm-sdk-device-gfx1100`.
> The compatible ROCm packages are installed automatically, you do not need to
> install ROCm separately:
>
> ```bash
> pip install --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ \
>     "torch[device-gfx1100]"
>
> pip freeze  # with approximate download sizes:
> # rocm-sdk-core==7.13.0a...              ~700 MB
> # rocm-sdk-libraries==7.13.0a...         ~100 MB  (host code, shared across GPUs)
> # rocm-sdk-device-gfx1100==7.13.0a...     ~50 MB  (only gfx1100 device code)
> # torch==2.11.0+rocm...                  ~100 MB  (host code, shared across GPUs)
> # amd-torch-device-gfx1100==2.11.0+...    ~50 MB  (only gfx1100 device code)
> # Total:                                 ~1.1 GB
> #
> # For comparison, a similar per-family (non-multi-arch) torch wheel for
> # gfx110X-all [gfx1100, gfx1101, gfx1102, gfx1103] is ~600 MB.
> ```

##### Using multi-arch PyTorch Python packages

After installing, verify PyTorch can see your GPU:

```python
import torch

print(torch.cuda.is_available())
# True
print(torch.cuda.get_device_name(0))
# e.g. AMD Radeon Pro W7900 Dual Slot
```

See [external-builds/pytorch/README.md](/external-builds/pytorch/README.md) for
more details on supported PyTorch versions and building from source.

See also the
[Testing the PyTorch installation](https://rocm.docs.amd.com/projects/install-on-linux/en/develop/install/3rd-party/pytorch-install.html#testing-the-pytorch-installation)
instructions in the AMD ROCm documentation.

#### Installing multi-arch JAX Python packages

Install JAX with ROCm support using the unified multi-arch index.

> [!IMPORTANT]
> Unlike PyTorch, the JAX wheels do **not** automatically install ROCm packages
> as a dependency. You must install ROCm first by following
> [Installing multi-arch ROCm Python packages](#installing-multi-arch-rocm-python-packages).
>
> Always pin `jax`, `jax_rocm7_plugin`, and `jax_rocm7_pjrt` to the same version.

```bash
# Set the version (currently supported: 0.9.1, 0.10.0, and 0.10.2)
JAX_VERSION=0.10.0

# 1. Install ROCm (replace device-gfx942 with your GPU)
pip install --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ \
    "rocm[libraries,device-gfx942]"

# 2. Install JAX ROCm wheels
pip install --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ \
    "jax_rocm7_plugin==${JAX_VERSION}" \
    "jax_rocm7_pjrt==${JAX_VERSION}"

# 3. Install matching jax from PyPI
pip install "jax==${JAX_VERSION}"
```

After installing, verify JAX can see your GPU:

```python
import jax

print(jax.devices())
# [RocmDevice(id=0), RocmDevice(id=1), ...]
```

### Installing multi-arch native packages

Native packages are installable via operating system package managers.

#### Installing multi-arch native Linux packages

ROCm native Linux packages are published for Debian-based and RPM-based
distributions.

> [!WARNING]
> Nightly builds are primarily intended for development and testing and are
> currently **unsigned**.

Multi-arch native packages use a simplified package model compared to the
[legacy per-family releases](/docs/packaging/legacy_per_family_releases.md):

| Package name       | Description                                                                                                      |
| ------------------ | ---------------------------------------------------------------------------------------------------------------- |
| `amdrocm`          | Installs all base ROCm libraries and runtime support for all supported GPU architectures                         |
| `amdrocm-core-sdk` | Installs the full ROCm SDK including runtime, development tools, and headers for all supported GPU architectures |

> [!TIP]
> To find the latest available release, browse the index pages:
>
> - **Debian packages**: https://rocm.nightlies.amd.com/packages-multi-arch/deb/
> - **RPM packages**: https://rocm.nightlies.amd.com/packages-multi-arch/rpm/
>
> Look for directories in the format `YYYYMMDD-<action-run-id>`
> (e.g., `20260701-28484694006`) and use the latest in the commands below.

##### Installing on Debian-based systems (Ubuntu, Debian, etc.)

```bash
# Step 1: Find the latest release from
#         https://rocm.nightlies.amd.com/packages-multi-arch/deb/
#         Look for directories like "20260701-28484694006"
# Step 2: Set the variable below
export RELEASE_ID=20260701-28484694006  # Replace with the latest date-runid

# Step 3: Add repository and install
sudo apt update
sudo apt install -y ca-certificates
echo "deb [trusted=yes] https://rocm.nightlies.amd.com/packages-multi-arch/deb/${RELEASE_ID} stable main" \
  | sudo tee /etc/apt/sources.list.d/rocm-multiarch-nightly.list
sudo apt update

# Install base runtime for all supported GPU architectures:
sudo apt install amdrocm
# Or install full SDK (runtime + dev tools + headers) for all supported GPU architectures:
sudo apt install amdrocm-core-sdk
```

##### Installing on RPM-based systems (RHEL, SLES, AlmaLinux, etc.)

```bash
# Step 1: Find the latest release from
#         https://rocm.nightlies.amd.com/packages-multi-arch/rpm/
#         Look for directories like "20260701-28484694006"
# Step 2: Set the variable below
export RELEASE_ID=20260701-28484694006  # Replace with the latest date-runid

# Step 3: Add repository and install
sudo dnf install -y ca-certificates
sudo tee /etc/yum.repos.d/rocm-multiarch-nightly.repo <<EOF
[rocm-multiarch-nightly]
name=ROCm Multi-Arch Nightly Repository
baseurl=https://rocm.nightlies.amd.com/packages-multi-arch/rpm/${RELEASE_ID}/x86_64
enabled=1
gpgcheck=0
priority=50
EOF

# Install base runtime for all supported GPU architectures:
sudo dnf clean all
sudo dnf install amdrocm
# Or install full SDK (runtime + dev tools + headers) for all supported GPU architectures:
sudo dnf install amdrocm-core-sdk
```

> [!NOTE]
> To install support for a specific GPU architecture only, you can use the
> per-arch package variant (e.g., `apt install amdrocm-gfx942` or `dnf install amdrocm-gfx942`). For a full list of
> supported GPU targets and their identifiers, see
> [Supported Python `[device-*]` install extras](#supported-python-device--install-extras).

<!-- TODO: include WSL documentation once we are satisfied with quality in nightly releases -->

<!--

##### Using ROCm from WSL

ROCm supports WSL via the DXG kernel interface. DXG detection is enabled by
default as of ROCm 7.13.

To use ROCm on WSL, install the optional `amdrocm-wsl` package which provides
the DXG support library:

```bash
# For Debian/Ubuntu:
sudo apt install amdrocm-wsl

# For RHEL/CentOS/Fedora:
sudo dnf install amdrocm-wsl
```

To explicitly disable DXG detection, set:

```bash
export HSA_ENABLE_DXG_DETECTION=0
``` -->

### Installing multi-arch tarballs

Standalone "ROCm SDK tarballs" are a flattened view of ROCm
[artifacts](docs/development/artifacts.md) matching the familiar folder
structure seen with system installs on Linux to `/opt/rocm/` or on Windows via
the HIP SDK:

```bash
install/
  .kpack/     # GPU-specific kernel packs
  bin/
  clients/
  include/
  lib/
  libexec/
  share/
```

Tarballs are _just_ these raw files. They do not come with "install" steps
such as setting environment variables.

Multi-arch tarballs separate GPU-specific kernel code into a `.kpack/`
directory. Two variants are available:

- **Per-family multi-arch tarballs** (e.g. `therock-dist-linux-gfx110X-all-7.13.0a20260430.tar.gz`)
  that include `.kpack` files only for one family.
- **Full multi-arch tarball** (e.g. `therock-dist-linux-multiarch-7.13.0a20260430.tar.gz`)
  that include `.kpack` files for all supported targets.

Browse and download tarballs from
https://rocm.nightlies.amd.com/tarball-multi-arch/.

To download and extract:

```bash
mkdir therock-tarball && cd therock-tarball

# Per-family (smaller, one GPU family):
wget https://rocm.nightlies.amd.com/tarball-multi-arch/therock-dist-linux-gfx110X-all-7.13.0a20260430.tar.gz

# Or multiarch (all GPUs):
wget https://rocm.nightlies.amd.com/tarball-multi-arch/therock-dist-linux-multiarch-7.13.0a20260430.tar.gz

mkdir install && tar -xf *.tar.gz -C install
```

After extraction, test the install:

```bash
./install/bin/rocminfo
ls install/.kpack/
# blas_lib_gfx1100.kpack  fft_lib_gfx1100.kpack  rand_lib_gfx1100.kpack  ...
```

> [!TIP]
> You may also want to add parts of the install directory to your `PATH` or set
> other environment variables like `ROCM_HOME`.
>
> See also [this issue](https://github.com/ROCm/TheRock/issues/1658) discussing
> relevant environment variables.

### Installing ASan-instrumented libraries

ROCm ships a set of libraries built with
[AddressSanitizer (ASan)](https://clang.llvm.org/docs/AddressSanitizer.html)
instrumentation. See also [Building ROCm with Sanitizers](./docs/development/sanitizers.md)
for information on building ASan-instrumented libraries from source. These instrumented libraries let you run your HIP or ROCm
application under AddressSanitizer to detect memory errors — such as
out-of-bounds accesses, use-after-free, and heap corruption — in **both host
(CPU) and device (GPU) code paths** that pass through the ROCm runtime and math
libraries.

Device-side (GPU) ASan additionally relies on the **XNACK**
(retry-on-page-fault) capability of supported AMD GPUs, which allows the
sanitizer runtime to service the shadow-memory accesses that instrumentation
generates on the device.

#### Available ASan packages

ASan packages are currently available for **Linux only** (not Windows) and are
published for the following GPU families:

| GPU Family | Targets                |
| ---------- | ---------------------- |
| gfx94X     | gfx942 (MI300X/MI300A) |
| gfx950     | gfx950 (MI350X/MI355X) |

> [!NOTE]
> The ROCm compiler toolchain (`amdclang` / `amdclang++`) supports ASan
> instrumentation on any XNACK-capable GPU architecture. The table above
> reflects which GPU families have pre-built ASan packages available in the
> nightly releases.

#### Prerequisites

- **Linux kernel 5.6 or higher** with Heterogeneous Memory Management (HMM)
  support for device-side ASan. Check with `uname -r`.
- An **XNACK-capable GPU** for device-side ASan. Host-side ASan works without
  XNACK, but device-side instrumentation requires it.
- The ROCm compiler toolchain (`amdclang` / `amdclang++`) from your ROCm
  installation.
- Sufficient free disk space — the instrumented libraries are larger than their
  uninstrumented counterparts.
- Familiarity with AddressSanitizer output. See the upstream
  [AddressSanitizer documentation](https://clang.llvm.org/docs/AddressSanitizer.html)
  for how to read reports.

#### Installing via native packages (DEB / RPM)

ASan-instrumented native packages are published for Debian-based and RPM-based
distributions.

##### Debian-based systems (Ubuntu, Debian, etc.)

```bash
# Step 1: Find the latest release from
#         https://rocm.nightlies.amd.com/packages-asan/deb/
#         Look for directories like "20260714-29296019987"
# Step 2: Set the variable below
export RELEASE_ID=20260714-29296019987  # replace with the latest date-runid

# Step 3: Add repository and install
echo "deb [trusted=yes] https://rocm.nightlies.amd.com/packages-asan/deb/${RELEASE_ID} stable main" \
  | sudo tee /etc/apt/sources.list.d/rocm-asan-nightly.list
sudo apt update

# Install all supported GPU architectures (larger download):
sudo apt install amdrocm
# Or install for a specific GPU architecture only (smaller download, recommended):
sudo apt install amdrocm7.15-gfx942   # MI300X / MI300A
sudo apt install amdrocm7.15-gfx950   # MI350X
```

##### RPM-based systems (RHEL, SLES, AlmaLinux, etc.)

```bash
# Step 1: Find the latest release from
#         https://rocm.nightlies.amd.com/packages-asan/rpm/
#         Look for directories like "20260714-29296019987"
# Step 2: Set the variable below
export RELEASE_ID=20260714-29296019987  # replace with the latest date-runid

# Step 3: Add repository and install
sudo tee /etc/yum.repos.d/rocm-asan-nightly.repo <<EOF
[rocm-asan-nightly]
name=ROCm ASAN Nightly Repository
baseurl=https://rocm.nightlies.amd.com/packages-asan/rpm/${RELEASE_ID}/x86_64
enabled=1
gpgcheck=0
priority=50
EOF

sudo dnf clean all

# Install all supported GPU architectures (larger download):
sudo dnf install amdrocm
# Or install for a specific GPU architecture only (smaller download, recommended):
sudo dnf install amdrocm7.15-gfx942   # MI300X / MI300A
sudo dnf install amdrocm7.15-gfx950   # MI350X
```

#### Installing via tarball

Browse https://rocm.nightlies.amd.com/tarball-asan/ for the latest available
release and set `ROCM_VERSION` accordingly (e.g. `7.15.0a20260714`).

```bash
export ROCM_VERSION=7.15.0a20260714   # replace with the latest from the link above
export ASAN_TARBALL_BASE_URL=https://rocm.nightlies.amd.com/asan/tarball/
```

Download the tarball that matches your GPU family:

```bash
# gfx94X (MI300X / MI300A):
wget ${ASAN_TARBALL_BASE_URL}therock-dist-linux-gfx94X-dcgpu-${ROCM_VERSION}.tar.gz

# gfx950 (MI350X):
wget ${ASAN_TARBALL_BASE_URL}therock-dist-linux-gfx950-dcgpu-${ROCM_VERSION}.tar.gz
```

Alternatively, download the multi-arch tarball covering all supported GPU
families (large download — use only if you need multiple architectures):

```bash
wget ${ASAN_TARBALL_BASE_URL}therock-dist-linux-multiarch-${ROCM_VERSION}.tar.gz
```

Extract into a dedicated directory. **Do not extract over your production ROCm
installation** — keep it in a separate location so you can opt in only when
running under ASan:

```bash
mkdir -p <EXTRACT_PATH>
tar -xzf therock-dist-linux-gfx94X-dcgpu-${ROCM_VERSION}.tar.gz -C <EXTRACT_PATH>
```

> [!NOTE]
> **Setting `ROCM_ASAN_PATH`:** The [Using ASan-instrumented libraries](#using-asan-instrumented-libraries)
> section below uses a single variable `ROCM_ASAN_PATH` to refer to the root of
> the ASan install tree. Set it based on how you installed:
>
> ```bash
> # DEB / RPM install:
> export ROCM_ASAN_PATH=/opt/rocm/core
>
> # Tarball install:
> export ROCM_ASAN_PATH=<EXTRACT_PATH>
> ```
>
> In both cases the layout under `ROCM_ASAN_PATH` is the same:
>
> ```
> ${ROCM_ASAN_PATH}/
> ├── bin/
> ├── include/
> └── lib/
>     ├── libamdhip64.so*
>     ├── libhsa-runtime64.so*
>     └── ...   # additional instrumented ROCm libraries
> ```

### Using ASan-instrumented libraries

For detailed instructions on configuring your environment to use ASan-instrumented
libraries, see [Using ASan-Instrumented Libraries](./docs/development/sanitizers.md#using-asan-instrumented-libraries)
in the development documentation.

## Verifying your installation

After installing ROCm via any of the methods above, you can verify that your
GPU is properly recognized.

### Verifying installation on Linux

GPU status on Linux can be checked via either:

```bash
rocminfo
# or
amd-smi
```

### Verifying installation on Windows

GPU status on Windows can be checked via

```bash
hipInfo.exe
```

### Additional installation troubleshooting

If your GPU is not recognized or you encounter issues:

- **Linux users**: Check system logs using `dmesg | grep amdgpu` for specific error messages
- Review memory allocation settings (see the [FAQ](https://github.com/ROCm/TheRock/blob/main/faq.md)
  for GTT configuration on unified memory systems)
- Ensure you have the latest [AMDGPU driver](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html#amdgpu-driver-installation)
  on Linux or [Adrenalin driver](https://www.amd.com/en/products/software/adrenalin.html) on Windows
