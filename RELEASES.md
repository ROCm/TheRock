# Releases

This page describes how to install and use our release artifacts for ROCm and
external builds like PyTorch and JAX. We produce build artifacts as part of our
Continuous Integration (CI) build/test workflows as well as release artifacts as
part of Continuous Delivery (CD) nightly releases.

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
> - We recommend installing the latest [AMDGPU driver](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html#amdgpu-driver-installation) on Linux and [Adrenaline driver](https://www.amd.com/en/products/software/adrenalin.html) on Windows
> - Linux users, please be aware of [Configuring permissions for GPU access](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/prerequisites.html#configuring-permissions-for-gpu-access) needed for ROCm

Table of contents:

- [Multi-arch releases](#multi-arch-releases)
  - [Multi-arch release status](#multi-arch-release-status)
  - [Installing multi-arch ROCm Python packages](#installing-multi-arch-rocm-python-packages)
  <!-- - [Installing multi-arch PyTorch Python packages](#installing-multi-arch-pytorch-python-packages) -->
  - [Installing multi-arch tarballs](#installing-multi-arch-tarballs)
- [Per-family releases](#per-family-releases)
  - [Installing per-family releases using pip](#installing-per-family-releases-using-pip)
    - [Python packages release status](#python-packages-release-status)
    - [Installing ROCm Python packages](#installing-rocm-python-packages)
    - [Using ROCm Python packages](#using-rocm-python-packages)
    - [Installing PyTorch Python packages](#installing-pytorch-python-packages)
    - [Using PyTorch Python packages](#using-pytorch-python-packages)
    - [Installing JAX Python packages](#installing-jax-python-packages)
    - [Using JAX Python packages](#using-jax-python-packages)
  - [Installing from tarballs](#installing-from-tarballs)
    - [Browsing release tarballs](#browsing-release-tarballs)
    - [Manual tarball extraction](#manual-tarball-extraction)
    - [Automated tarball extraction](#automated-tarball-extraction)
    - [Using installed tarballs](#using-installed-tarballs)
  - [Installing from native packages](#installing-from-native-packages)
    - [Native packages release status](#native-packages-release-status)
    - [Installing on Debian-based systems](#installing-on-debian-based-systems-ubuntu-debian-etc)
    - [Installing on RPM-based systems](#installing-on-rpm-based-systems-rhel-sles-almalinux-etc)
- [Verifying your installation](#verifying-your-installation)

## Multi-arch releases

> [!IMPORTANT]
> We are introducing multi-arch releases with
> [#3323](https://github.com/ROCm/TheRock/issues/3323). Rather than build
> ROCm for GPU family subsets like the [per-family releases](#per-family-releases),
> these multi-arch releases build all GPU architectures together and split
> GPU-specific code (kernel packs) from architecture-neutral host code as a
> packaging step.
>
> This new setup will streamline package installation, so please note the
> differences in the install instructions.

See this table for key differences between release types:

|                      | Multi-arch releases                    | Per-family releases                    |
| -------------------- | -------------------------------------- | -------------------------------------- |
| GPU code separation  | Split into device packages / `.kpack`  | Bundled into each artifact             |
| GPU selection        | Package extras/variants                | Chosen by index URL or tarball name    |
| Multiple GPU targets | Install extras / use multiarch tarball | Separate venvs or tarballs per family  |
| Download size        | Smaller per target                     | Larger (all targets in family bundled) |

### Multi-arch release status

[![Multi-arch release](https://github.com/ROCm/rockrel/actions/workflows/multi_arch_release.yml/badge.svg)](https://github.com/ROCm/rockrel/actions/workflows/multi_arch_release.yml)

**Package availability:**

| Package type            | Linux                                                                 | Windows                                                               |
| ----------------------- | --------------------------------------------------------------------- | --------------------------------------------------------------------- |
| ROCm Python packages    | ✅ Available                                                          | ✅ Available                                                          |
| PyTorch Python packages | 🟡 In progress ([#3332](https://github.com/ROCm/TheRock/issues/3332)) | 🟡 In progress ([#3332](https://github.com/ROCm/TheRock/issues/3332)) |
| JAX Python packages     | 🟠 Planned                                                            | -                                                                     |
| ROCm tarballs           | 🟡 In progress (missing index)                                        | 🟡 In progress (missing index)                                        |
| Native Linux packages   | 🟡 In progress ([#3333](https://github.com/ROCm/TheRock/issues/3333)) | 🟠 Planned ([#1987](https://github.com/ROCm/TheRock/issues/1987))     |

### Installing multi-arch ROCm Python packages

> [!TIP]
> We highly recommend working within a [Python virtual environment](https://docs.python.org/3/library/venv.html):
>
> ```bash
> python -m venv .venv
> source .venv/bin/activate
> ```

Install ROCm with device support for your GPU using the unified index and one or
more `device-gfx*` pip extras:

```bash
pip install --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ --pre "rocm[devel,device-gfx####]"
```

<!-- TODO: Advertise wheel variants / WheelNext once available  -->

Where `####` is the GFX target taken from the following table:

| Product Name                       | GFX Target | Device Extra     |
| ---------------------------------- | ---------- | ---------------- |
| MI300A/MI300X                      | gfx942     | `device-gfx942`  |
| MI350X/MI355X                      | gfx950     | `device-gfx950`  |
| AMD RX 7900 XTX                    | gfx1100    | `device-gfx1100` |
| AMD RX 7800 XT                     | gfx1101    | `device-gfx1101` |
| AMD RX 7700S / Framework Laptop 16 | gfx1102    | `device-gfx1102` |
| AMD Radeon 780M Laptop iGPU        | gfx1103    | `device-gfx1103` |
| AMD Strix Halo iGPU                | gfx1151    | `device-gfx1151` |
| AMD RX 9060 / XT                   | gfx1200    | `device-gfx1200` |
| AMD RX 9070 / XT                   | gfx1201    | `device-gfx1201` |

For example, install the gfx1100 device code like so:

```bash
pip install --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ --pre "rocm[devel,device-gfx1100]"
```

After installing, verify your installation:

```bash
rocm-sdk test
```

<!-- ### Installing multi-arch PyTorch Python packages -->

<!-- TODO(#3332): Document torch packages once they are available in nightly releases -->

### Installing multi-arch tarballs

Standalone "ROCm SDK tarballs" are a flattened view of ROCm
[artifacts](docs/development/artifacts.md) matching the familiar folder
structure seen with system installs on Linux to `/opt/rocm/` or on Windows via
the HIP SDK:

```
install/
  .kpack/     # GPU-specific kernel packs (multi-arch only)
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

- **Per-family tarballs** (e.g. `therock-dist-linux-gfx110X-all-7.13.0a20260427.tar.gz`)
  that include `.kpack` files only for one family.
- **Multiarch tarball** (e.g. `therock-dist-linux-multiarch-7.13.0a20260427.tar.gz`)
  that include `.kpack` files for all supported targets.

> [!NOTE]
> Multi-arch tarball releases are coming soon (index pages are not yet generated).

<!-- TODO: Document tarball download/install once index pages are generated
           and the CloudFront path is finalized. -->

## Per-family releases

Per-family releases use **GPU-family-specific index URLs** — you choose the
index URL that matches your GPU family, and all packages for that family are
served from that URL.

> [!NOTE]
> Multi-arch releases (above) are the newer approach and will soon replace
> per-family releases. Both are available during the transition.

### Installing per-family releases using pip

We recommend installing ROCm and projects like PyTorch and JAX via `pip`, the
[Python package installer](https://packaging.python.org/en/latest/guides/tool-recommendations/).

We currently support Python 3.10, 3.11, 3.12, 3.13, and 3.14 (PyTorch 2.9+ only).

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
> If you _really_ want a system-wide install, you can pass `--break-system-packages` to `pip` outside a virtual enivornment.
> In this case, commandline interface shims for executables are installed to `/usr/local/bin`, which normally has precedence over `/usr/bin` and might therefore conflict with a previous installation of ROCm.

#### Python packages release status

> [!IMPORTANT]
> Known issues with the Python wheels are tracked at
> https://github.com/ROCm/TheRock/issues/808.
>
> ⚠️ Windows packages are new and may be unstable! ⚠️

| Platform |                                                                                                                                                                                                                                         ROCm Python packages |                                                                                                                                                                                                                                               PyTorch Python packages |                                                                                                                                                                                                                                       JAX Python packages |
| -------- | -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Linux    | [![Release portable Linux packages](https://github.com/ROCm/TheRock/actions/workflows/release_portable_linux_packages.yml/badge.svg?branch=main)](https://github.com/ROCm/TheRock/actions/workflows/release_portable_linux_packages.yml?query=branch%3Amain) | [![Release Linux PyTorch Wheels](https://github.com/ROCm/TheRock/actions/workflows/release_portable_linux_pytorch_wheels.yml/badge.svg?branch=main)](https://github.com/ROCm/TheRock/actions/workflows/release_portable_linux_pytorch_wheels.yml?query=branch%3Amain) | [![Release Linux JAX Wheels](https://github.com/ROCm/TheRock/actions/workflows/release_portable_linux_jax_wheels.yml/badge.svg?branch=main)](https://github.com/ROCm/TheRock/actions/workflows/release_portable_linux_jax_wheels.yml?query=branch%3Amain) |
| Windows  |                      [![Release Windows packages](https://github.com/ROCm/TheRock/actions/workflows/release_windows_packages.yml/badge.svg?branch=main)](https://github.com/ROCm/TheRock/actions/workflows/release_windows_packages.yml?query=branch%3Amain) |             [![Release Windows PyTorch Wheels](https://github.com/ROCm/TheRock/actions/workflows/release_windows_pytorch_wheels.yml/badge.svg?branch=main)](https://github.com/ROCm/TheRock/actions/workflows/release_windows_pytorch_wheels.yml?query=branch%3Amain) |                                                                                                                                                                                                                                                         — |

#### Index page listing

For now, `rocm`, `torch`, and `jax` packages are published to GPU-architecture-specific index
pages and must be installed using an appropriate `--find-links` argument to `pip`.
They may later be pushed to the
[Python Package Index (PyPI)](https://pypi.org/) or other channels using a process
like https://wheelnext.dev/. **Please check back regularly
as these instructions will change as we migrate to official indexes and adjust
project layouts.**

| Product Name                       | GFX Target | GFX Family   | Install instructions                                                                               |
| ---------------------------------- | ---------- | ------------ | -------------------------------------------------------------------------------------------------- |
| MI300A/MI300X                      | gfx942     | gfx94X-dcgpu | [rocm](#rocm-for-gfx94X-dcgpu) // [torch](#torch-for-gfx94X-dcgpu) // [jax](#jax-for-gfx94X-dcgpu) |
| MI350X/MI355X                      | gfx950     | gfx950-dcgpu | [rocm](#rocm-for-gfx950-dcgpu) // [torch](#torch-for-gfx950-dcgpu) // [jax](#jax-for-gfx950-dcgpu) |
| AMD RX 7900 XTX                    | gfx1100    | gfx110X-all  | [rocm](#rocm-for-gfx110X-all) // [torch](#torch-for-gfx110X-all) // [jax](#jax-for-gfx110X-all)    |
| AMD RX 7800 XT                     | gfx1101    | gfx110X-all  | [rocm](#rocm-for-gfx110X-all) // [torch](#torch-for-gfx110X-all) // [jax](#jax-for-gfx110X-all)    |
| AMD RX 7700S / Framework Laptop 16 | gfx1102    | gfx110X-all  | [rocm](#rocm-for-gfx110X-all) // [torch](#torch-for-gfx110X-all) // [jax](#jax-for-gfx110X-all)    |
| AMD Radeon 780M Laptop iGPU        | gfx1103    | gfx110X-all  | [rocm](#rocm-for-gfx110X-all) // [torch](#torch-for-gfx110X-all) // [jax](#jax-for-gfx110X-all)    |
| AMD Strix Halo iGPU                | gfx1151    | gfx1151      | [rocm](#rocm-for-gfx1151) // [torch](#torch-for-gfx1151) // [jax](#jax-for-gfx1151)                |
| AMD RX 9060 / XT                   | gfx1200    | gfx120X-all  | [rocm](#rocm-for-gfx120X-all) // [torch](#torch-for-gfx120X-all) // [jax](#jax-for-gfx120X-all)    |
| AMD RX 9070 / XT                   | gfx1201    | gfx120X-all  | [rocm](#rocm-for-gfx120X-all) // [torch](#torch-for-gfx120X-all) // [jax](#jax-for-gfx120X-all)    |

#### Installing ROCm Python packages

We provide several Python packages which together form the complete ROCm SDK.

- See [ROCm Python Packaging via TheRock](./docs/packaging/python_packaging.md)
  for information about the each package.
- The packages are defined in the
  [`build_tools/packaging/python/templates/`](https://github.com/ROCm/TheRock/tree/main/build_tools/packaging/python/templates)
  directory.

| Package name         | Description                                                        |
| -------------------- | ------------------------------------------------------------------ |
| `rocm`               | Primary sdist meta package that dynamically determines other deps  |
| `rocm-sdk-core`      | OS-specific core of the ROCm SDK (e.g. compiler and utility tools) |
| `rocm-sdk-devel`     | OS-specific development tools                                      |
| `rocm-sdk-libraries` | OS-specific libraries                                              |

##### rocm for gfx94X-dcgpu

Supported devices in this family:

| Product Name  | GFX Target |
| ------------- | ---------- |
| MI300A/MI300X | gfx942     |

Install instructions:

```bash
pip install --index-url https://rocm.nightlies.amd.com/v2/gfx94X-dcgpu/ "rocm[libraries,devel]"
```

##### rocm for gfx950-dcgpu

Supported devices in this family:

| Product Name  | GFX Target |
| ------------- | ---------- |
| MI350X/MI355X | gfx950     |

Install instructions:

```bash
pip install --index-url https://rocm.nightlies.amd.com/v2/gfx950-dcgpu/ "rocm[libraries,devel]"
```

##### rocm for gfx110X-all

Supported devices in this family:

| Product Name                       | GFX Target |
| ---------------------------------- | ---------- |
| AMD RX 7900 XTX                    | gfx1100    |
| AMD RX 7800 XT                     | gfx1101    |
| AMD RX 7700S / Framework Laptop 16 | gfx1102    |
| AMD Radeon 780M Laptop iGPU        | gfx1103    |

Install instructions:

```bash
pip install --index-url https://rocm.nightlies.amd.com/v2/gfx110X-all/ "rocm[libraries,devel]"
```

##### rocm for gfx1151

Supported devices in this family:

| Product Name        | GFX Target |
| ------------------- | ---------- |
| AMD Strix Halo iGPU | gfx1151    |

Install instructions:

```bash
pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ "rocm[libraries,devel]"
```

##### rocm for gfx120X-all

Supported devices in this family:

| Product Name     | GFX Target |
| ---------------- | ---------- |
| AMD RX 9060 / XT | gfx1200    |
| AMD RX 9070 / XT | gfx1201    |

Install instructions:

```bash
pip install --index-url https://rocm.nightlies.amd.com/v2/gfx120X-all/ "rocm[libraries,devel]"
```

#### Using ROCm Python packages

After installing the ROCm Python packages, you should see them in your
environment:

```bash
pip freeze | grep rocm
# rocm==6.5.0rc20250610
# rocm-sdk-core==6.5.0rc20250610
# rocm-sdk-devel==6.5.0rc20250610
# rocm-sdk-libraries-gfx110X-all==6.5.0rc20250610
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
gfx1100;gfx1101;gfx1102
```

To initialize the `rocm[devel]` package, use the `rocm-sdk` tool to _eagerly_ expand development
contents:

```console
$ rocm-sdk init
Devel contents expanded to '.venv/lib/python3.12/site-packages/_rocm_sdk_devel'
```

These contents are useful for using the package outside of Python and _lazily_ expanded on the
first use when used from Python.

Once you have verified your installation, you can continue to use it for
standard ROCm development or install PyTorch, JAX, or another supported Python ML
framework.

#### Installing PyTorch Python packages

Using the index pages [listed above](#installing-rocm-python-packages), you can
also install `torch`, `torchaudio`, `torchvision`, and `apex`.

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
>   | 2.10          | 2.10               | 0.25                | 1.10.0       |
>   | 2.9           | 2.9                | 0.24                | 1.9.0        |
>   | 2.8           | 2.8                | 0.23                | 1.8.0        |
>
>   For example, `torch` 2.8 and compatible wheels can be installed by specifying
>
>   ```
>   torch==2.8 torchaudio==2.8 torchvision==0.23 apex==1.8.0
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

> [!TIP]
> If you previously installed PyTorch with the `pytorch-triton-rocm` package,
> please uninstall it before installing the new packages:
>
> ```bash
> pip uninstall pytorch-triton-rocm
> ```
>
> The triton package is now named `triton`.

##### torch for gfx94X-dcgpu

Supported devices in this family:

| Product Name  | GFX Target |
| ------------- | ---------- |
| MI300A/MI300X | gfx942     |

```bash
pip install --index-url https://rocm.nightlies.amd.com/v2/gfx94X-dcgpu/ torch torchaudio torchvision
# Optional additional packages on Linux:
#   apex
```

##### torch for gfx950-dcgpu

Supported devices in this family:

| Product Name  | GFX Target |
| ------------- | ---------- |
| MI350X/MI355X | gfx950     |

```bash
pip install --index-url https://rocm.nightlies.amd.com/v2/gfx950-dcgpu/ torch torchaudio torchvision
# Optional additional packages on Linux:
#   apex
```

##### torch for gfx110X-all

Supported devices in this family:

| Product Name                       | GFX Target |
| ---------------------------------- | ---------- |
| AMD RX 7900 XTX                    | gfx1100    |
| AMD RX 7800 XT                     | gfx1101    |
| AMD RX 7700S / Framework Laptop 16 | gfx1102    |
| AMD Radeon 780M Laptop iGPU        | gfx1103    |

```bash
pip install --index-url https://rocm.nightlies.amd.com/v2/gfx110X-all/ torch torchaudio torchvision
# Optional additional packages on Linux:
#   apex
```

##### torch for gfx1151

Supported devices in this family:

| Product Name        | GFX Target |
| ------------------- | ---------- |
| AMD Strix Halo iGPU | gfx1151    |

```bash
pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ torch torchaudio torchvision
# Optional additional packages on Linux:
#   apex
```

##### torch for gfx120X-all

Supported devices in this family:

| Product Name     | GFX Target |
| ---------------- | ---------- |
| AMD RX 9060 / XT | gfx1200    |
| AMD RX 9070 / XT | gfx1201    |

```bash
pip install --index-url https://rocm.nightlies.amd.com/v2/gfx120X-all/ torch torchaudio torchvision
# Optional additional packages on Linux:
#   apex
```

#### Using PyTorch Python packages

After installing the `torch` package with ROCm support, PyTorch can be used
normally:

```python
import torch

print(torch.cuda.is_available())
# True
print(torch.cuda.get_device_name(0))
# e.g. AMD Radeon Pro W7900 Dual Slot
```

See also the
[Testing the PyTorch installation](https://rocm.docs.amd.com/projects/install-on-linux/en/develop/install/3rd-party/pytorch-install.html#testing-the-pytorch-installation)
instructions in the AMD ROCm documentation.

#### Installing JAX Python packages

Using the index pages [listed above](#installing-rocm-python-packages), you can
also install `jaxlib`, `jax_rocm7_plugin`, and `jax_rocm7_pjrt`.

> [!NOTE]
> By default, pip will install the latest stable versions of each package.
>
> - If you want to install other versions, the currently supported versions are:
>
>   | jax version | jaxlib version   |
>   | ----------- | ---------------- |
>   | 0.9.1       | 0.9.1 (upstream) |
>   | 0.8.2       | 0.8.2            |
>   | 0.8.0       | 0.8.0            |
>
>   See also
>
>   - [Supported JAX versions in TheRock](https://github.com/ROCm/TheRock/tree/main/external-builds/jax#supported-jax-versions)

> [!WARNING]
> Unlike PyTorch, the JAX wheels do **not** automatically install `rocm[libraries]`
> as a dependency. You must have ROCm installed separately via a
> [tarball installation](#installing-from-tarballs).

> [!IMPORTANT]
> The `jax` package itself is **not** published to the TheRock index.
> After installing `jaxlib`, `jax_rocm7_plugin`, and `jax_rocm7_pjrt` from the
> GPU-family index, install `jax` from [PyPI](https://pypi.org/project/jax/):
>
> ```bash
> pip install jax
> ```

##### jax for gfx94X-dcgpu

Supported devices in this family:

| Product Name  | GFX Target |
| ------------- | ---------- |
| MI300A/MI300X | gfx942     |

```bash
pip install --index-url https://rocm.nightlies.amd.com/v2/gfx94X-dcgpu/ jaxlib jax_rocm7_plugin jax_rocm7_pjrt
# Install jax from PyPI
pip install jax
```

##### jax for gfx950-dcgpu

Supported devices in this family:

| Product Name  | GFX Target |
| ------------- | ---------- |
| MI350X/MI355X | gfx950     |

```bash
pip install --index-url https://rocm.nightlies.amd.com/v2/gfx950-dcgpu/ jaxlib jax_rocm7_plugin jax_rocm7_pjrt
# Install jax from PyPI
pip install jax
```

##### jax for gfx110X-all

Supported devices in this family:

| Product Name                       | GFX Target |
| ---------------------------------- | ---------- |
| AMD RX 7900 XTX                    | gfx1100    |
| AMD RX 7800 XT                     | gfx1101    |
| AMD RX 7700S / Framework Laptop 16 | gfx1102    |
| AMD Radeon 780M Laptop iGPU        | gfx1103    |

```bash
pip install --index-url https://rocm.nightlies.amd.com/v2/gfx110X-all/ jaxlib jax_rocm7_plugin jax_rocm7_pjrt
# Install jax from PyPI
pip install jax
```

##### jax for gfx1151

Supported devices in this family:

| Product Name        | GFX Target |
| ------------------- | ---------- |
| AMD Strix Halo iGPU | gfx1151    |

```bash
pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ jaxlib jax_rocm7_plugin jax_rocm7_pjrt
# Install jax from PyPI
pip install jax
```

##### jax for gfx120X-all

Supported devices in this family:

| Product Name     | GFX Target |
| ---------------- | ---------- |
| AMD RX 9060 / XT | gfx1200    |
| AMD RX 9070 / XT | gfx1201    |

```bash
pip install --index-url https://rocm.nightlies.amd.com/v2/gfx120X-all/ jaxlib jax_rocm7_plugin jax_rocm7_pjrt
# Install jax from PyPI
pip install jax
```

#### Using JAX Python packages

After installing the JAX packages with ROCm support, JAX can be used normally:

```python
import jax

print(jax.devices())
# [RocmDevice(id=0)]
```

For building JAX from source or running the full JAX test suite, see the
[external-builds/jax README](/external-builds/jax/README.md).

### Installing from tarballs

Standalone "ROCm SDK tarballs" are a flattened view of ROCm
[artifacts](docs/development/artifacts.md) matching the familiar folder
structure seen with system installs on Linux to `/opt/rocm/` or on Windows via
the HIP SDK:

```bash
install/  # Extracted tarball location, file path of your choosing
  .info/
  bin/
  clients/
  include/
  lib/
  libexec/
  share/
```

Tarballs are _just_ these raw files. They do not come with "install" steps
such as setting environment variables.

> [!WARNING]
> Tarballs and per-commit CI artifacts are primarily intended for developers
> and CI workflows.
>
> For most users, we recommend installing via package managers:
>
> - [Installing multi-arch releases using pip](#installing-multi-arch-rocm-python-packages)
> - [Installing per-family releases using pip](#installing-per-family-releases-using-pip)
> - [Installing from native packages](#installing-from-native-packages)

#### Browsing release tarballs

Release tarballs are uploaded to the following locations:

| Tarball index                             | S3 bucket                                                                                | Description                                        |
| ----------------------------------------- | ---------------------------------------------------------------------------------------- | -------------------------------------------------- |
| https://repo.amd.com/rocm/tarball/        | (not publicly accessible)                                                                | Stable releases                                    |
| https://rocm.nightlies.amd.com/tarball/   | [`therock-nightly-tarball`](https://therock-nightly-tarball.s3.amazonaws.com/index.html) | Nightly builds from the default development branch |
| https://rocm.prereleases.amd.com/tarball/ | (not publicly accessible)                                                                | ⚠️ Prerelease builds for QA testing ⚠️             |
| https://rocm.devreleases.amd.com/tarball/ | [`therock-dev-tarball`](https://therock-dev-tarball.s3.amazonaws.com/index.html)         | ⚠️ Development builds from project maintainers ⚠️  |

#### Manual tarball extraction

To download a tarball and extract it into place manually:

```bash
mkdir therock-tarball && cd therock-tarball
# For example...
wget https://rocm.nightlies.amd.com/tarball/therock-dist-linux-gfx110X-all-7.12.0a20260202.tar.gz
mkdir install && tar -xf *.tar.gz -C install
```

#### Automated tarball extraction

For more control over artifact installation—including per-commit CI builds,
specific release versions, the latest nightly release, and component
selection—see the
[Installing Artifacts](docs/development/installing_artifacts.md) developer
documentation. The
[`install_rocm_from_artifacts.py`](build_tools/install_rocm_from_artifacts.py)
script can be used to install artifacts from a variety of sources.

#### Using installed tarballs

After installing (downloading and extracting) a tarball, you can test it by
running programs from the `bin/` directory:

```bash
ls install
# bin  include  lib  libexec  llvm  share

# Now test some of the installed tools:
./install/bin/rocminfo
./install/bin/test_hip_api
```

> [!TIP]
> You may also want to add parts of the install directory to your `PATH` or set
> other environment variables like `ROCM_HOME`.
>
> See also [this issue](https://github.com/ROCm/TheRock/issues/1658) discussing
> relevant environment variables.

> [!TIP]
> After extracting a tarball, metadata about which commits were used to build
> TheRock can be found in the `share/therock/therock_manifest.json` file:
>
> ```bash
> cat install/share/therock/therock_manifest.json
> # {
> #   "the_rock_commit": "567dd890a3bc3261ffb26ae38b582378df298374",
> #   "submodules": [
> #     {
> #       "submodule_name": "half",
> #       "submodule_path": "base/half",
> #       "submodule_url": "https://github.com/ROCm/half.git",
> #       "pin_sha": "207ee58595a64b5c4a70df221f1e6e704b807811",
> #       "patches": []
> #     },
> #     ...
> ```

### Installing from native packages

In addition to Python wheels and tarballs, ROCm native Linux packages are
published for Debian-based and RPM-based distributions.

> [!WARNING]
> These builds are primarily intended for development and testing and are currently **unsigned**.

#### Native packages release status

| Platform |                                                                                                                                                                                                                                  Native packages |
| -------- | -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Linux    | [![Build Native Linux Packages](https://github.com/ROCm/TheRock/actions/workflows/build_native_linux_packages.yml/badge.svg?branch=main)](https://github.com/ROCm/TheRock/actions/workflows/build_native_linux_packages.yml?query=branch%3Amain) |
| Windows  |                                                                                                                                                                                                                                    (Coming soon) |

#### GPU family and package mapping

| Product Name                       | GFX Target | GFX Family | Runtime Package | Development Package      |
| ---------------------------------- | ---------- | ---------- | --------------- | ------------------------ |
| MI300A/MI300X                      | gfx942     | gfx94X     | amdrocm-gfx94x  | amdrocm-core-sdk-gfx94x  |
| MI350X/MI355X                      | gfx950     | gfx950     | amdrocm-gfx950  | amdrocm-core-sdk-gfx950  |
| AMD RX 7900 XTX                    | gfx1100    | gfx110x    | amdrocm-gfx110x | amdrocm-core-sdk-gfx110x |
| AMD RX 7800 XT                     | gfx1101    | gfx110x    | amdrocm-gfx110x | amdrocm-core-sdk-gfx110x |
| AMD RX 7700S / Framework Laptop 16 | gfx1102    | gfx110x    | amdrocm-gfx110x | amdrocm-core-sdk-gfx110x |
| AMD Radeon 780M Laptop iGPU        | gfx1103    | gfx110x    | amdrocm-gfx110x | amdrocm-core-sdk-gfx110x |
| AMD Strix Point iGPU               | gfx1150    | gfx1150    | amdrocm-gfx1150 | amdrocm-core-sdk-gfx1150 |
| AMD Strix Halo iGPU                | gfx1151    | gfx1151    | amdrocm-gfx1151 | amdrocm-core-sdk-gfx1151 |
| AMD Fire Range iGPU                | gfx1152    | gfx1152    | amdrocm-gfx1152 | amdrocm-core-sdk-gfx1152 |
| AMD Strix Halo XT                  | gfx1153    | gfx1153    | amdrocm-gfx1153 | amdrocm-core-sdk-gfx1153 |
| AMD RX 9060 / XT                   | gfx1200    | gfx120X    | amdrocm-gfx120x | amdrocm-core-sdk-gfx120x |
| AMD RX 9070 / XT                   | gfx1201    | gfx120X    | amdrocm-gfx120x | amdrocm-core-sdk-gfx120x |
| Radeon VII                         | gfx906     | gfx906     | amdrocm-gfx906  | amdrocm-core-sdk-gfx906  |
| MI100                              | gfx908     | gfx908     | amdrocm-gfx908  | amdrocm-core-sdk-gfx908  |
| MI200 series                       | gfx90a     | gfx90a     | amdrocm-gfx90a  | amdrocm-core-sdk-gfx90a  |
| AMD RX 5700 XT                     | gfx1010    | gfx101x    | amdrocm-gfx101x | amdrocm-core-sdk-gfx101x |
| AMD RX 6900 XT                     | gfx1030    | gfx103x    | amdrocm-gfx103x | amdrocm-core-sdk-gfx103x |
| AMD RX 6800 XT                     | gfx1031    | gfx103x    | amdrocm-gfx103x | amdrocm-core-sdk-gfx103x |

> [!TIP]
> To find the latest available release:
>
> - **Step 1**: Browse the index pages:
>   - **Debian packages**: https://rocm.nightlies.amd.com/deb/
>   - **RPM packages**: https://rocm.nightlies.amd.com/rpm/
> - **Step 2**: Look for directories in the format `YYYYMMDD-<action-run-id>` (e.g., `20260310-12345678`)
> - **Step 3**: Use the latest date in the installation commands below

#### Installing on Debian-based systems (Ubuntu, Debian, etc.)

```bash
# Step 1: Find the latest release from https://rocm.nightlies.amd.com/deb/
#         Look for directories like "20260310-12345678"
# Step 2: Look at the "GPU family and package mapping" table above to find
#         the GFX Family for your GPU (e.g., gfx94x, gfx110x, gfx1151)
# Step 3: Set the variables below

export RELEASE_ID=20260310-12345678  # Replace with actual date-runid
export GFX_ARCH=gfx110x              # Replace with GFX Family from the mapping table

# Step 4: Add repository and install
sudo apt update
sudo apt install -y ca-certificates
echo "deb [trusted=yes] https://rocm.nightlies.amd.com/deb/${RELEASE_ID} stable main" \
  | sudo tee /etc/apt/sources.list.d/rocm-nightly.list
sudo apt update
sudo apt install amdrocm-core-sdk-${GFX_ARCH}
# If only runtime is needed, install amdrocm-${GFX_ARCH} instead
```

#### Installing on RPM-based systems (RHEL, SLES, AlmaLinux etc.)

> [!NOTE]
> The following instructions are for RHEL-based operating systems.

```bash
# Step 1: Find the latest release from https://rocm.nightlies.amd.com/rpm/
#         Look for directories like "20260310-12345678"
# Step 2: Look at the "GPU family and package mapping" table above to find
#         the GFX Family for your GPU (e.g., gfx94x, gfx110x, gfx1151)
# Step 3: Set the variables below

export RELEASE_ID=20260310-12345678  # Replace with actual date-runid
export GFX_ARCH=gfx110x              # Replace with GFX Family from the mapping table

# Step 4: Add repository and install
sudo dnf install -y ca-certificates
sudo tee /etc/yum.repos.d/rocm-nightly.repo <<EOF
[rocm-nightly]
name=ROCm Nightly Repository
baseurl=https://rocm.nightlies.amd.com/rpm/${RELEASE_ID}/x86_64
enabled=1
gpgcheck=0
priority=50
EOF
sudo dnf install amdrocm-core-sdk-${GFX_ARCH}
# If only runtime is needed, install amdrocm-${GFX_ARCH} instead
```

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
  on Linux or [Adrenaline driver](https://www.amd.com/en/products/software/adrenalin.html) on Windows
- For platform-specific troubleshooting when using PyTorch or JAX, see:
  - [Using ROCm Python packages](#using-rocm-python-packages)
  - [Using PyTorch Python packages](#using-pytorch-python-packages)
  - [Using JAX Python packages](#using-jax-python-packages)
