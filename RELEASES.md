# Releases

This page describes how to install and use our release artifacts for ROCm and
external builds like PyTorch. We produce build artifacts as part of our
Continuous Integration (CI) build/test workflows as well as release artifacts as
part of Continuous Delivery (CD) nightly releases. For the development-status of GPU architecture support in TheRock, please see the [SUPPORTED_GPUS.md](./SUPPORTED_GPUS.md) document, which tracks readiness and onboarding progress for each AMD GPU architecture.

See also the
[Roadmap for support](ROADMAP.md) and
[Build artifacts overview](docs/development/artifacts.md) pages.

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

- [Installing releases using pip](#installing-releases-using-pip)
  - [Python packages release status](#python-packages-release-status)
  - [Installing ROCm Python packages](#installing-rocm-python-packages)
  - [Using ROCm Python packages](#using-rocm-python-packages)
  - [Installing PyTorch Python packages](#installing-pytorch-python-packages)
  - [Using PyTorch Python packages](#using-pytorch-python-packages)
- [Installing from tarballs](#installing-from-tarballs)
  - [Browsing release tarballs](#browsing-release-tarballs)
  - [Manual tarball extraction](#manual-tarball-extraction)
  - [Automated tarball extraction](#automated-tarball-extraction)
  - [Using installed tarballs](#using-installed-tarballs)
- [Verifying your installation](#verifying-your-installation)

## Installing releases using pip

We recommend installing ROCm and projects like PyTorch via `pip`, the
[Python package installer](https://packaging.python.org/en/latest/guides/tool-recommendations/).

We currently support Python 3.11, 3.12, and 3.13.

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

### Python packages release status

> [!IMPORTANT]
> Known issues with the Python wheels are tracked at
> https://github.com/ROCm/TheRock/issues/808.
>
> ⚠️ Windows packages are new and may be unstable! ⚠️

| Platform |                                                                                                                                                                                                                                         ROCm Python packages |                                                                                                                                                                                                                                               PyTorch Python packages |
| -------- | -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Linux    | [![Release portable Linux packages](https://github.com/ROCm/TheRock/actions/workflows/release_portable_linux_packages.yml/badge.svg?branch=main)](https://github.com/ROCm/TheRock/actions/workflows/release_portable_linux_packages.yml?query=branch%3Amain) | [![Release Linux PyTorch Wheels](https://github.com/ROCm/TheRock/actions/workflows/release_portable_linux_pytorch_wheels.yml/badge.svg?branch=main)](https://github.com/ROCm/TheRock/actions/workflows/release_portable_linux_pytorch_wheels.yml?query=branch%3Amain) |
| Windows  |                      [![Release Windows packages](https://github.com/ROCm/TheRock/actions/workflows/release_windows_packages.yml/badge.svg?branch=main)](https://github.com/ROCm/TheRock/actions/workflows/release_windows_packages.yml?query=branch%3Amain) |             [![Release Windows PyTorch Wheels](https://github.com/ROCm/TheRock/actions/workflows/release_windows_pytorch_wheels.yml/badge.svg?branch=main)](https://github.com/ROCm/TheRock/actions/workflows/release_windows_pytorch_wheels.yml?query=branch%3Amain) |

### Index page listing

For now, `rocm` and `torch` packages are published to GPU-architecture-specific index
pages and must be installed using an appropriate `--find-links` argument to `pip`.
They may later be pushed to the
[Python Package Index (PyPI)](https://pypi.org/) or other channels using a process
like https://wheelnext.dev/. **Please check back regularly
as these instructions will change as we migrate to official indexes and adjust
project layouts.**

| Product Name                       | GFX Target | GFX Family   | Install instructions                                               |
| ---------------------------------- | ---------- | ------------ | ------------------------------------------------------------------ |
| MI300A/MI300X                      | gfx942     | gfx94X-dcgpu | [rocm](#rocm-for-gfx94X-dcgpu) // [torch](#torch-for-gfx94X-dcgpu) |
| MI350X/MI355X                      | gfx950     | gfx950-dcgpu | [rocm](#rocm-for-gfx950-dcgpu) // [torch](#torch-for-gfx950-dcgpu) |
| AMD RX 7900 XTX                    | gfx1100    | gfx110X-all  | [rocm](#rocm-for-gfx110X-all) // [torch](#torch-for-gfx110X-all)   |
| AMD RX 7800 XT                     | gfx1101    | gfx110X-all  | [rocm](#rocm-for-gfx110X-all) // [torch](#torch-for-gfx110X-all)   |
| AMD RX 7700S / Framework Laptop 16 | gfx1102    | gfx110X-all  | [rocm](#rocm-for-gfx110X-all) // [torch](#torch-for-gfx110X-all)   |
| AMD Radeon 780M Laptop iGPU        | gfx1103    | gfx110X-all  | [rocm](#rocm-for-gfx110X-all) // [torch](#torch-for-gfx110X-all)   |
| AMD Strix Halo iGPU                | gfx1151    | gfx1151      | [rocm](#rocm-for-gfx1151) // [torch](#torch-for-gfx1151)           |
| AMD RX 9060 / XT                   | gfx1200    | gfx120X-all  | [rocm](#rocm-for-gfx120X-all) // [torch](#torch-for-gfx120X-all)   |
| AMD RX 9070 / XT                   | gfx1201    | gfx120X-all  | [rocm](#rocm-for-gfx120X-all) // [torch](#torch-for-gfx120X-all)   |

### Installing ROCm Python packages

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

#### rocm for gfx94X-dcgpu

Supported devices in this family:

| Product Name  | GFX Target |
| ------------- | ---------- |
| MI300A/MI300X | gfx942     |

Install instructions:

```bash
pip install --index-url https://rocm.nightlies.amd.com/v2/gfx94X-dcgpu/ "rocm[libraries,devel]"
```

#### rocm for gfx950-dcgpu

Supported devices in this family:

| Product Name  | GFX Target |
| ------------- | ---------- |
| MI350X/MI355X | gfx950     |

Install instructions:

```bash
pip install --index-url https://rocm.nightlies.amd.com/v2/gfx950-dcgpu/ "rocm[libraries,devel]"
```

#### rocm for gfx110X-all

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

#### rocm for gfx1151

Supported devices in this family:

| Product Name        | GFX Target |
| ------------------- | ---------- |
| AMD Strix Halo iGPU | gfx1151    |

Install instructions:

```bash
pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ "rocm[libraries,devel]"
```

#### rocm for gfx120X-all

Supported devices in this family:

| Product Name     | GFX Target |
| ---------------- | ---------- |
| AMD RX 9060 / XT | gfx1200    |
| AMD RX 9070 / XT | gfx1201    |

Install instructions:

```bash
pip install --index-url https://rocm.nightlies.amd.com/v2/gfx120X-all/ "rocm[libraries,devel]"
```

### Using ROCm Python packages

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
standard ROCm development or install PyTorch or another supported Python ML
framework.

### Installing PyTorch Python packages

Using the index pages [listed above](#installing-rocm-python-packages), you can
also install `torch`, `torchaudio`, and `torchvision`.

> [!NOTE]
> By default, pip will install the latest stable versions of each package.
>
> - If you want to allow installing prerelease versions, use the `--pre`
>
> - If you want to install other versions, take note of the compatibility
>   matrix:
>
>   | torch version | torchaudio version | torchvision version |
>   | ------------- | ------------------ | ------------------- |
>   | 2.10          | 2.10               | 0.25                |
>   | 2.9           | 2.9                | 0.24                |
>   | 2.8           | 2.8                | 0.23                |
>   | 2.7           | 2.7.1a0            | 0.22.1              |
>
>   For example, `torch` 2.7.1 and compatible wheels can be installed by specifying
>
>   ```
>   torch==2.7.1 torchaudio==2.7.1a0 torchvision==0.22.1
>   ```
>
>   See also
>
>   - [Supported PyTorch versions in TheRock](https://github.com/ROCm/TheRock/tree/main/external-builds/pytorch#supported-pytorch-versions)
>   - [Installing previous versions of PyTorch](https://pytorch.org/get-started/previous-versions/)
>   - [torchvision installation - compatibility matrix](https://github.com/pytorch/vision?tab=readme-ov-file#installation)
>   - [torchaudio installation - compatibility matrix](https://docs.pytorch.org/audio/main/installation.html#compatibility-matrix)

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

#### torch for gfx94X-dcgpu

Supported devices in this family:

| Product Name  | GFX Target |
| ------------- | ---------- |
| MI300A/MI300X | gfx942     |

```bash
pip install --index-url https://rocm.nightlies.amd.com/v2/gfx94X-dcgpu/ torch torchaudio torchvision
```

#### torch for gfx950-dcgpu

Supported devices in this family:

| Product Name  | GFX Target |
| ------------- | ---------- |
| MI350X/MI355X | gfx950     |

```bash
pip install --index-url https://rocm.nightlies.amd.com/v2/gfx950-dcgpu/ torch torchaudio torchvision
```

#### torch for gfx110X-all

Supported devices in this family:

| Product Name                       | GFX Target |
| ---------------------------------- | ---------- |
| AMD RX 7900 XTX                    | gfx1100    |
| AMD RX 7800 XT                     | gfx1101    |
| AMD RX 7700S / Framework Laptop 16 | gfx1102    |
| AMD Radeon 780M Laptop iGPU        | gfx1103    |

```bash
pip install --index-url https://rocm.nightlies.amd.com/v2/gfx110X-all/ torch torchaudio torchvision
```

#### torch for gfx1151

Supported devices in this family:

| Product Name        | GFX Target |
| ------------------- | ---------- |
| AMD Strix Halo iGPU | gfx1151    |

```bash
pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ torch torchaudio torchvision
```

#### torch for gfx120X-all

Supported devices in this family:

| Product Name     | GFX Target |
| ---------------- | ---------- |
| AMD RX 9060 / XT | gfx1200    |
| AMD RX 9070 / XT | gfx1201    |

```bash
pip install --index-url https://rocm.nightlies.amd.com/v2/gfx120X-all/ torch torchaudio torchvision
```

### Using PyTorch Python packages

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

## Installing from tarballs

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
> - [Installing releases using pip](#installing-releases-using-pip)
> - (TODO) Installing native Linux deb/RPM packages

### Browsing release tarballs

Release tarballs are uploaded to the following locations:

| Tarball index                             | S3 bucket                                                                                | Description                                        |
| ----------------------------------------- | ---------------------------------------------------------------------------------------- | -------------------------------------------------- |
| https://repo.amd.com/rocm/tarball/        | (not publicly accessible)                                                                | Stable releases                                    |
| https://rocm.nightlies.amd.com/tarball/   | [`therock-nightly-tarball`](https://therock-nightly-tarball.s3.amazonaws.com/index.html) | Nightly builds from the default development branch |
| https://rocm.prereleases.amd.com/tarball/ | (not publicly accessible)                                                                | ⚠️ Prerelease builds for QA testing ⚠️             |
| https://rocm.devreleases.amd.com/tarball/ | [`therock-dev-tarball`](https://therock-dev-tarball.s3.amazonaws.com/index.html)         | ⚠️ Development builds from project maintainers ⚠️  |

### Manual tarball extraction

To download a tarball and extract it into place manually:

```bash
mkdir therock-tarball && cd therock-tarball
# For example...
wget https://rocm.nightlies.amd.com/tarball/therock-dist-linux-gfx110X-all-7.12.0a20260202.tar.gz
mkdir install && tar -xf *.tar.gz -C install
```

### Automated tarball extraction

For more control over artifact installation—including per-commit CI builds,
specific release versions, the latest nightly release, and component
selection—see the
[Installing Artifacts](docs/development/installing_artifacts.md) developer
documentation. The
[`install_rocm_from_artifacts.py`](build_tools/install_rocm_from_artifacts.py)
script can be used to install artifacts from a variety of sources.

### Using installed tarballs

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

## Verifying your installation

After installing ROCm via either pip packages or tarballs, you can verify that
your GPU is properly recognized.

### Linux

Run one of the following commands to verify that your GPU is detected and properly
initialized by the ROCm stack:

```bash
rocminfo
# or
amd-smi
```

### Windows

Run the following command to verify GPU detection:

```bash
hipInfo.exe
```

### Additional troubleshooting

If your GPU is not recognized or you encounter issues:

- **Linux users**: Check system logs using `dmesg | grep amdgpu` for specific error messages
- Review memory allocation settings (see the [FAQ](https://github.com/ROCm/TheRock/blob/main/faq.md)
  for GTT configuration on unified memory systems)
- Ensure you have the latest [AMDGPU driver](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html#amdgpu-driver-installation)
  on Linux or [Adrenaline driver](https://www.amd.com/en/products/software/adrenalin.html) on Windows
- For platform-specific troubleshooting when using PyTorch, see:
  - [Using ROCm Python packages](#using-rocm-python-packages)
  - [Using PyTorch Python packages](#using-pytorch-python-packages)
