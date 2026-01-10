---
author: Jason Bonnell (jbonnell-amd)
created: 2025-12-15
modified: 2025-12-16
status: draft
---

# Optiq Integration into TheRock

This RFC proposes the integration of [roc-optiq](https://github.com/ROCm/roc-optiq) into TheRock's build system and CI/CD infrastructure.

## Overview

roc-optiq is a visualizer tool for the ROCm profiler tools. Since this would be the first GUI application to be shipped with TheRock, there are many considerations needed on how best to approach this.

## Platform(s)

Windows and Linux

## Dependencies

As a first step, we'll need to migrate roc-optiq to the rocm-systems super-repo.

Afterwards, we can create a new directory (something such as `/gui` or `/visualization-tools`) to house the `CMakeLists.txt` and `artifact.toml` files needed to build the project.

### Build-time Dependencies

- **Vulkan SDK (Windows and Linux)**

  - We're currently using version 1.4.328.1, downloaded from [vulkan.lunarg.com](https://vulkan.lunarg.com/sdk/home)
    - Windows: https://sdk.lunarg.com/sdk/download/1.4.328.1/windows/vulkansdk-windows-X64-1.4.328.1.exe
      - File Size: 240MB
    - Linux: https://sdk.lunarg.com/sdk/download/1.4.328.1/linux/vulkansdk-linux-x86_64-1.4.328.1.tar.xz
      - File Size: 312MB
  - Will need to be added to `third-party/vulkan-sdk`

- **Linux Packages**

  - See an example from our Ubuntu workflow below
    ```
    apt install -y build-essential cmake git libncurses-dev libwayland-dev \
        libx11-dev libxcursor-dev libxi-dev libxinerama-dev libxkbcommon-dev \
        libxrandr-dev pkg-config wget xorg-dev libdbus-1-dev
    ```
  - All `*-dev` package will likely need to be built from source via `third-party` directory
  - Output of LDD on the roc-optiq executable for Ubuntu 22.04

  ```
  root@c875277ff213:/home/roc-optiq/build# ldd roc-optiq
        linux-vdso.so.1 (0x00007ffe4d644000)
        libglfw.so.3 => not found
        libvulkan.so.1 => /opt/vulkan/1.4.328.1/x86_64/lib/libvulkan.so.1 (0x00007f0027c0b000)
        libdbus-1.so.3 => /lib/x86_64-linux-gnu/libdbus-1.so.3 (0x00007f0027bbd000)
        libstdc++.so.6 => /lib/x86_64-linux-gnu/libstdc++.so.6 (0x00007f0027991000)
        libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x00007f00278aa000)
        libgcc_s.so.1 => /lib/x86_64-linux-gnu/libgcc_s.so.1 (0x00007f0027888000)
        libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007f002765f000)
        /lib64/ld-linux-x86-64.so.2 (0x00007f0028372000)
        libdl.so.2 => /lib/x86_64-linux-gnu/libdl.so.2 (0x00007f002765a000)
        libpthread.so.0 => /lib/x86_64-linux-gnu/libpthread.so.0 (0x00007f0027655000)
        libsystemd.so.0 => /lib/x86_64-linux-gnu/libsystemd.so.0 (0x00007f002758e000)
        liblzma.so.5 => /lib/x86_64-linux-gnu/liblzma.so.5 (0x00007f0027561000)
        libzstd.so.1 => /lib/x86_64-linux-gnu/libzstd.so.1 (0x00007f0027492000)
        liblz4.so.1 => /lib/x86_64-linux-gnu/liblz4.so.1 (0x00007f0027472000)
        libcap.so.2 => /lib/x86_64-linux-gnu/libcap.so.2 (0x00007f0027467000)
        libgcrypt.so.20 => /lib/x86_64-linux-gnu/libgcrypt.so.20 (0x00007f0027329000)
        libgpg-error.so.0 => /lib/x86_64-linux-gnu/libgpg-error.so.0 (0x00007f0027303000)
  ```

  - Some of these packages are only needed for tests (determine which ones)
  - See example CI workflows from the roc-optiq repo for [Linux (Ubuntu)](https://github.com/ROCm/roc-optiq/blob/main/.github/workflows/ubuntu-ci.yml), [Linux (RedHat)](https://github.com/ROCm/roc-optiq/blob/main/.github/workflows/redhat-ci.yml), and [Windows](https://github.com/ROCm/roc-optiq/blob/main/.github/workflows/windows-ci.yml) for an idea on how we are currently building the executable for those platforms

### Run-time Dependencies

- GLFW
  - On Windows, this is distributed as `glfw3.dll` via the installer
  - On Linux, this dependency is handled as part of the package installation
- Being able to launch a GUI application (cannot be run on a headless server)

There are plans to potentially integrate with rocprofiler-compute analysis scripts in the future, which may result in a rocprofiler-compute runtime dependency being added at a later date.

## Deployment

The executable can be deployed via a build artifact from TheRock. Additionally, we would like to have the following additional deployment options:

- repo.radeon.com
  - .deb and .rpm packages for Linux
  - .exe for Windows
- Installshield installer for Windows
  - Unsure if this can be done via TheRock, or if it will need to be handled elsewhere

Since this will be the first GUI based inclusion in TheRock, there are additional packaging considerations

- Should we disable building `roq-optiq` unless explitically specified that we want `gui` packages included in the build?
  - For example, adding a flag such as `-DTHEROCK_ENABLE_GUI_BUILDS` that is OFF by default
- We will need to sign the Windows application being packaged, otherwise we will be delivering an unsigned executable as part of the build process

## Current Status and Roadmap

### Current Capabilities

- roc-optiq standalone repository can deploy releases with .exe, .rpm, .deb packages for Linux and Windows

### Near-Term Roadmap

- Migrate from the standalone roc-optiq repository to the rocm-systems super-repo
- Integrate the build for roc-optiq within TheRock repository
- Ensure release parity with what is currently offered in the standalone repository

### Long-Term Roadmap

- ???
