---
author: Jason Bonnell (jbonnell-amd)
created: 2025-12-15
modified: 2025-12-15
status: draft
---

# Optiq Integration into TheRock

This RFC proposes the integration of [roc-optiq](https://github.com/ROCm/roc-optiq), a graph-based deep learning library for AMD GPUs, into TheRock's build system and CI/CD infrastructure.

## Overview

roc-optiq is a visualizer tool for the ROCm profiler tools. It is deployed as a single executable.

## Platform(s)

Windows and Linux

## Dependencies

As a first step, we'll need to migrate roc-optiq to the rocm-systems super-repo

### Build-time Dependencies

- Vulkan SDK (currently, we're using version 1.4.328.1)
- Linux Packages (see Ubuntu example below)
  ```
  apt install -y build-essential cmake git libncurses-dev libwayland-dev \
      libx11-dev libxcursor-dev libxi-dev libxinerama-dev libxkbcommon-dev \
      libxrandr-dev pkg-config wget xorg-dev libdbus-1-dev
  ```
- See example CI workflows from the roc-optiq repo for [Linux (Ubuntu)](https://github.com/ROCm/roc-optiq/blob/main/.github/workflows/ubuntu-ci.yml), [Linux (RedHat)](https://github.com/ROCm/roc-optiq/blob/main/.github/workflows/redhat-ci.yml), and [Windows](https://github.com/ROCm/roc-optiq/blob/main/.github/workflows/windows-ci.yml) for an idea on how we are currently building the executable for those platforms

### Run-time Dependencies

- ROCm profiler tools (already included in TheRock)
- Being able to launch a GUI application (cannot be run on a headless server)

## Deployment

The executable can be deployed via a build artifact from TheRock. Additionally, we would like to have the following additional deployment options:

- repo.radeon.com
  - .deb and .rpm packages for Linux
  - .exe for Windows
- Installshield installer for Windows
  - Unsure if this can be done via TheRock, or if it will need to be handled elsewhere

## Current Status and Roadmap

### Current Capabilities

- roc-optiq standalone repository can deploy releases with .exe, .rpm, .deb packages for Linux and Windows

### Near-Term Roadmap

- Migrate from the standalone roc-optiq repository to the rocm-systems super-repo
- Integrate the build for roc-optiq within TheRock repository
- Ensure release parity with what is currently offered in the standalone repository

### Long-Term Roadmap

- ???
