---
author: Liam Berry (LiamfBerry), Saad Rahim (saadrahim)
created: 2025-11-14
modified: 2026-01-12
status: draft
---

# ROCm software ecosystem package repository structure

## Overview

ROCm software ecosystem spans the ROCm Core SDK, expansions like ROCm-DS, the AMD GPU driver and standalone projects like RVS. Software packaging for this ecosystem needs a well defined hierarchy reflected in the package distribution folder structure on repo.amd.com.

## Repository Structure

`builds.amd.com` / `repo.amd.com` will have the following folder structure:

- **amdgpu** *(reserved for future use)*  
- **archives** *(unmaintained, for reference only)*  
- **rocm-ecosystem**
  - **nightly** *(Retention policy: 30 dev, 120 nightly — builds.amd.com)*
    - **pyindex** *(central nightly wheel repository for all ROCm components)*
    - **core**
      - tarball
      - whl
      - packages
    - **Linux Distros [a–z]**
    - **windows**
      - MSI and EXE files for Windows
    - **expansions [a–z]**
      - tarball
      - whl
      - packages
    - **TBD** *(extras or individual releases)*
      - Structure options:
        - Flat or per-project folders  
        - Per-project allows S3 bucket permission granularity by group but complicates duplication on `repo.amd.com`
      - Flat structure:
        - All projects share structure
        - Each project has its own S3 bucket
        - All projects must have unique product names (→ unique artifact names)
      - tarball
      - whl
      - packages
    - **pytorch**
      - tarball *(maybe?)*
      - whl
    - **jax**
    - **onnx-runtime**

  - **prerelease** *(Retention policy: 2 years — builds.amd.com)*
    - Mirrors nightly folder structure
    - Tested by QA
    - Must match structure of `repo.amd.com`

  - **stable** *(repo.amd.com)* or **sts** *(short-term support)*
    - Current ROCm Core release from TheRock

  - **lts** *(repo.amd.com)*
    - `YYYYMM`
      - Mirrors stable folder structure
