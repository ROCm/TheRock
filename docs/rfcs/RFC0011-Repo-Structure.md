---
author: Saad Rahim (saadrahim)
created: 2026-04-08
modified: 2026-04-08
status: draft
---

# ROCm software ecosystem package repository structure

## Overview

repo.amd.com's open source software release publications need standardization. In scope is the ROCm software ecosystem which spans the ROCm Core SDK, expansions like ROCm-DS, and standalone projects like RVS. Software packaging for this ecosystem needs a well defined hierarchy reflected in the package distribution folder structure. As software is published on repo.amd.com, the planned hierarchy must be extensible by other software ecosystem published by AMD. As a result, this proposal includes the ability to add the AMD GPU driver to this structure in the future.

## Definitions

*
* nightly - nightly builds from the develop branch
* prerelease - builds from the release candidate branches
* stable - GA releases of ROCm with a short term support lifecycle, tagged as ROCm releases.
* lts - Future long term stability (LTS) releases
* expansions - SDK built with dependencies on the ROCm Core SDK
* pyindex - folder name for a central repository for python packages

## Repository Structure

`repo.amd.com` will have the following folder structure:

- **amdgpu** *(reserved for future use)*  
- **archives** *(unmaintained releasees, for reference only)*
- **rocm** (current rocm folder with non production releases, move to archives in 6 months)
- **rocm-ecosystem**
  - **nightly** *(Retention policy: 30 dev, 120 nightly — builds.amd.com)*
    - **pyindex** *(central nightly wheel repository for all ROCm components)*
    - **core**
      - tarball
      - zip
      
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
