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

* nightly - nightly builds from the develop branch
* prerelease - builds from the release candidate branches
* stable - GA releases of ROCm with a short term support lifecycle, tagged as ROCm releases.
* lts - Future long term stability (LTS) releases
* expansions - SDK built with dependencies on the ROCm Core SDK
* pyindex - folder name for a central repository for python packages

## Repository Structure

`repo.amd.com` will have the following folder structure:

- **amdgpu** *(reserved for future use)*
- **amd-repos**
     - packages
       - **Linux Distros [a–z]** 
- **archives** *(unmaintained releasees, for reference only)*
- **rocm** (current rocm folder with non production releases, move to archives in 6 months)
- **rocm-platform**
  - **standard**
    - **nightly** *(Retention policy: 30 dev, 120 nightly)*
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

  - **prerelease** *(Retention policy: 2 years)*
    - Mirrors nightly folder structure
    - Tested by QA
    - Must match structure of `repo.amd.com`

  - **stable**
    - Current ROCm Core release from TheRock
    - **standard** (includes default build packages, asan build packages, default-debug symbol packages, and asan-debug system packages)
    - **rpath** (includes rpath variant of standard packages)

  - **lts**
    - `YYYYMM`
      - Mirrors stable folder structure

## Repository Package

Allow users to install the ROCm repositories via a convienient package. The package provides
all the repository files. Example, the rpm repo file will add files to /etc/yum.repos.d/ and
debian repo file will add to /etc/apt/sources.list.d/. The
repo file will also install the gpg key, ideally prompting the user to accept the key. Updating
the repo file will update the gpg key to the latest.

- amdrocm-repo.rpm
- amdrocm-repo-rpath.rpm # rpath is only available for rpm based releases today
- amdrocm-repo.deb
- amdrocm-repo-lts-YYYYMM.rpm #reserved for future LTS release streams

The rocm-repo file uses an environment variable, $rocm_release_type, to identify the release type. 
This to allow users to manually switch to nightly or prerelease repositories.

The repository package is to include the latest amdgpu driver folder from repo.radeon.com.
This is temporary until amdgpu is moved to repo.amd.com.

This the repo packages are published in the amd-repos folder in the repo.amd.com hierarchy.
The repo packages adds the amd-repos repository as well.

It is designed to work with the following commands
```
wget https://repo.amd.com/amd-repos/$OS/rocm-repo.rpm
rpm -ivh rocm-repo.rpm

or directly via package manager
yum install https://repo.amd.com/amd-repos/$OS/rocm-repo.rpm
```
