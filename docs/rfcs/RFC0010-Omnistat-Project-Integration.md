---
author: Jason Bonnell (jbonnell-amd)
created: 2026-01-21
modified: 2026-01-21
status: draft
---

# Omnistat Integration into TheRock

This RFC proposes the integration of [omnistat](https://github.com/ROCm/omnistat) into TheRock's build system and CI/CD infrastructure.

## Overview

Omnistat is a set of Python utilities and data collectors to support scale-out cluster telemtry targeting AMD Instinct™ MI accelerators.

# Platform(s)

Linux

# Dependencies

- rocprofiler-sdk
- rocm-smi
- amdsmi
- python3.8 or above
  - pip packages (see [requirementx.txt](https://github.com/ROCm/omnistat/blob/main/requirements.txt))

## Build-Time Dependencies

- If building the ROCprofiler extension, will require some additional pip packages (cmake-build-extension and nanobind) to install the extension
  - See documentation regarding this [here](https://rocm.github.io/omnistat/installation/extensions.html)

## Run-Time Dependencies

The same as mentioned above in the Dependencies section.

## Deployment

One of the main reasons we would like to be integrated in with TheRock would be to utilize the packaging infrastructure. Ideally, we want to be able to deploy Omnistat via:

- pip packages
  - `pip install omnistat`
  - Automatically handle dependencies via pip
- .deb and .rpm packages
  - Generated from TheRock artifacts
- TheRock artifacts/tarballs
  - As part of the overall TheRock build process

## Next Steps

Once approved, we would like to include `omnistat` as a sub-module in TheRock repository adn begin work on adding it to the packaging infrastructure. We'll need allignment on naming and organization of this new submodule and component as well.
