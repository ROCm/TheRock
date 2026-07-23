---
author: Gabriel Pham
created: 2025-12-03
modified: 2026-04-15
status: Draft
discussion: [TBD - Add GitHub issue/PR link]
---

# RFC0012: CUID Project Package

## Overview
The Component Universal IDentification (CUID) standard defines a UUIDv8 type identifier for AMD components.
This identifier is meant to be reproducible and unique across different systems in order to identify components
coming on to or transferring between systems. Additionally, implementation of the standard across all AMD
tools would allow a consistent identifier to be made and displayed by all the different AMD tools and therefore
allow users to identify and track the same device across both systems and AMD tools. The CUID library
implements the CUID standard to allow other applications/tools to easily create and read CUIDs.

## Background
Platform management tools, such as AMD-SMI, debugging-, and profiling tool chains, require a consistent,
uniform and robust method for identifying individual components. This method should not rely on specific
device driver support but be grounded in the architectural properties of the hardware. A consistent identifier
is essential for unambiguously locating components within a platform, rack, or cluster, thereby facilitating
debugging, profiling, failure analysis, and simplifying component maintenance.

Currently, component identification mechanisms show significant variation depending on device type, installed
drivers, and intended purpose, which makes cross-tool chain alignment quite challenging. While some components
are recognized by the operating system through PCIe mechanisms, others are identified through ACPI device objects.
Therefore, the identification mechanism needs to be capable of operating with either method, recognizing components
within a platform and potentially across platforms in a cluster or data center, particularly when components are
physically moved between systems for testing or maintenance.

## Proposal
The CUID project was added to the rocm-systems mono-repo back in February, but is now looking for a wider, more
standardized release. We seek to do this by creating the new ROCm-common package and integrating that package into
the ROCm stack. We've begun working with the packaging team to help us integrate our build process for theROCk, and
we need to determine what are the next steps in this process for fully integrating our package into theROCk.

## Components of the CUID Project
  - CUID Library - the headers and other code which we expect other AMD tools/applications to include in their
  own projects to be able to read or generate CUIDs.
  - CUID CLI - A quick way to view the CUID and other device information of the AMD components on a system.
  - CUID daemon - A service that will discover devices on a system and track when devices are added or removed
  from the system in order to maintain the CUID information associated with each device.

## Requirements

### Library Dependencies and Systems Utilized
  - OpenSSL - Used to provide the cryptographic operations for creating derived CUIDs
  - Threads - Used by the daemon to help provide constant service
  - Udev - Used by the daemon to listen for changes in devices on the system
  - Systemd - Used by the daemon to start service and allow sercice restart on failures

### File Locations
  - CUID information files stored in /tmp (as /tmp/cuid and /tmp/priv_cuid)
  - amdcuid_config and hmac_key.bin stored in /etc/amdcuid
  - The default installation prefix will be /opt/rocm/core to facilitate dependent projects' integration of the
  CUID library code
    - The CLI tool binary (amdcuid_tool) and Daemon binary (amdcuid_daemon) will be located in /opt/rocm/core/bin
    - Both the library code's shared object and static library files (libamd_cuid.so and libamd_cuid.a) will be
    located in /opt/rocm/core/lib
    - The public API header, amd_cuid.h, will be located at /opt/rocm/core/include/amdcuid
    - post install and pre rm scripts will be installed to /opt/rocm/core/share/amdcuid
    - optional testing binary (amdcuid_test) and documentation will also be installed to the /opt/rocm/core/share/amdcuid directory
  - amdcuid_daemon.service file will be created and then symlinked to /lib/systemd/system on DEB-based systems or /usr/lib/systemd/system
  on RPM-based systems to enable systemd capabilities
  - A udev rules file will be installed to /etc/udev/rules.d which will allow the daemon to make use of the udev
  subsytem for detecting changes in the devices on a system

#### Directory structure of all relevant files
```
/opt/rocm/core-[major.minor]
|---/bin
|   |---/amdcuid_daemon
|   |---/amdcuid_tool
|---/lib
|   |---/libamd_cuid.so
|   |---/libamd_cuid.a
|---/include
|   |---/amdcuid/amd_cuid.h
|---/share
|   |---/amdcuid
|       |---/amdcuid_postinst.sh
|       |---/amdcuid_prerm.sh
/tmp
|---/cuid
|---/priv_cuid
/etc
|---/amdcuid
|   |---/amdcuid_daemon.conf
|   |---/hmac_key.bin
|---/udev/rules.d
|   |---/90-amdcuid.rules
/lib/systemd/system (or /usr/lib/systemd/system depending on the system)
|---/amdcuid/amdcuid_daemon.service
```

### Behaviors
  - In a multi-version install environment, CUIDs generated must remain consistent across versions, so all versions
  must refer to the same configuration file and hmac key file (both stored in /etc/amdcuid)
  - Similarly, only a single instance of the daemon should be running at a time to ensure that CUIDs generated are
  consistent for all devices on the system
  - The daemon service is required for users to generate and query devices. If users choose not to install the daemon,
  they will not have access to CUID information. Therefore, the daemon is rereuied to run in some capacity.
    - Users may configure the daemon (by editing amdcuid_daemon.conf) to run continuously or as a simpler boot time run
    to save resources in heavily constrained environments if they so need. When run continuously, the daemon will utilize
    the systemd system to provide restart-on-failure capability to ensure availability

## Package Composition Proposal
A meta package called `amdrocm-cuid-service` will be created and will consist of the following subh packages:
  - `amdrocm-cuid-daemon` which will contain and install the daemon service which is required to generate CUIDs on a system
  which can later be queried by other AMD tools which integrate the library or by the CLI tool.
  - `amdrocm-cuid-lib` which contains the library. This package will be needed by other AMD tools (such as Amd Smi, Amdxio,
  AGFHC, rocprofiler, etc.) for them to be able to query for CUIDs.
  - `amdrocm-cuid-tool` which will contain the CLI tool. This package is meant for end users such as system/data center
  administrators. It may not be required on all systems, so the package can be made optional.

## Current Status and Roadmap

### Current Capability
  - Library has implemented a number of use cases such as with GPU, CPU, and others such that it may be used
  now by other tools to generate and read CUIDs for those component types. Development will continue to add
  other components and improvements.
  - CLI tool has been developed to allow users to quickly view or generate CUIDs on their own, assuming they
  have the correct privilege
  - Daemon service has been developed to generate and update CUIDs for devices on a system in real time