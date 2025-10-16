# ManyLinux Builds

ROCm builds of TheRock are designed with accomodations to support the ability to distribute binaries to a wide variety of glibc-based Linux systems. In the Python ecosystem, this is often termed "manylinux", and we adopt the same techniques. However, there are additional constraints when considering building an SDK which are different compared to choices that were made for manylinux Python wheel packaging. This document describes the design, the constraints this puts on software that we include in the build and information for those needing to extend the system.

It should be noted that manylinux builds of ROCm are just one way to use the sources: it is perfectly viable to build the project using native system dependencies in a way that is tailored to a specific distribution. It is just that as a vendor of binary distributions in various forms, AMD has decided that it is worth it to us to formalize how ROCm is built, packaged, and consumed in a distro-neutral fashion. Such binary compatible artifacts are the primary output of our build system via:

- Release tarballs
- Python wheels
- DEB and RPM vendor packages distributed through our own repositories

Doing primary builds of the software to maximize compatibility helps us manage costs across the fragmented Linux ecosystem.

See the [Environment Setup Guide](../environment_setup_guide.md) for information about distribution-specific builds.

## What is a ROCm ManyLinux build?

Three things come together to make a ROCm manylinux build:

1. Built using [our manylinux docker image](../../dockerfiles/build_manylinux_x86_64.Dockerfile), which is a fork from the [upstream images](https://github.com/pypa/manylinux) containing some additional packages and pins needed to satisfy our constraints.
1. Built with the `-DTHEROCK_BUNDLE_SYSDEPS=ON` flag (which is the default), which configures the build system to produce vendored versions of any dependencies that would otherwise be resolved at runtime on the target system.
1. Vendored versions of all needed sysdeps in [third-party/sysdeps](../../third-party/sysdeps). Bundled sysdeps are always installed into the `lib/rocm_sysdeps` prefix and RPATHs project wide are configured to use this if they have any such deps.

### ManyLinux Docker Image

As with upstream, the manylinux docker image is based on a build of AlmaLinux, targeting the specific glibc version that OS shipped with (presently 2.28). Usage of Alma (or a RHEL derivitive) is important because of the availability of GCC dev-toolset backports for the life of the distribution. This lets us always have access to modern compilers and standard libraries, built in a way that they can run on any similarly built glibc/libstdc++ version as the base OS. Even though glibc 2.28 was released in 2018, we get to use new compilers and new standard library features that would otherwise require upgrade of the entire target OS.

#### GCC Toolset

Upstream manylinux images eagerly pull in new gcc-toolset as they become available. For the ROCm version of the image, we manually pin back to older versions as necessary. In general, since ROCm is built on a custom clang/llvm build for AMDGPU, it can never use the absolute newest gcc-toolset because there is always a window of time where the latest GCC version is incompatible with the latest released LLVM. While not a problem for most because by the time new versions of operating systems roll out, this incompatibility windows will have been resolved, it is a problem for ROCm because we are often shipping an LLVM based on the latest release and have no automatic buffer against this incompatibility window. In practice, we try to track the gcc-toolset that PyTorch uses, but sometimes we have to upgrade out of sync due to bugs.

#### Constraints

- In order to avoid introducing unexpected system dependencies, the docker image used to build ROCm never has `*-devel` packages installed beyond those that are supported as part of the gcc-toolset. This constraint turns potentially silent cross-version dependencies into hard errors at build time.
- It is acceptable to install extra development tool packages so long as they do not activate additional `*-devel` packages for system libraries.

### `-DTHEROCK_BUNDLE_SYSDEPS=ON` Mode

In this mode, the sysdeps projects are enabled and added as dependencies to anything that uses them such that `find_package` or `pkgconfig` will pull them in. When disabled, these will be resolved against system dependencies as normal.

### Vendored Libraries

When it makes sense, we prefer to pull dependencies in as static libraries. This makes sense for build-only deps, header-only deps, etc, but it is not appropriate for things that may be used from multiple DSOs. In these cases, we have to configure each library so that it cannot conflict with their system library counter-parts, even if both are loaded into the same global namespace. (Note that on Windows, there is no global namespace, so we tend to build most of these statically vs as DLLs)

All bundled system deps on Linux will be installed into the overall tree at `lib/rocm_sysdeps`, allowing most normal libraries to simply add an additional origin-relative RPATH (unconditionally), picking them up if available.

All such system libraries are altered in the following ways:

- Any distribution specific `lib*/` dirs are changed to just be `lib/`.
- Any packaging files are setup to be relocatable.
- Shared libraries are built with symbol versioning, using the
  `AMDROCM_SYSDEPS_1.0` version.
- Shared library SONAMEs are rewritten to prepend `rocm_sysdeps_` so that they
  can co-exist with system libraries installed with their original SONAME.

Patches are maintained and applied automatically to meet these requirements on a project-by-project basis.

## Compatibility

By not depending on anything beyond the support libraries (glibc) that are properly versioned in the base manylinux image, the resulting ROCm distribution should work without any further alteration on any glibc-based Linux distribution with at least the built glibc version (currently 2.28). While there can be subtle bugs from time to time, this same scheme is used by large swaths of software and benefits from many parties caring about ensuring the compatibility.

Compatibility issues can come from a couple of places:

1. Due to a build or configuration error, ROCm accidentally depends on a system library outside of the compatible set. This can easily be checked by either: a) inspecting each DSO with `ldd` or b) Loading each DSO on a minimal docker image of the distribution-under-test (without any of the used libraries) and verifying that it loads without error.
1. glibc version specific bugs. There is no generic workaround for these and it is best to keep a catalog of issues as they arise so that automated tests can be added. There are limited places in the codebase which use such new parts of glibc that they experience a lot of variability. It is best when doing specialty system programming that is known to have glibc version dependence to proactively note this and ensure an appropriate test plan is in place for that specific case (e.g. it is known that `dlmopen` had many limitations in 2.28 which were lifted in later versions, therefore if introducing code that uses `dlmopen`, this should be flagged for appropriate test case development).
