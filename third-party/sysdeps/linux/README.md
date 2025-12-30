# Linux bundled system deps

When `-DTHEROCK_BUNDLE_SYSDEPS=ON` on Linux, this build tree will produce a
`sysdeps` artifact that includes project wide system dependencies that need
to be distributed with the SDK in portable builds. This includes private
versions of all libraries that would otherwise need to be installed on the
system to satisfy the set of features enabled.

All bundled system deps on Linux will be installed into the overall tree at
`lib/rocm_sysdeps`, allowing most normal libraries to simply add an additional
origin-relative RPATH (unconditionally), picking them up if available.

All such system libraries are altered in the following ways:

- Any distribution specific `lib*/` dirs are changed to just be `lib/`.
- Any packaging files are setup to be relocatable.
- Shared libraries are built with symbol versioning, using the
  `AMDROCM_SYSDEPS_1.0` version.
- Shared library SONAMEs are rewritten to prepend `rocm_sysdeps_` so that they
  can co-exist with system libraries installed with their original SONAME.

## Artifacts

The `sysdeps` artifact contains "core" libraries that are common on base images and used by multiple ROCm components. Additional, optional artifacts are managed separately for lesser used or large libraries that we do not wish to unconditionally install.
