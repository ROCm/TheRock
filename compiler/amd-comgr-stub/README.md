# amd-comgr-stub

## Overview

Generates stubs for amd_comgr such that it is delay loaded in a private
namespace on compatible Unix-like operating systems.

## Design

This consists of a few parts:

1. amd_comgr static library: This is a drop-in replacement for the real
   amd_comgr library. It exports static symbols with hidden visibility
   corresponding to the public exports of the real library. It is called
   amd_comgr_stub.a on disk.
1. amd_comgr_loader shared library: This exposes a single C-extern
   `amd_comgr_stub_dlopen` which the stub generator calls in place of dlopen.
   Since this is a shared library in the caller namespace, it can hold the
   process-wide static of the child namespace id and keep the real amd_comgr
   library loaded until destruction.
1. The RPATH of $ORIGIN on the amd_comgr_loader ensures that, since it is
   always adjacent to the real amd_comgr.so.\* on disk, dlmopen will use the
   adjacent library with no further path munging. It is merely sufficient
   that the dlmopen call happen within a process-scoped shared library (vs
   in the static stub library, which could be loaded into any library or
   executable on the system and could therefore not robustly know how to
   get back to its adjacent, real amd_comgr.so.\* files).
1. The stub generator queries the soname at build time and asks for it
   as a literal in calls to `amd_comgr_stub_dlopen`. In this way, we are
   always asking for the actual fully versioned library. It is a quirk of
   the binding generator that every call will be the same. We enforce this
   in the loader, just to be safe.

CMake exports for the `amd_comgr` package will be generated under
`lib/cmake/amd_comgr_stub`. In this way, applications can choose to link
against the stub library by using it in their CMAKE_PREFIX_PATH. In the future,
we may include auto-redirection in the real `amd_comgr` package, but this
mechanism as-is is sufficient for all in tree deps on amd_comgr.

The namespace and backing shared library will be lazy loaded on first use.
If callers have logic to handle their own lazy loading, they should disable it
if the `amd_comgr` library is detected to be STATIC.

## Testing

The `test_amd_comgr_stub` test simply gets and prints the version of the
library, which forces loading and initialization.
