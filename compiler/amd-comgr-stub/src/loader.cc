// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#ifndef _GNU_SOURCE
#define _GNU_SOURCE // For dlmopen
#endif

#include <dlfcn.h>
#include <errno.h>

#include <cstdlib>
#include <stdio.h>
#include <string>

namespace {

struct LibraryInitialization {
  Lmid_t dlopen_namespace = 0;
  void *dlopen_handle = nullptr;
  std::string initial_lib_name;

  LibraryInitialization(const char *lib_name) : initial_lib_name(lib_name) {
    dlopen_handle = dlmopen(LM_ID_NEWLM, lib_name, RTLD_LOCAL | RTLD_NOW);
    if (dlopen_handle) {
      if (dlinfo(dlopen_handle, RTLD_DI_LMID, &dlopen_namespace)) {
        // This form of call to dlinfo should not fail by construction, but
        // abort if it does for safety.
        perror("dlinfo query for LMID failed");
        std::abort();
      }
    }
  }
  ~LibraryInitialization() {
    if (dlopen_handle) {
      dlclose(dlopen_handle);
    }
  }
};

} // namespace

// dlopen callback that the static stub uses.
extern "C" void *amd_comgr_stub_dlopen(const char *lib_name) {
  static LibraryInitialization init(lib_name);
  if (!init.dlopen_handle) {
    errno = ENOENT;
    return nullptr;
  }

  // Make sure the same lib is being requested as the primordial request.
  if (init.initial_lib_name != lib_name) {
    errno = EINVAL;
    return nullptr;
  }

  // The primordial open was successful: return a fresh handle as requested.
  return dlmopen(init.dlopen_namespace, lib_name, RTLD_LOCAL | RTLD_NOW);
}
