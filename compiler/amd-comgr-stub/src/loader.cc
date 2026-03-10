// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#ifndef _GNU_SOURCE
#define _GNU_SOURCE // For dlmopen
#endif

#include <dlfcn.h>
#include <errno.h>
#include <locale.h>

#include <cstdlib>
#include <cstring>
#include <stdio.h>
#include <string>

extern "C" void *amd_comgr_stub_dlopen(const char *lib_name);
extern "C" void amd_comgr_namespace_init();

namespace {

struct LibraryInitialization {
  Lmid_t dlopen_namespace = 0;
  void *self_dlopen_handle = nullptr;
  void *dlopen_handle = nullptr;
  std::string initial_lib_name;
  bool initialized_namespace = false;

  LibraryInitialization(const char *lib_name) : initial_lib_name(lib_name) {
    char *namespace_mode = std::getenv("AMD_COMGR_NAMESPACE");
    bool enable_namespace =
        namespace_mode && std::strcmp(namespace_mode, "1") == 0;
    if (enable_namespace) {
      InitializeNamespace(lib_name);
    }

    // Fallback.
    if (!initialized_namespace) {
      if (enable_namespace) {
        fprintf(stderr, "warning: could not open comgr into isolated "
                        "namespace. Falling back to base.\n");
      }
      // Do not load into namespace.
      dlopen_handle = dlopen(lib_name, RTLD_LOCAL | RTLD_NOW);
    }
  }

  void InitializeNamespace(const char *lib_name) {
    // Warning: Initializing comgr into a namespace is still experimental and
    // has sharp edges depending on glibc version. As such, it is opt-in with
    // an env var and we print more error messages than we should in a final
    // build.
    // TODO: When enabling this by default, trim error messages.
    Dl_info dl_info;
    if (dladdr((void *)amd_comgr_stub_dlopen, &dl_info) == 0) {
      fprintf(stderr, "error: could not determing self library name: %s\n",
              dl_info.dli_fname);
      return;
    }

    // dlmopen self to establish the namespace.
    self_dlopen_handle =
        dlmopen(LM_ID_NEWLM, dl_info.dli_fname, RTLD_NOW | RTLD_NODELETE);
    if (!self_dlopen_handle) {
      perror("could not dlmopen self");
      return;
    }

    // Get the created namespace.
    if (dlinfo(self_dlopen_handle, RTLD_DI_LMID, &dlopen_namespace) != 0) {
      // This form of call to dlinfo should not fail by construction, but
      // abort if it does for safety.
      perror("dlinfo query for LMID failed");
      std::abort();
    }

    // Perform namespace initialization.
    // While not required, glibc has various sharp edges related to operating
    // in a namespace. We therefore have an explicit step where we "prime it"
    // with the expectation that failures happen early and are more easily
    // detectable.
    void *namespace_init =
        dlsym(self_dlopen_handle, "amd_comgr_namespace_init");
    if (!namespace_init) {
      fprintf(stderr, "error: could not dlsym amd_comgr_namespace_init\n");
      return;
    }
    reinterpret_cast<void (*)()>(namespace_init)();

    // dlmopen the actual comgr lib.
    dlopen_handle =
        dlmopen(dlopen_namespace, lib_name, RTLD_NOW | RTLD_NODELETE);
    if (dlopen_handle) {
      initialized_namespace = true;
    } else {
      perror("could not dlmopen comgr");
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
void *amd_comgr_stub_dlopen(const char *lib_name) {
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
  void *handle;
  if (init.initialized_namespace) {
    handle = dlmopen(init.dlopen_namespace, lib_name, RTLD_NOW | RTLD_NODELETE);
  } else {
    handle = dlopen(lib_name, RTLD_NOW | RTLD_NODELETE);
  }
  if (!handle) {
    perror("could not delay-load amd_comgr");
  }
  return handle;
}

void amd_comgr_namespace_init() {
  // TODO: Add various diagnostics against libc within the namespace as needed.
}
