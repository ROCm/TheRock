// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// No-op definitions of every entry point into the GPU half of the LLVM profile
// runtime (InstrProfilingPlatformROCm.cpp in libclang_rt.profile_rocm.a).
// therock_subproject.cmake links this into each coverage-instrumented
// executable and shared library so that the real ones are never linked; see the
// coverage comments there.

#pragma GCC visibility push(hidden)

void __llvm_profile_offload_register_shadow_variable(void *Ptr) { (void)Ptr; }

void __llvm_profile_offload_register_section_shadow_variable(void *Ptr) {
  (void)Ptr;
}

void __llvm_profile_offload_register_dynamic_module(int ModuleLoadRc,
                                                    void **Ptr,
                                                    const void *Image) {
  (void)ModuleLoadRc;
  (void)Ptr;
  (void)Image;
}

void __llvm_profile_offload_unregister_dynamic_module(void *Ptr) { (void)Ptr; }

int __llvm_profile_hip_collect_device_data(void) { return 0; }
