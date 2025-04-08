# roctracer-compat

Compatibility shim that provides source-level redirection for the old
`libroctx64.so` tracing library, causing any uses to be redirected to the
newer roctx library that is part of rocprofv3 at build time.

To use this:

```
find_package(roctracer-compat REQUIRED)
target_link_libraries(mylib PRIVATE roctracer-compat::roctx64)
```

If you can make source level changes to new headers and libraries, you can
forgoe the shim and instead use:

```
find_package(rocprofiler-sdk-roctx REQUIRED)
target_link_libraries mylib PRIVATE rocprofiler-sdk-roctx::rocprofiler-sdk-roctx-shared-library)
```

The new approach is available starting in ROCm 6.4. roctracer will be removed in
a future ROCm version, at which time, all users must either be using the
compatibility shim or the new library directly.

At runtime, both approaches are identical.
