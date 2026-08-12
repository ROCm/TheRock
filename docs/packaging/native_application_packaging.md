# Native Application Packaging with ROCm

Finding ROCm headers and libraries at build time does not configure shared
library loading after an application is installed. Native application packages
must either depend on an installed ROCm runtime or provide the runtime libraries
they need, then ensure that the operating-system loader can find them.

ROCm does not yet provide one runtime-staging interface for every component.
Follow the deployment and redistribution documentation for the ROCm release and
components being packaged.

## Linux shared-library loading

Linux applications commonly use one of these approaches:

- **System-installed libraries:** Debian or RPM packages declare dependencies on
  ROCm runtime packages. The system dynamic loader finds the installed
  libraries through its configured paths and cache.
- **A relative runtime search path:** An application ships supported shared
  libraries in its own package and records a path relative to the executable or
  library using `$ORIGIN` in `DT_RUNPATH` or `DT_RPATH`.
- **An activation environment:** A launcher, shell setup script, or test harness
  adds ROCm library directories to `LD_LIBRARY_PATH` before starting the
  application.

For example, an application with this layout can set its installed runtime
search path to `$ORIGIN/../lib`:

```text
my-application/
  bin/
    my-application
  lib/
    libamdhip64.so.7
    other required libraries
```

```cmake
set_target_properties(
  my_application
  PROPERTIES INSTALL_RPATH "$ORIGIN/../lib"
)
```

`$ORIGIN` expands to the directory containing the executable or shared object.
`DT_RUNPATH` applies to an object's direct dependencies, so packaged libraries
may also need their own runtime search paths.

`LD_LIBRARY_PATH` is useful when testing an extracted ROCm installation or when
an application is always started by a launcher. An executable that relies on it
must be started with that environment configured.

See [`ld.so(8)`](https://man7.org/linux/man-pages/man8/ld.so.8.html) for the
loader's search order and the CMake documentation for
[`BUILD_RPATH`](https://cmake.org/cmake/help/latest/prop_tgt/BUILD_RPATH.html)
and
[`INSTALL_RPATH`](https://cmake.org/cmake/help/latest/prop_tgt/INSTALL_RPATH.html).

## Windows DLL loading

For an ordinary unpackaged Windows application, the DLL search order includes
the following locations in this relative order:

1. The directory containing the executable.
1. The Windows system directory, normally `System32`.
1. The process current directory.
1. Directories in `PATH`.

Applications commonly load their DLLs by:

- Placing the required DLLs and their dependencies beside the executable.
- Using a launcher to select an installed runtime and configure its DLL
  directory before starting the application.
- Explicitly loading DLLs from known paths when the application's architecture
  supports doing so before the DLLs are needed.
- Statically linking components that publish and support static libraries.

> [!WARNING]
> Adding a ROCm `bin` directory to `PATH` does not override a same-named DLL in
> `System32`. Windows also loads normally linked DLLs before `main()` runs, so
> changing the search path at the beginning of `main()` is too late for those
> DLLs.

Launchers and test harnesses can use APIs such as `SetDllDirectoryW`,
`AddDllDirectory`, and `LoadLibraryExW` to control loading for their processes.
See [Windows Support](../development/windows_support.md#dll-search-and-runtime-loading)
for an example and
[Dynamic-link library search order](https://learn.microsoft.com/en-us/windows/win32/dlls/dynamic-link-library-search-order)
for the complete Windows rules.

> [!IMPORTANT]
> The current
> [HIP SDK deployment guidelines](https://rocm.docs.amd.com/projects/install-on-windows/en/develop/conceptual/deployment-guidelines.html)
> state that applications do not redistribute the HIP runtime and that static
> linking to distributed HIP SDK components is unsupported. Check the guidance
> for the ROCm release being packaged.

### Windows packaging transition

ROCm's Windows packaging model is being reworked. Current and older GPU driver
installations can place HIP and COMGR runtime DLLs in `System32`. This couples
user-space ROCm libraries to the driver installation and gives those DLLs
precedence over directories in `PATH`.

The draft
[Windows Packaging Requirements RFC](https://github.com/ROCm/TheRock/pull/3973)
proposes moving ROCm user-space runtime libraries into versioned ROCm package
directories. Compatibility packages would temporarily provide the older HIP
runtime versions in `System32`, while new runtime versions would use
user-space deployment and discovery mechanisms.

New applications should not assume that ROCm runtime DLLs will continue to be
provided by the GPU driver or found in `System32`.

## Packaging and testing notes

> [!NOTE]
> The libraries named directly on an application's link line may not be its
> complete runtime dependency set. ROCm libraries can load additional libraries,
> device code, and data files at runtime.

- Package or declare the dependencies required by the features the application
  uses. A list based only on inspecting the executable can be incomplete.
- Keep ROCm installations read-only while running tests. Use a temporary
  application directory or configure the loader from a test harness.
- Test the installed application without build-tree paths or development-only
  environment variables.
- On Windows, test with a same-named runtime DLL already present in `System32`
  while that configuration remains supported.
