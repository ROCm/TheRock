# Native Application Packaging with ROCm

This guide summarizes deployment options for native applications that use ROCm.
ROCm components do not yet expose a uniform runtime-staging mechanism. Check the
license, compatibility, and deployment documentation for the ROCm release and
components being packaged.

## Build discovery is not runtime discovery

A build can successfully find ROCm headers, CMake packages, import libraries,
and shared libraries without producing an application that can find its runtime
dependencies after it is installed elsewhere.

Treat these as separate contracts:

1. **Build contract:** Which SDK version, headers, compiler, CMake packages, and
   link libraries does the project consume?
1. **Runtime contract:** Which shared libraries, device data, compiler services,
   and driver interfaces does the built application need?
1. **Deployment contract:** Who installs those files, where do they live, how
   does the loader select them, and who services security and compatibility
   updates?

## Deployment models

The models below can be combined. For example, a Python framework wheel can own
process initialization while depending on separate package-managed ROCm runtime
wheels.

| Model                            | Typical shape                                                                    | Advantages                                                         | Tradeoffs                                                                                                                  |
| -------------------------------- | -------------------------------------------------------------------------------- | ------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------- |
| Centrally installed runtime      | Application plus a separately installed ROCm release                             | Runtime can be shared and serviced centrally                       | Requires a version-selection and loader-activation contract; installation state can vary between machines                  |
| Application-local shared runtime | EXE plus DLLs on Windows; `bin/` plus private `lib/` on Linux                    | Predictable and portable application layout                        | Application owns closure selection, size, licensing, and updates                                                           |
| Launcher or test harness         | Parent configures the loader and starts children                                 | Keeps the source SDK immutable; useful for tests and managed tools | Inner executable is not independently runnable under the same contract                                                     |
| Explicit or delay-loaded runtime | Bootstrap, delay-load hook, plugin interface, or loader API selects a known path | Supports deliberate version selection and optional components      | Must run before ordinary imports and account for generated/runtime-loaded dependencies                                     |
| Static linking                   | Supported implementation archives linked into the application                    | Can reduce shared-library discovery requirements                   | Component-specific support; larger binaries and rebuild-based servicing; may still have dynamic driver/plugin dependencies |
| Framework or language packages   | Wheels or another package graph install and activate native libraries            | Integrates version solving with the framework's installation flow  | Startup becomes dependent on framework-specific initialization and package layout                                          |
| Container or application image   | Application and user-space runtime are assembled into one image                  | Reproducible Linux user-space stack                                | Large deployment unit; host kernel, devices, and compatible driver remain external                                         |

### Choosing a starting point

These are useful starting points rather than requirements:

- A ROCm-owned tool or test installed as part of the same distribution can
  often live in the distribution's runtime `bin` directory.
- A test that compiles temporary executables can use a harness-owned temporary
  directory and scope loader configuration to its child processes.
- A directly runnable application can consider an application-local shared
  runtime when the relevant release supports redistribution.
- A framework can express ROCm runtime components in its package graph and run
  initialization before loading native extensions.
- A Linux service with a large, tightly coupled native stack can treat a
  container as its user-space deployment unit.

A downstream application should not install itself into ROCm's shared `bin`
directory merely to gain DLL precedence. It should own its deployment layout or
use a documented central-runtime activation mechanism.

## Windows shared-library options

### Application-local DLLs

The simplest directly runnable Windows layout places the intended DLLs and
their dependencies in the same directory as the executable:

```text
my-application/
  my-application.exe
  amdhip64_7.dll
  amd_comgr.dll
  rocm_kpack.dll
  other supported runtime files
```

The executable directory is searched before `System32`. This can provide
deterministic selection even while a legacy same-named runtime is installed
globally.

The list above is illustrative. A PE import scan does not identify modules
loaded with `LoadLibrary`, device/kernel packages, configuration, licenses, or
optional feature dependencies. Use component packaging metadata when available.

> [!IMPORTANT]
> The current
> [HIP SDK deployment guidelines](https://rocm.docs.amd.com/projects/install-on-windows/en/develop/conceptual/deployment-guidelines.html)
> state that applications do not redistribute the HIP runtime. Follow the
> deployment and licensing guidance for the ROCm release being packaged.

Several executables can share one application-owned `bin` directory instead of
carrying a private copy beside every executable.

### Launcher-managed central runtime

A launcher with no ordinary HIP imports can discover a ROCm installation,
validate its runtime directory, configure DLL loading, and create the HIP
process. A test runner that owns child process creation can use the same pattern.

`SetDllDirectoryW` is one available mechanism for an unpackaged child process.
More restrictive designs can use `SetDefaultDllDirectories`, `AddDllDirectory`,
and `LoadLibraryExW` flags where their process and dependency-loading model
allows it.

Registry or configuration discovery alone is not sufficient for an executable
with ordinary HIP imports: Windows resolves those imports before application
`main()` can inspect the discovered path.

### Explicit loading and application modules

Applications that need stronger version selection can avoid importing HIP from
their initial executable. Some possible architectures are:

- A small host executable loads the selected runtime by absolute path, then
  loads an application DLL containing the compiled HIP code.
- A linker delay-load hook resolves the intended runtime path when the first HIP
  import is used.
- A stable loader or dispatch API loads the runtime and exposes versioned entry
  points to the application.

Compiled HIP programs can contain generated fat-binary registration calls that
execute during static initialization. Placing `LoadLibraryExW()` at the top of
an otherwise ordinarily linked `main()` does not solve startup loading.

Derive known paths from an application-owned layout, a documented versioned
installation, or trusted configuration. Do not embed the developer's SDK path
found by CMake.

### Static linking

Do not assume that a Windows `.lib` is a static implementation library. It can
instead be an import library whose code remains in a DLL. In current TheRock
Windows distributions, `amdhip64.lib` is an import library for
`amdhip64_7.dll`, and the installed `hip::amdhip64` CMake target is shared.

The current
[HIP SDK deployment guidelines](https://rocm.docs.amd.com/projects/install-on-windows/en/develop/conceptual/deployment-guidelines.html)
state that static linking to distributed HIP SDK components is unsupported.
A static build mode in a ROCm source project does not imply that its distributed
SDK supports static linking.

## Linux shared-library options

### Distribution-managed runtime

Debian and RPM packages can declare dependencies on compatible ROCm runtime
packages. The package manager then owns installation, upgrades, and removal,
and the system dynamic loader resolves libraries from its configured paths and
cache.

This model couples the application to the distribution's package graph and
selected ROCm versions. Installation and updates occur separately from
application execution.

### Relocatable application layout

Linux ELF binaries can encode a runtime search path relative to themselves
using `$ORIGIN` in `DT_RUNPATH` or `DT_RPATH`. A common private layout is:

```text
my-application/
  bin/
    my-application
  lib/
    libamdhip64.so.7
    other supported runtime libraries
```

For example, an installed executable can use:

```cmake
set_target_properties(
  my_application
  PROPERTIES INSTALL_RPATH "$ORIGIN/../lib"
)
```

`DT_RUNPATH` applies to an object's direct dependencies rather than every
descendant in the graph. Bundled libraries may therefore need their own
RUNPATH.

See [`ld.so(8)`](https://man7.org/linux/man-pages/man8/ld.so.8.html) for the
loader order and token expansion rules.

### Activation environment

Wrappers and test runners commonly prepend a selected runtime directory to
`LD_LIBRARY_PATH`. This is useful for testing an extracted SDK without
registering it system-wide and is similar in role to a Windows parent using
`SetDllDirectoryW`.

An executable that relies only on this wrapper is not independently relocatable.
For an application bundle intended for direct execution, prefer a deliberate
installed RUNPATH or another package-owned activation mechanism.

## ROCm-specific considerations

A native ROCm application's runtime closure can include more than the library
named on its link line:

- The HIP runtime and libraries used directly by the application.
- Libraries loaded dynamically for optional features or plugins.
- COMGR, HIP RTC, compiler/linker components, or device libraries used for
  runtime compilation.
- Architecture-specific kernel packages and databases.
- Configuration, licenses, and notices.
- Host runtime dependencies such as the supported Microsoft Visual C++ runtime.
- A compatible host driver and operating-system interface, which generally
  remain outside an application bundle or container.

TheRock does not yet provide uniform machine-readable runtime and redistribution
metadata for this set. Projects should document the release and packaging model
they test.

## Responsibility boundaries

Responsibilities by layer include:

- A ROCm component identifies its runtime files, compatible versions,
  dynamically loaded dependencies, and redistribution constraints.
- A downstream application chooses a deployment model, includes or declares its
  dependencies, configures loading before native code needs it, and updates any
  private copy it distributes.
- A package manager, framework initializer, launcher, or container entry point
  implements activation when the application is not independently runnable.
  That requirement is part of the installed product's interface, not merely a
  CI detail.
- A test harness constructs an isolated deployment, keeps the SDK and
  installation prefix immutable, and verifies the user-facing loading contract.
  A harness-specific loading contract should be identified as such.

## Testing a deployment

Useful checks include:

- Build and install into a new temporary prefix rather than the SDK tree.
- Treat the SDK or extracted ROCm installation as read-only test input.
- Run the installed application from a different current directory.
- Test without development-only environment variables and build directories.
- Record the actual loaded library paths, not only process exit status.
- Test with another ROCm version already installed. On Windows, include a
  same-named compatibility DLL in `System32` when that configuration is in
  scope.
- Exercise runtime compilation, plugins, and data-dependent paths in addition
  to process startup.
- Test both the minimal declared package composition and a fully merged SDK;
  unrelated files in a full installation can hide missing dependencies.
- Confirm that test setup did not add, replace, or edit files in the installation
  under test.

Tests that are intentionally harness-managed should say so. Direct invocation
of the inner executable may legitimately fail outside the harness, but that
must not be confused with validation of a redistributable application.

## Ecosystem design examples

This source audit describes the projects as of August 2026. Follow the linked
sources for current behavior.

### PyTorch built with TheRock packages

TheRock's default PyTorch wheel does not copy ROCm into `torch`. It declares a
version-matched dependency on `rocm[libraries]` and generates `_rocm_init.py`,
which calls `rocm_sdk.initialize_process()` before PyTorch loads its native
extensions. The helper preloads a project-selected library set and checks the
ROCm package version.

This shares ROCm components between packages but requires the initializer,
package metadata, and coordinated versions.

TheRock also has an experimental Windows
[`windows_patch_fat_wheel.py`](../../external-builds/pytorch/windows_patch_fat_wheel.py)
flow that copies a filtered ROCm tree into `torch/lib/rocm` and patches PyTorch's
DLL directory setup. This produces a larger, self-contained wheel.

See the [PyTorch build tooling](../../external-builds/pytorch/build_prod_wheels.py),
[PyTorch packaging notes](../../external-builds/pytorch/README.md#bundling-pytorch-and-rocm-together-into-a-fat-wheel),
and upstream
[`torch/__init__.py`](https://github.com/pytorch/pytorch/blob/main/torch/__init__.py).

### JAX ROCm plugin wheels

TheRock's current JAX flow is Linux-only and packages ROCm support as separate
PJRT and plugin wheels whose names include the ROCm major version. Tests install
matching ROCm libraries and device packages separately. The plugin wheels carry
some private build/runtime assets, including their linker and bitcode, while
the shared ROCm runtime remains a package dependency of the environment. The
test runner deliberately clears build-time path variables to confirm that the
installed wheels use those packaged assets.

The package names expose the accelerator boundary and ROCm major version. The
plugin, PJRT, JAX, and ROCm package versions must be coordinated. This flow does
not currently cover Windows.

See [Build JAX with ROCm support](../../external-builds/jax/README.md), the
[JAX test runner](../../external-builds/jax/run_jax_tests.py), and the
[ROCm/JAX source](https://github.com/ROCm/jax).

### llama.cpp HIP backend

llama.cpp's HIP backend finds HIP, hipBLAS, and rocBLAS CMake packages and links
its `ggml-hip` target to their shared targets. It can build compute backends as
dynamically loaded modules with `GGML_BACKEND_DL`, while its HIP CMake logic
explicitly rejects `GGML_STATIC`.

This separates the optional accelerator backend from the core application. The
build logic assumes that a compatible ROCm SDK/runtime is discoverable and does
not define an application-local ROCm package.

See llama.cpp's
[`ggml` build options](https://github.com/ggml-org/llama.cpp/blob/master/ggml/CMakeLists.txt)
and
[`ggml-hip` CMake configuration](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-hip/CMakeLists.txt).

### SGLang ROCm containers

SGLang's ROCm deployment flow uses ROCm/PyTorch base images and layers SGLang,
accelerator-specific kernels, communication libraries, and other native
dependencies into a container. The base image and PyTorch own much of the core
ROCm runtime selection; the SGLang image then pins and builds a tested stack for
a target GPU architecture and ROCm release.

The container is the user-space deployment unit. The image is large and
platform-specific, and it still requires compatible host devices, kernel
interfaces, and drivers.

See SGLang's
[`rocm.Dockerfile`](https://github.com/sgl-project/sglang/blob/main/docker/rocm.Dockerfile).

## Open design questions

Open questions include:

- Which files form the supported redistributable closure for each component?
- Should ROCm provide a CMake runtime-staging helper for application-local
  packages?
- How should native Windows applications discover and activate a versioned
  central runtime before ordinary imports are resolved?
- Which static runtime configurations, if any, will be distributed and
  supported?
- How should applications express compatibility with runtime, compiler, device
  package, and driver versions?
- Which deployment checks should every ROCm project and downstream package run
  in CI?
