# Build Artifacts

A primary output of the build system is a set of archive slices of key components. In typical CI flows, these are archived into tarballs and made available to successor jobs as artifacts. Successor jobs can use them for a variety of things:

- Bootstrapping multi-stage builds with pre-built components.
- Driving successor platform/channel specific packaging flows (i.e. generate debs, rpms, wheels, etc).
- Direct inputs to standalone build jobs.
- Distribution of flat archives to users who benefit from this.

Generally, each artifact is an extract of the top level build tree, containing a subset of leaf project stage/ directories. In this way, simply extracting the artifacts over a build directory is sufficient in successor jobs to make that slice of projects be pre-built". In this way, a monolithic build can be broken down and parallelized at sub-project granularity if desired.

## Artifact Layout

After each sub-project build stage, corresponding artifact sub-directories will be populated in the `build/artifacts` directory. As a visual-aid, consider the directory listing of the `base`, `sysdeps`, and `rand` artifacts:

```
$ ls -1d artifacts/{base_*,sysdeps_*,rand_*}
artifacts/base_dbg_generic
artifacts/base_dev_generic
artifacts/base_doc_generic
artifacts/base_lib_generic
artifacts/base_run_generic
artifacts/base_test_generic
artifacts/rand_dbg_gfx1100
artifacts/rand_dev_gfx1100
artifacts/rand_doc_gfx1100
artifacts/rand_lib_gfx1100
artifacts/rand_run_gfx1100
artifacts/sysdeps_dev_generic
artifacts/sysdeps_doc_generic
artifacts/sysdeps_lib_generic
artifacts/sysdeps_run_generic
```

Here we see directories following the artifact naming convention of `{name}_{component}_{target}`, where each field is defined as:

- `name`: Project-wide unique name of the artifact.
- `component`: The component sub-division of the artifact, defining the role of files contained (see below).
- `target`: Either "generic", indicating that it is not confined to any specific GPU target or a target-family as defined by the `therock_amdgpu_targets.cmake`. Note that each build tree is only relevant to a single *host* target, so the host platform is not encoded in the artifact name.

If everything is set up correctly in the build, all relevant files in a sub-project's `stage/` directory will be included in one of its constituent artifact components, differentiated by the role it plays. Note that there can be many sub-projects that go into making a single artifact (i.e. sub-projects that are logically grouped together are grouped into the same artifact).

As a convenience, all artifacts defined in the project are flattened by default into the `build/dist/rocm` directory by the `therock-dist-rocm` target. Since this includes everything, this directory should be considered a "complete" ROCM SDK.

### Artifact Contents

Consider an (abbreviated) tree of the `sysdeps_lib_generic` artifact at the time of writing:

```
$ tree artifacts/sysdeps_lib_generic
artifacts/sysdeps_lib_generic
├── artifact_manifest.txt
└── third-party
    └── sysdeps
        └── linux
            ├── bzip2
            │   └── build
            │       └── stage
            │           └── lib
            │               └── rocm_sysdeps
            │                   └── lib
            │                       ├── libbz2.so -> librocm_sysdeps_bz2.so
            │                       └── librocm_sysdeps_bz2.so
            ├── elfutils
            │   └── build
            │       └── stage
            │           └── lib
            │               └── rocm_sysdeps
            │                   └── lib
            │                       ├── libasm.so -> librocm_sysdeps_asm.so.1
            │                       ├── libdw.so -> librocm_sysdeps_dw.so.1
            │                       ├── libelf.so -> librocm_sysdeps_elf.so.1
            │                       ├── librocm_sysdeps_asm.so.1
            │                       ├── librocm_sysdeps_dw.so.1
            │                       └── librocm_sysdeps_elf.so.1
...
            └── zstd
                └── build
                    └── stage
                        └── lib
                            └── rocm_sysdeps
                                └── lib
                                    ├── librocm_sysdeps_zstd.so.1
                                    └── libzstd.so -> librocm_sysdeps_zstd.so.1
```

And the `artifact_manifest.txt`:

```
third-party/sysdeps/linux/bzip2/build/stage
third-party/sysdeps/linux/elfutils/build/stage
third-party/sysdeps/linux/libdrm/build/stage
third-party/sysdeps/linux/numactl/build/stage
third-party/sysdeps/linux/sqlite3/build/stage
third-party/sysdeps/linux/zlib/build/stage
third-party/sysdeps/linux/zstd/build/stage
```

If you compare this to the overall `build/` tree, you will find that the artifact directory has an identical layout, consisting of a selected slice of files underneath every `stage/` directory in the sub-projects that make up the `sysdeps` artifact.

The `artifact_manifest.txt` file contains the relative paths of each stage directory. If each of these directories were flattened, they will produce a single unified install-tree of all files needed to depend on the sub-project's shared libraries (note: this flattening can be done with the `build_tools/fileset_tool.py artifact-flatten` command).

Preserving the build-tree and project structure in this way allows us to use artifacts to bootstrap a build if wishing to use replace some parts of the source build with sub-projects that were built separately (i.e. in a multi-stage build in CI or as part of a development workflow by downloading dep artifacts from a CI server). In such a situation, all bootstrap artifacts simply need to be copied into the `build/` tree and a special marker file added for each stage directory which instructs the build system to not actually configure/build/install but just use the `stage/` directories as-is.

Artifact directories are populated as part of `all` but can be built manually via targets like `therock-artifact-{name}`. All artifacts can be built with `therock-artifacts`.

### Artifact Archives

The build system also generates a `therock-archive-{name}` for each artifact. This will create a `.tar.xz` file and sha256sum of the artifact directory. All archives can be built with `therock-archives`. These archives are built and streamed to the CI cloud storage server for subsequent phases and packaging workflows.

Building the artifacts is done via the `build_tools/fileset_too.py artifact-archive` command. This command always ensures that the `artifact_manifest.txt` is written to the tar file first, as this is a precondition that the `artifact-flatten` command requires in order to process them.

Archives are not built by default as part of `all` and must be explicitly requested. CI systems typically include a `therock-archives` target in their build to ensure this.

## Building Artifacts

Artifacts are constructed by adding a `therock_provide_artifact()` command to a CMake file. Working forward on our sysdeps example, here is the directive to create its artifact:

```
therock_provide_artifact(sysdeps
  TARGET_NEUTRAL
  DESCRIPTOR artifact.toml
  COMPONENTS
    dev
    doc
    lib
    run
  SUBPROJECT_DEPS
    therock-bzip2
    therock-elfutils
    therock-libdrm
    therock-numactl
    therock-sqlite3
    therock-zlib
    therock-zstd
)
```

This says several important things:

- The artifact is target neutral (i.e. it will end with `_generic` vs a GPU target family).
- It consists of components `dev`, `doc`, `lib`, `run`.
- It is defined by an `artifact.toml` file in the current directory.
- It is assembled from the given subproject's `stage/` directories.

### Artifact Descriptors

The artifact descriptor uses a pattern based language to define what files are included in each component. Since by default, each named component has a default set of patterns, often, no further configuration is needed beyond declaring the build-directory relative locations from which to draw files.

Abbreviated example:

```
# bzip2
[components.dev."third-party/sysdeps/linux/bzip2/build/stage"]
[components.lib."third-party/sysdeps/linux/bzip2/build/stage"]

# elfutils
[components.dev."third-party/sysdeps/linux/elfutils/build/stage"]
[components.lib."third-party/sysdeps/linux/elfutils/build/stage"]

# libdrm
[components.dev."third-party/sysdeps/linux/libdrm/build/stage"]
[components.lib."third-party/sysdeps/linux/libdrm/build/stage"]
include = [
  "**/share/**",
]
```

Each component-dir map supports the following attributes (see `fileset_tool.py artifact` documentation, which is the code that physically transforms the descriptor into artifact directories):

- `default_patterns`: Boolean (default true) whether to use default include/exclude patterns for the given component name.
- `include`: String or list of string path patterns of files that should be included unless if they also match an `exclude` pattern. If `default_patterns` is true, these will be added to the default patterns.
- `exclude`: String or list of string path patterns of files that should be excluded. If `default_patterns` is true, these will be added to the default patterns.
- `force_include`: String or list of string path patterns of files that will always be included, regardless of whether they match an exclude pattern.
- `optional`: Boolean (default false) that if true will not cause an error if the listed stage directory does not exist. Use for optional sub-projects.

All path patterns follow a subset of the [ant path pattern language](https://ant.apache.org/manual/dirtasks.html), which has been implemented by various systems over the years. In brief, `*` matches any number of characters within a path component, and `**` matches any number of path components (including zero).

## Component Types

While artifacts can be defined with any component type mnemonic, the following are standardized across the build and have default patterns that match the majority of situations:

- `dbg`: Platform-specific debug-symbol files. These are typically produced in a platform specific way by the build system and bundled into one component.
- `dev`: Files needed in order to depend on the artifact's contents at build time. This typically includes static libraries, CMake package config files, pkgconfig files, modulefiles, and any tools needed at build time. Notably it does not include shared libraries but does include import libraries (Windows). It is expected that the `dev` component is combined with the `lib` component to produce a fully functional development tree.
- `lib`: Files needed in order to depend on the artifact's contents as a library at runtime. This typically includes shared libraries, DLLs, dylibs, etc. It also includes any file level dependencies that the shared-libraries require in order to function (i.e. for HIP, this can include headers, compiler resources, etc).
- `run`: Files needed in order to use the artifact's contents as a tool. This includes CLI tools (not required at build time), etc.
- `test`: Additional files on top of all above artifacts needed in order to run tests, build test projects, etc. This typically includes test binaries, data file dependencies, and standalone test project trees.

## Current Artifact Inventory

For a fully up to date list, grep for `therock_provide_artifact`. This list is maintained on a best-effort basis.

Sub-projects are being continuously added to TheRock. This section aims to provide a bit more commentary as to how they are organized.

### Common Artifacts

These artifacts are built if any project features requiring them are enabled:

- `host-blas`: An appropriate host BLAS/LAPACK library.
- `host-suite-sparse`: SuiteSparse library.
- `sysdeps`: Privately built shared libraries that are built internally vs relying on system deps. Includes things like drm, compression libs, etc. All of these use project-local SONAMEs and symbol versioning that isolates them from system provided libraries.

### Compiler Artifacts

- `amd-llvm`: The AMD LLVM compiler, configured to target the current host and all AMDGPU targets.
- `hipify`: Hipify tools, built on top of `amd-llvm`.

### Core Artifacts

- `base`: Base ROCM tools and structural components. ROCM sub-projects that do not depend on anything outside of this set are included here so that everything can depend on them.
- `core-runtime`: Low level runtime components used for interfacing with kernel drivers.
- `core-hip`: HIP runtime, compiler interface, and build tools.

### Profiler Artifacts

- `rocprofiler-sdk`: The rocprofv3 tools and libraries (excluding `rocprofiler-register`, which is in `base`).

### Kernel Libraries

- `blas`: All basic linear algebra libraries (BLAS, SOLVER, SPARSE).
- `fft`: Fast fourier transform libraries.
- `prim`: C++ template based primitives libraries (rocPRIM, hipCUB, rocThrust, etc).
- `rand`: Random number generator libraries.
- `rccl`: Collective communication libraries.
- `MIOpen`: MIOpen kernel-select/fusion library.

> [!NOTE]
> After adding a new artifact via `therock_provide_artifact()`, you may need to update `install_rocm_from_artifacts.py` to allow CI workflows and users to selectively install it. <br>
> See the [Adding Support for New Components](./installing_artifacts.md#adding-support-for-new-components) guide for step-by-step instructions.
