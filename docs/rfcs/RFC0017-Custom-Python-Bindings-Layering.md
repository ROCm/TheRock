---
author: Brian Harrison (bharriso)
created: 2026-06-23
modified: 2026-11-19
status: draft
discussion: https://github.com/ROCm/TheRock/issues/6048
---

# Custom Python Bindings: Layering, Packaging, and Validation

This RFC defines where custom Python bindings for ROCm components live, how they build against the ROCm SDK, and how TheRock packages, validates, and releases them. It applies to any ROCm component with a hand-written Python API, starting with Origami, TensileLite, hipDNN, rocSHMEM, and rocjitsu.

## Summary

- **Where.** Each binding is a standalone Python project in its component's super-repo, at `bindings/python/<package-name>/`, beside the `bindings/fortran/` tree proposed in the RFC0015 update ([TheRock#9433](https://github.com/ROCm/TheRock/pull/9433)). There is no separate bindings repository.
- **What it is.** A canonical Python project that builds on top of its component's installed native library. It is not part of that library's build.
- **How it builds.** `python -m pip install ./bindings/python/<package-name>` builds the binding against an installed ROCm. The binding never rebuilds, bundles, or replaces a ROCm library.
- **How it ships.** TheRock builds the wheel in a separate job after the SDK wheels, from the same source revision, and publishes it to the SDK's Python index. Each wheel depends on the SDK build it was tested with, relaxed to a patch-compatible range once promoted to stable. Users install the published wheel; source builds are for developers and system ROCm installs.
- **How it is validated.** TheRock installs the wheel into a fresh environment as a consumer would and checks that it calls the shipped SDK libraries. Each run records the SDK build, package versions, source revisions, and test inputs.
- **Python versions.** New native extensions use the CPython Stable ABI. Builds per Python minor version are exceptions, agreed before they are added.
- **Rollout.** `bindings/python/` READMEs first, then an Origami pilot behind a feature flag that is off by default, then hipDNN and TensileLite independently.

## Background

### ROCm Python packages

| Kind                      | Examples                                                                 | What it is                                                                                    |
| :------------------------ | :----------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------- |
| ROCm SDK wheels           | `rocm`, `rocm-sdk-core`, `rocm-sdk-libraries`, `rocm-sdk-devel`          | Native ROCm libraries, plus the `rocm_sdk` helper that locates and preloads them.             |
| Generated C API bindings  | `hip-python` ([TheRock#5609](https://github.com/ROCm/TheRock/pull/5609)) | 1:1 wrappers generated from ROCm C headers.                                                   |
| Custom component bindings | `hipdnn-frontend`, `rocm-origami`                                        | Hand-written Python APIs over a component's C++ or higher-level API. The subject of this RFC. |

### Why this matters

Today each component builds and ships its Python package its own way, and nothing ties that package to the native libraries ROCm ships. Two defects show the cost:

- **Two copies of Origami in one process.** The Origami Python package compiled its own copy of the Origami library while ROCm shipped `liborigami` separately. A process that loaded both could call into the wrong copy and corrupt its heap. TheRock's tests did not install that Python package, so release-candidate testing found the crash instead.
- **A Python extension inside an SDK wheel.** Building hipDNN's extension as part of the native build ([TheRock#5429](https://github.com/ROCm/TheRock/pull/5429)) shipped a Python-3.12-only extension with a build-machine RPATH in `rocm-sdk-devel`, which broke `rocm-sdk test` on Python 3.13 ([TheRock#5678](https://github.com/ROCm/TheRock/issues/5678)). [TheRock#6425](https://github.com/ROCm/TheRock/pull/6425) removed that build.

The first is a binding that replaces the library under test; the second is a binding commingled with the native build. This RFC rules out both.

### Current state

- **hipDNN:** `projects/hipdnn/python/` has a nanobind extension project (Stable ABI, Python 3.12+) and a separate wheel project with a custom packer. The wheel declares no SDK dependency. TheRock does not build it.
- **Origami:** `shared/origami/python/` is a scikit-build-core project that compiles its own copy of the library by default. [rocm-libraries#10975](https://github.com/ROCm/rocm-libraries/pull/10975) (open) links it against the installed shared `liborigami` instead and renames the package to `rocm-origami`.
- **TensileLite:** hipBLASLt's kernel generator, a Python package in `projects/hipblaslt/tensilelite` that the native hipBLASLt build also uses. Its `rocisa` extension is built with hipBLASLt and ships in `rocm-sdk-libraries` as an accepted test- and codegen-only exception ([TheRock#5701](https://github.com/ROCm/TheRock/issues/5701)).
- Neither super-repo has a `bindings/` directory yet.

## Design principles

**P1. A binding is a canonical Python project.** A standard build backend builds it from its `pyproject.toml`, and it installs with `pip` from its own directory.

**P2. The binding builds on its component; it is not part of it.** The native library builds, installs, and tests without the binding, and has no Python build options. The binding uses only the component's installed headers, CMake packages, and shared libraries.

**P3. ROCm supplies every native library.** The binding links and loads the ROCm it was built against. It never compiles, embeds, or vendors its own copy of a ROCm library.

**P4. Bindings sit next to their component.** Same super-repo and owners, so an API change and its binding update land in the same pull request.

**P5. Release what was tested.** TheRock publishes the wheel it validated, against the SDK it validated, and records the evidence.

## Non-goals

- Generated C API bindings (`hip-python`) and Fortran bindings ([RFC0015](./RFC0015-Fortran-Bindings-Split.md)).
- Bindings in other languages.
- Mandating one binding framework. The choice is the component's, within the Python version policy below.
- A release cadence separate from the SDK.

## Layout and ownership

```text
rocm-libraries/                     # rocm-systems uses the same layout
  projects/hipdnn/                  # native libraries
  shared/origami/
  bindings/
    fortran/<lib>/                  # RFC0015 update
    python/
      README.md                     # points to this RFC
      rocm-origami/
      hipdnn-frontend/
        pyproject.toml
        README.md                   # user documentation
        CMakeLists.txt              # extension build: find_package() against installed ROCm
        src/                        # extension sources and the hipdnn_frontend import package
        tests/
```

- The directory name is the package's `pip install` name (`[project] name`); the import package uses underscores.
- A component may ship several packages, each in its own directory. One project builds both the extension and the wheel.
- The extension finds its component with `find_package(<package> CONFIG REQUIRED)`, never through `add_subdirectory()` or relative paths into the native project.
- The component team owns `bindings/python/<package-name>/` through CODEOWNERS, including its user documentation. TheRock maintainers own the packaging and validation workflows.

### Mixing generated and custom bindings

A custom binding may depend on `hip-python`. For example, hipFile needs error capture on one or two functions that a generated wrapper does not provide. In order of preference:

1. Extend `hip-python` through its override mechanism, if the behavior is general and its owners agree.
1. Add a narrow custom extension for only those functions, re-exporting the generated bindings for the rest.
1. Write a full custom binding only when the API is not a C API wrapper, as with hipDNN's graph API.

The custom package owns the user-facing import and ROCm initialization, and follows the same validation.

## Building and loading

### Source builds

Developers, and users with a system ROCm, build from the checkout. The build finds ROCm through `CMAKE_PREFIX_PATH`, which `pip` passes into its isolated build:

```sh
# Against ROCm SDK wheels (indexes and device extras: RELEASES.md)
python -m pip install "rocm[libraries,devel]"
ROCM_SDK_VERSION="$(rocm-sdk version)" CMAKE_PREFIX_PATH="$(rocm-sdk path --cmake)" \
  python -m pip install ./bindings/python/hipdnn-frontend

# Against a system ROCm
CMAKE_PREFIX_PATH=/opt/rocm python -m pip install ./bindings/python/hipdnn-frontend
```

The build fails if the component, or a compatible version of it, is missing.

### ROCm metadata and loading

A build against SDK wheels receives the SDK version through `ROCM_SDK_VERSION`, set from `rocm-sdk version`, as in the example above; TheRock's binding wheel job (see [Build](#build)) always sets it. With it, the build adds:

- an install requirement on that SDK (see [Release](#release)), following [`python_packaging.md`](../packaging/python_packaging.md);
- `device-<target>` extras that forward to the matching `rocm` device wheels, so users select device code the same way as for the SDK;
- a generated `_rocm_init.py` that calls `rocm_sdk.initialize_process()` before the extension is imported, because SDK wheel libraries are not on the loader's search path.

Without it, the build adds none of these, and the package loads ROCm from the system loader path, which for a tarball install means setting `LD_LIBRARY_PATH`.

In both cases, extensions carry only `$ORIGIN`-relative RPATHs, and wheel repair tools must not copy ROCm libraries into the wheel.

### Python versions

- New native extensions use the CPython Stable ABI (`abi3`): one wheel per platform covers every Python minor at or above its floor.
- `Requires-Python` states the floor and the wheel tags match the extension. CI tests every SDK-supported Python minor at or above the floor.
- A per-minor build is an exception, agreed with TheRock maintainers before it is added. Pure-Python packages need no matrix.

## Packaging and validation in TheRock

```text
super-repo revision + patches
  └─► native build ─► SDK wheels ─► binding wheel build ─► fresh-environment install
                                                                 └─► validate ─► record ─► publish
```

### Build

- Binding wheel jobs consume the SDK wheels built from the same super-repo revision, either in the same run or reused from a prior run; reuse records both revisions. A binding job can be rerun without rerunning the native build.
- There is one job per binding package and ABI tag: `abi3`, or one per Python minor under an exception. Job selection runs only the jobs whose component or binding sources changed, reusing cached native artifacts ([TheRock#7782](https://github.com/ROCm/TheRock/issues/7782), [TheRock#8651](https://github.com/ROCm/TheRock/issues/8651)).
- A feature flag ([`flags.md`](../development/flags.md)), off by default, gates the jobs during rollout. Once it is on, super-repo pull request CI runs the binding jobs for changed components, and a failure blocks the merge.

### Validate

For every platform and Python version the wheel claims:

1. **Consumer install.** In a fresh environment with no system ROCm on the loader path, install the wheel with the device extra for the test GPU and let `pip` resolve the SDK wheels. The resolved SDK must be the one the run built or selected.
1. **No conflicts.** The binding wheel installs no file an SDK wheel owns, and no SDK wheel contains a copy of the binding.
1. **Shipped libraries.** Import the package, call at least one native API, and confirm the loaded native libraries come from the SDK wheels. A bundled or substituted copy fails.
1. **Tests.** Run the binding's tests on CPU and GPU runners. Tests that generate native code, such as TensileLite kernel generation, are reported separately from tests of the shipped libraries.

Each run records the SDK build, SDK and binding package versions, source revisions with patches, wheel digests, Python version and environment, loaded library paths and versions, test inputs, generated artifacts, and results. The record is stored with the run's artifacts.

### Release

- **Version.** The binding's version plus a local label for its SDK build, from TheRock's version scripts ([`versioning.md`](../packaging/versioning.md)), for example `0.2.0+rocm10.1.0a20261119`.
- **Dependency.** Nightly and prerelease wheels pin the exact SDK build they were validated with, `rocm[libraries]==<SDK version>`, and check the same version at import. Installing one installs the matching SDK wheels, or moves an existing install to that build, so two packages install together only when they share an SDK build.
- **Stable releases.** Promotion to stable rewrites the pin to a patch-compatible range (`~=X.Y.Z`) and the import check to `X.Y.*`, as TheRock's promotion tooling does today for its SDK-dependent framework wheels, so later SDK patch and post releases work without a rebuild. Because promotion rewrites the wheel's metadata, the release records the validated wheel's digest next to the promoted one.
- **Index.** Binding wheels publish as separate wheels to the same per-channel Python index as the SDK wheels ([`RELEASES.md`](../../RELEASES.md)), only after passing validation.
- **Cadence.** Every SDK build ships its bindings, rebuilt for it. TheRock releases continuously, so a native change that breaks its binding is a build break, fixed by the component team in the same pull request.

## Rollout

1. **Directory setup.** Add `bindings/python/README.md`, pointing to this RFC, in `rocm-libraries` and `rocm-systems`.

1. **Origami pilot.** Move Origami's Python package to `bindings/python/rocm-origami/` as its own project that links the shared `liborigami` from the installed SDK, rejects a static one, and loads ROCm as described above. Origami's current combined native and Python build, including the shared-library linking from rocm-libraries#10975, moves into this project. Then add the TheRock job, validation, and publishing, including:

   - the `ROCM_SDK_VERSION` build input and `device-*` extras;
   - a package classification for binding wheels in TheRock's upload and release tooling, which today accepts only core, PyTorch, and JAX packages;
   - stable promotion support for binding wheels, including their device extras.

   The pilot is done when:

   - Origami installs with `pip` from its directory against an installed SDK without rebuilding or replacing `liborigami`;
   - TheRock builds, validates, and publishes its wheel with the matching SDK, with provenance recorded;
   - Origami's Python floor or per-minor exception is decided; and
   - the guide lets the next component follow without new design work.

After the pilot, these proceed independently:

- **hipDNN** moves `projects/hipdnn/python/` to `bindings/python/hipdnn-frontend/` as one project and retires its custom packer.
- **TensileLite** adopts the packaging and validation rules; its location is an open question.
- **Other components** adopt the layout as they add bindings. Maintainers review the first layouts closely in code review.
- A **developer guide** under `docs/packaging/` links back to this RFC.

## Stakeholders

Agreement is needed from the owners of hipBLASLt/TensileLite, Origami, hipDNN, rocSHMEM, rocjitsu, hipFile, and `hip-python`; ROCm Core; and TheRock build, CI, and release.

## Open questions

- **Python 3.10 and 3.11.** nanobind's Stable ABI needs Python 3.12, but the SDK supports 3.10 and 3.11 ([TheRock#5701](https://github.com/ROCm/TheRock/issues/5701)) and Origami's consumers include 3.11 environments. The Origami pilot either needs a per-minor build exception or accepts a 3.12 floor.
- **TensileLite.** The native hipBLASLt build uses TensileLite, and `rocisa` is built with hipBLASLt. Whether the package moves to `bindings/python/`, and whether `rocisa` leaves the native build, is settled with the hipBLASLt maintainers.
- **Import namespace.** Whether custom bindings use their own top-level import packages or the shared `rocm.*` namespace that `hip-python` keeps open is settled with the `hip-python` owners.

## Alternatives considered

**A separate `rocm-bindings` repository.** Binding updates would land after the native change, in another repository, and an independent release channel would validate bindings against past SDK releases rather than the SDK they ship with, which is the gap the Origami defect fell through.

**Bindings in the native build.** It puts Python version policy in the native build, and it leaked an extension into an SDK wheel (see [Why this matters](#why-this-matters)).

**Self-contained wheels.** Bundling the native library loads a second copy beside the SDK's, which is the Origami defect.

**Other SDK dependency modes.** An environment variable that points a published wheel at a system ROCm adds a second, unvalidated loading path. A wheel with no declared SDK dependency hides the requirement from `pip` until import. System-ROCm users build from source instead.

**Separate extension and wheel projects.** With `pip` building the extension and the wheel from one project, a split adds a handoff for no benefit.

## Related documents

- [RFC0008: Multi-Arch Packaging](./RFC0008-Multi-Arch-Packaging.md) and [RFC0012: Repo Structure](./RFC0012-Repo-Structure.md): device wheels and Python indexes.
- [RFC0015](./RFC0015-Fortran-Bindings-Split.md) and its update [TheRock#9433](https://github.com/ROCm/TheRock/pull/9433): Fortran bindings and the `bindings/` tree.
- [`python_packaging.md`](../packaging/python_packaging.md) and [`versioning.md`](../packaging/versioning.md): SDK wheels, `rocm-sdk`, and version formats.

## Revision history

- 2026-06-23: Initial draft (Brian Harrison).
- 2026-08-06: Address review feedback: Python package background, release channel, mixing generated and custom bindings (Brian Harrison).
- 2026-10-01: Merge `main`; number as RFC0017 (Brian Harrison).
- 2026-11-19: Replace the `rocm-bindings` repository with `bindings/python/<package-name>/` in each super-repo, per review on [TheRock#6050](https://github.com/ROCm/TheRock/pull/6050). TheRock builds, validates, and publishes binding wheels after the SDK, following its SDK-dependent wheel versioning and promotion. Add validation, provenance, and the Stable ABI policy. Name Origami as the pilot (Brian Harrison).
