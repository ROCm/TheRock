---
author: Brian Harrison (bharriso)
created: 2026-06-23
modified: 2026-06-23
status: draft
discussion: https://github.com/ROCm/TheRock/issues/6048
---

# Custom Python Bindings Layering

This RFC proposes the ownership and packaging model for ROCm library Python bindings that are not generated C API wrappers. hipDNN is the motivating example, but the model is intended to apply to any ROCm library that needs a custom Python binding layer over component-specific C, C++, or higher-level APIs.

## Summary

ROCm should separate native library ownership, generated C API bindings, custom Python binding extension builds, and final Python wheel packaging.

The proposed target state is:

1. **Native component repositories remain native-first.** `rocm-libraries/projects/<component>` owns native libraries, headers, CMake package exports, plugins, and native tests.
1. **TheRock / `rocm_sdk` publishes normal ROCm SDK native artifacts.** TheRock does not own custom Python binding version policy, nanobind extension matrices, or final binding wheel release metadata.
1. **Custom binding projects live in `rocm-bindings`.** For hipDNN, `rocm-bindings/hipdnn/hipdnn-nanobind` owns the native Python extension build, and `rocm-bindings/hipdnn/hipdnn-frontend` owns the final wheel.
1. **`hip-python` is proposed to be relocated into the same bindings repository.** It remains the generated low-level C API binding area for HIP and ROCm library C APIs. `hipdnn_backend.h` is one example generator input, not a custom hipDNN frontend subproject.

This document describes the proposed target state. Current hipDNN sources still keep nanobind sources, package metadata, and tests under `projects/hipdnn/python`.

## Decision requested

Reviewers are asked to approve the ownership boundaries and artifact contracts in this RFC:

- custom binding projects live in `rocm-bindings`, with nanobind extension builds co-located with final wheel projects;
- TheRock/`rocm_sdk` publishes normal native ROCm artifacts, not custom binding wheels or Python ABI matrices;
- relocated `hip-python` owns generated C API bindings;
- custom frontend wheels are ROCm SDK add-ons with explicit runtime, device, ABI, and validation contracts.

Exact CI implementation, upload automation, wheel backend hooks, and durable developer how-to content are follow-up work after this RFC is accepted.

## Motivation

The current hipDNN Python binding prototype mixes native build logic, nanobind extension build logic, Python package metadata, and tests in `rocm-libraries/projects/hipdnn/python`.

That has worked for experimentation, but it creates unclear ownership for production packaging:

- Native hipDNN builds would need to know Python wheel policy and Python ABI support.
- TheRock would be tempted to repack staged Python files into a wheel as a CI/testing convenience.
- Python API additions would require updates to nanobind code, wheel metadata, and Python tests that live in different ownership layers.
- Generated backend C API wrappers and custom Pythonic frontend APIs could be confused as one deliverable.

The target model puts the binding-specific release decisions in the bindings repository, where the extension build, wheel build, tests, and supported Python ABI matrix can evolve together.

## Goals

- Define where custom Python binding projects live.
- Define which layer owns native libraries, extension artifacts, final wheels, and generated C API bindings.
- Keep `rocm-libraries` native-first.
- Keep TheRock focused on normal ROCm SDK native artifacts.
- Let `rocm-bindings` own Python version support, ABI tagging, extension artifacts, wheel packaging, tests, and release/upload policy.
- Make build-time dependency resolution distinct from runtime dependency loading.
- Provide a repeatable pattern for other ROCm library custom bindings.

## Non-goals

- Define the final repository name or governance for `rocm-bindings`. This RFC uses `ROCm/rocm-bindings` as a placeholder.
- Define the full implementation plan for relocated `hip-python`; this RFC only proposes that it lives in the same bindings repository and owns generated C API bindings.
- Change hipDNN native API ownership.
- Specify final upload buckets, release automation, or exact CI workflows.
- Require all custom bindings to use nanobind. hipDNN does, and this RFC uses hipDNN as the concrete example.

## Proposed layering

| Layer                                            | Owns                                                                                                                                              | Does not own                                                                                     |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `rocm-libraries/projects/<component>`            | Native library sources, headers, CMake package exports, plugins, native tests                                                                     | Custom Python extension sources, final Python wheel metadata, Python binding release policy      |
| TheRock / `rocm_sdk`                             | Native ROCm build orchestration, native artifact split, ROCm SDK runtime/devel/device packages                                                    | Custom Python binding version matrix, nanobind extension artifacts, final binding wheel metadata |
| `rocm-bindings/<component>/<component>-nanobind` | C++/nanobind extension sources, extension build, ABI-specific extension artifacts, artifact manifest, producer-side tests                         | Final user-facing wheel metadata and installed-wheel tests                                       |
| `rocm-bindings/<component>/<component>-frontend` | Final Python wheel, import package, runtime ROCm initialization, package metadata, wheel tags, installed-wheel tests, docs, release/upload policy | Native library implementation, generated C API bindings, TheRock artifact split                  |
| relocated `hip-python`                           | Auto-generated Python bindings over ROCm C APIs across projects                                                                                   | Custom Pythonic frontend APIs such as hipDNN `Graph` and `TensorAttributes`                      |

For hipDNN, the target project split is:

```text
ROCm/rocm-bindings/
  hipdnn/
    hipdnn-nanobind/
      CMakeLists.txt                 # nanobind extension build
      src/                           # C++ binding sources
      pyext-manifest.json.in         # artifact manifest template
      tests/                         # producer-side build/load/manifest tests

    hipdnn-frontend/
      pyproject.toml                 # final published wheel metadata
      hipdnn_frontend/                # checked-in Python import package
      tests/                         # installed-wheel/API/GPU tests
      samples/                       # source samples; install only by explicit policy
      tools/                         # artifact staging/build helpers

  hip-python/                        # proposed relocated generated C API binding project
    ...                              # generated Python bindings for HIP and library C APIs
```

`hipdnn-nanobind` intentionally has no `pyproject.toml` in the target layout. It is a CMake/native artifact producer, not a Python wheel project. The `pyproject.toml` belongs to `hipdnn-frontend`, which builds the installable wheel from a staged nanobind artifact.

The distribution/project name `hipdnn-frontend` and import package name `hipdnn_frontend` intentionally differ. Python distributions commonly use hyphens; Python import packages use underscores.

The native extension is not checked into the `hipdnn-frontend` source tree. It is built by `hipdnn-nanobind`, staged into a temporary wheel build tree, and then appears in the built wheel and installed package. If typing artifacts are generated, `hipdnn-frontend` owns packaging `.pyi` stubs and `py.typed` consistently.

## Artifact flow

```text
rocm-libraries/projects/hipdnn
  -> native hipDNN libraries, headers, CMake configs

TheRock / rocm_sdk
  -> normal ROCm SDK runtime/devel artifacts containing native hipDNN outputs

rocm-bindings/hipdnn/hipdnn-nanobind
  -> Python-native extension artifacts built with find_package() against installed ROCm/hipDNN devel artifacts

rocm-bindings/hipdnn/hipdnn-frontend
  -> installable wheels that stage the matching nanobind artifact
```

TheRock does not need to know the `hipdnn-frontend` Python version matrix. It only needs to publish the normal native hipDNN runtime/devel artifacts through `rocm_sdk` or equivalent native install artifacts.

`hipdnn-nanobind` decides which extension artifacts to build for the supported Python ABI surface. `hipdnn-frontend` consumes one matching artifact per wheel.

## Current hipDNN state

Current hipDNN has native build logic, nanobind extension build logic, and Python package logic in `projects/hipdnn/python`.

Current facts:

- `projects/hipdnn/python/CMakeLists.txt` has both parent-CMake mode and scikit-build mode.
- In scikit-build mode, it finds `hipdnn_frontend`, `hipdnn_backend`, and HIP through `CMAKE_PREFIX_PATH`.
- It builds `hipdnn_frontend_python` with `nanobind_add_module(... STABLE_ABI ...)`.
- It links `hipdnn_frontend`, `hipdnn_backend`, and `hip::host`.
- It installs differently depending on mode: scikit-build installs into `hipdnn_frontend`, while parent-CMake mode installs the extension under lib/bin and Python files/tests under lib/python.
- `projects/hipdnn/python/pyproject.toml` is useful local/development scikit-build metadata. It should not be treated as the final release-wheel contract.
- TheRock's current hipDNN frontend wheel packer files are transitional CI helpers, and the staged frontend artifact path is currently not the desired active release path. They stage an existing `hipdnn_frontend` package directory and run a wheel build, but they should be retired once the binding projects own their build/release pipeline.

Target changes:

- Move the nanobind C++ extension project from `rocm-libraries/projects/hipdnn/python` to `rocm-bindings/hipdnn/hipdnn-nanobind`.
- Move final wheel/package metadata to `rocm-bindings/hipdnn/hipdnn-frontend`.
- Keep native hipDNN APIs and CMake exports in `rocm-libraries/projects/hipdnn`.
- Retire TheRock-specific frontend wheel repacking once the binding projects own their build/release pipeline.

The current scikit-build path goes away in the target layout because scikit-build is only needed when a Python wheel build is responsible for driving CMake. In the proposed split, `hipdnn-nanobind` is built directly as a native/CMake project, and `hipdnn-frontend` is a wheel-packaging project that stages prebuilt extension artifacts.

## Build-time dependency resolution and runtime loading

There are two distinct dependency paths.

### `hipdnn-nanobind` build-time dependency resolution

`hipdnn-nanobind` builds a native Python extension. Its dependencies are resolved at configure/build time like a normal CMake project, not through Python runtime loading.

The project should consume installed ROCm/hipDNN devel artifacts through `CMAKE_PREFIX_PATH`, `ROCM_PATH`, or the equivalent `rocm_sdk` devel prefix. The expected pattern should use the native hipDNN frontend package as the direct frontend dependency, and should also declare a direct backend dependency while the binding exposes backend APIs such as plugin-path configuration:

```cmake
find_package(nanobind REQUIRED CONFIG)
find_package(hipdnn_frontend CONFIG REQUIRED)
find_package(hipdnn_backend CONFIG REQUIRED)
find_package(hip CONFIG REQUIRED)
```

The installed `hipdnn_frontend` CMake config carries its own transitive backend/data-sdk requirements, but the current binding also includes backend headers and calls backend extension APIs directly. If those backend-only bindings are removed later, the direct backend dependency and manifest fields can be dropped.

In build orchestration terms, `hipdnn-nanobind` has build-time dependencies on the native hipDNN frontend/backend development artifacts and HIP development package. If represented in a TheRock-style graph, this is a build dependency on installed native artifacts, not a runtime dependency of the final `hipdnn-frontend` wheel.

The `hipdnn-nanobind` manifest must state the minimum native ROCm, hipDNN frontend, and hipDNN backend versions it supports, plus the exact versions/source revisions used to build the artifact.

### `hipdnn-frontend` runtime dependency loading

The custom frontend wheel should depend on ROCm SDK runtime/library packages rather than bundle ROCm shared libraries. Its wheel metadata must declare the ROCm runtime dependency and version policy, for example:

```text
Requires-Dist: rocm[libraries] == <matching ROCm package version>
```

The exact version selector is a release policy decision, but it must be explicit and must match the native ROCm version recorded in the nanobind artifact manifest.

For GPU-capable installs and release tests, the runtime contract must also include the required ROCm device packages. In kpack-split TheRock packaging, `rocm-sdk-libraries` contains host libraries while per-ISA device payloads live in `rocm-sdk-device-*` wheels selected through `rocm` extras such as `rocm[device-gfx942]` or `rocm[device-all]`. The release matrix must state which device package set is required for each GPU target.

Wheel metadata ensures packages are installed. It does not make the OS dynamic loader find native libraries in sibling wheels. The import package must initialize ROCm native dependencies before importing the extension.

Recommended import pattern:

```python
# hipdnn_frontend/__init__.py
from ._rocm_init import initialize_rocm

initialize_rocm()

from .hipdnn_frontend_python import *
```

`_rocm_init.py` should:

- call `rocm_sdk.initialize_process(preload_shortnames=[...], check_version=...)` when ROCm SDK wheels are installed;
- use the ROCm version recorded in the staged nanobind manifest or final wheel metadata as the `check_version` policy;
- choose whether version mismatches warn or fail as an explicit release policy;
- use `ROCM_PATH`, `HIP_PATH`, or `ROCM_HOME` for non-wheel/native installs;
- call `os.add_dll_directory()` on Windows fallback paths;
- keep loader setup separate from public API re-exports.

This makes `hipdnn-frontend` an add-on to `rocm_sdk`: it consumes and extends the SDK, but is not part of the SDK's core wheel set.

## Wheel metadata and platform policy

`hipdnn-frontend` wheels contain a native extension staged from `hipdnn-nanobind`, so they must be built as platform wheels, not pure Python wheels.

The wheel build must ensure:

- the wheel is not tagged `py3-none-any`;
- the wheel's Python, ABI, and platform tags match the staged extension artifact, for example `cp312-abi3-<platform>` or `cp311-cp311-<platform>`;
- Linux platform tags are not over-claimed. A `manylinux_*` tag is only valid when the wheel satisfies the relevant audit/package-index policy;
- external ROCm shared-library dependencies are satisfied by declared ROCm runtime packages and validated by release tests.

The follow-up packaging guide should define the exact backend hook/tooling that enforces `Root-Is-Purelib`, `WHEEL` tags, `RECORD` contents, and audit checks.

## Python ABI and artifact strategy

Python extension modules are not generic shared libraries. They must be treated as Python ABI artifacts.

`rocm-bindings`, not TheRock, owns the supported Python version matrix for custom binding wheels.

### Stable ABI artifact family

Use this when:

- the binding can use the Python Limited API / stable ABI;
- the support floor for that wheel family can be Python 3.12 or newer;
- one extension per platform can serve all Python versions at or above that floor.

For hipDNN, the current source already points in this direction:

- CMake requires Python 3.12 with `Development.SABIModule`.
- `nanobind_add_module(... STABLE_ABI ...)` is used.
- The current local scikit-build pyproject sets `wheel.py-api = "cp312"`.

`hipdnn-nanobind` should produce a stable-ABI artifact per platform. Linux example:

```text
hipdnn-nanobind-artifacts/
  cp312-abi3-<platform>/
    hipdnn_frontend_python.abi3.so
    hipdnn_frontend_python.pyi          # if typing support is generated/shipped
    py.typed                            # include when shipping typing support
    pyext-manifest.json
```

Windows uses the platform's Python extension suffix, for example `.pyd`, while preserving the same Python/ABI/platform-tag contract.

`hipdnn-frontend` should produce a wheel like:

```text
hipdnn_frontend-<version>-cp312-abi3-<platform>.whl
```

Required checks:

- The extension filename must match ABI3 expectations.
- The wheel tag must be `cp312-abi3-<platform>`, not `cp312-cp312-<platform>`.
- `Requires-Python` must be consistent with the ABI floor.
- The manifest, extension filename, package metadata, and wheel tag must agree.

### CPython-minor artifact family

Use this when:

- a supported Python version cannot use the stable ABI artifact;
- the binding cannot use stable ABI;
- the project intentionally supports Python versions below the stable ABI floor.

`hipdnn-nanobind` must then build one extension artifact per CPython minor and platform:

```text
hipdnn-nanobind-artifacts/
  cp3XY-cp3XY-<platform>/
    hipdnn_frontend_python.<platform extension suffix>
    pyext-manifest.json
```

`hipdnn-frontend` then builds matching wheels:

```text
hipdnn_frontend-<version>-cp310-cp310-<platform>.whl
hipdnn_frontend-<version>-cp311-cp311-<platform>.whl
hipdnn_frontend-<version>-cp312-cp312-<platform>.whl
```

Never package a normal `cpython-312` extension as if it were Python-minor-independent.

## Extension artifact manifest

Every `hipdnn-nanobind` artifact should include a manifest. The manifest records the artifact identity and compatibility contract:

- producer and consumer project names;
- Python distribution, import package, and extension module names;
- Python tag, ABI tag, platform tag, and `Requires-Python` floor;
- ROCm runtime package requirement and platform-specific preload shortnames;
- minimum supported and built-against native ROCm, hipDNN frontend, and hipDNN backend versions;
- native and binding source revisions;
- extension artifact digest;
- manifest schema version.

`hipdnn-frontend` must fail the wheel build when the selected artifact, package metadata, runtime preload policy, or native input versions do not match the requested wheel target.

Typing support is optional. If typing artifacts are generated, the manifest and wheel build must ensure `.pyi` stubs and `py.typed` are packaged together. The current hipDNN source does not ship these today.

## Validation and release workflow

CI must validate every Python ABI/CPython target in the supported artifact and wheel matrix.

At minimum, each matrix entry must:

- build the matching `hipdnn-nanobind` artifact;
- verify the artifact filename, ABI tag, manifest, and digest;
- build the matching `hipdnn-frontend` wheel;
- verify the wheel is non-pure and has the expected Python, ABI, and platform tags;
- install the wheel into a clean environment for that Python tag;
- import `hipdnn_frontend`;
- run CPU/API tests;
- run GPU tests when hardware is available.

Do not validate one Python environment and infer that the others work. For `abi3` wheels, test every supported CPython minor that the wheel claims to support.

## Alternatives considered

### Keep the nanobind project in the native hipDNN layer

The original option was to keep `hipdnn_frontend_python` in `rocm-libraries/projects/hipdnn/python` and have TheRock/native hipDNN builds produce Python extension artifacts for the frontend wheel project to consume.

This was rejected as the preferred target because it splits one logical binding change across layers:

- a native API addition lands in `rocm-libraries`;
- the nanobind binding must be updated in the native layer;
- Python package metadata and installed-wheel tests still live in `rocm-bindings`;
- CI must wait for native artifacts to propagate before the Python wheel/test project can validate the binding change end-to-end.

That split makes it hard to land the C++ binding, Python packaging, and Python tests as one reviewable unit.

It also pushes Python release policy down into the native layer. The native hipDNN build would need to know which Python versions, ABI families, and wheel artifacts `rocm-bindings` plans to support, then generate and ship extension artifacts for that support matrix on native builds. That is undesirable when the native library may be built even when no Python binding wheel is being released.

The preferred target moves the nanobind project into `rocm-bindings/hipdnn/hipdnn-nanobind` because the binding project owns:

- the supported Python version/ABI matrix;
- extension artifact naming and manifest schema;
- release artifact selection;
- compatibility with the final wheel project;
- producer-side tests that should run with the frontend wheel tests.

This keeps release artifacts isolated to the owning binding repository and lets binding changes, packaging changes, and tests move together. The split can be revisited if reviewers prefer tighter native-layer ownership, but the expected cost is more sequencing friction and more Python-version knowledge in the native build.

## Migration plan

1. **Agree on this RFC.** Decide the ownership boundaries, repository layout, and artifact contracts.
1. **Create or update the `rocm-bindings` repository layout.** Relocate `hip-python` there and reserve `rocm-bindings/hipdnn/` for custom hipDNN bindings.
1. **Move hipDNN nanobind sources.** Move `projects/hipdnn/python/src/*.cpp` and the extension CMake build into `rocm-bindings/hipdnn/hipdnn-nanobind`.
1. **Create the frontend wheel project.** Move final wheel metadata, import package, loader setup, tests, and samples into `rocm-bindings/hipdnn/hipdnn-frontend`.
1. **Define and validate the manifest schema.** Make wheel builds fail on mismatched ABI, Python tag, platform tag, native input versions, or extension module names.
1. **Wire CI.** Build nanobind artifacts, build frontend wheels, install them, and run CPU/GPU tests.
1. **Retire transitional TheRock packers.** Once `rocm-bindings` owns the build/release pipeline, remove TheRock-specific hipDNN frontend wheel repacking.
1. **Publish developer documentation.** After the RFC is accepted and implementation details settle, add a durable developer guide under TheRock docs, likely in `docs/packaging/`, and link it back to this RFC.

## Risks and mitigations

### Cross-repo API sequencing

A native hipDNN API addition lands first in `rocm-libraries`. The binding update cannot be fully built/tested until that native change is available through TheRock/ROCm SDK artifacts.

```text
rocm-libraries native API change
  -> TheRock / rocm_sdk artifact update
  -> rocm-bindings/hipdnn/hipdnn-nanobind update
  -> rocm-bindings/hipdnn/hipdnn-frontend wheel/test update
```

Mitigations:

- Record minimum native artifact version/source revision in the nanobind manifest.
- Fail fast when the required native API is absent.
- Provide a local override path so developers can point `hipdnn-nanobind` at a local hipDNN build before TheRock artifacts publish.
- Keep binding API additions, nanobind changes, and frontend tests in the same `rocm-bindings/hipdnn` review whenever possible.

### Nanobind/frontend handoff gap

If `hipdnn-nanobind` only publishes native extension artifacts and `hipdnn-frontend` owns installed-wheel tests, regressions can hide at the handoff boundary.

Examples:

- wrong ABI tag;
- wrong extension filename;
- missing package data;
- loader failures from missing ROCm preloads;
- missing symbols/classes;
- tests landing only after the extension is already available to consumers.

Mitigations:

- Keep both projects under `rocm-bindings/hipdnn`.
- Run an integrated CI path: build nanobind artifact -> build frontend wheel -> install wheel -> run frontend tests.
- Keep producer-side smoke tests in `hipdnn-nanobind` and full installed-wheel tests in `hipdnn-frontend`.
- Let `hipdnn-frontend` consume a freshly built local nanobind artifact for development.

### Temporary TheRock packer becoming release policy

The current TheRock hipDNN wheel packer is useful for CI experimentation, but it has placeholder metadata and CPython-specific tag behavior. It should not become the release path by accident.

Mitigations:

- Mark the TheRock packer as transitional.
- Retire it once `rocm-bindings/hipdnn` owns `hipdnn-nanobind` and `hipdnn-frontend`.
- Keep final wheel metadata only in `hipdnn-frontend`.

### Version skew across layers

Each layer depends on direct versioned inputs: `hipdnn-nanobind` depends on native ROCm, hipDNN frontend, and hipDNN backend devel artifacts, and `hipdnn-frontend` depends on a compatible `hipdnn-nanobind` artifact plus ROCm runtime/library packages. If those minimum input versions are implicit, an artifact can build or install successfully but fail at import/runtime because the native API, extension ABI, or loader expectations do not match.

Mitigations:

- Every component artifact must declare the minimum version of each direct versioned input it supports.
- For hipDNN, `hipdnn-nanobind` declares minimum supported native ROCm, native `hipdnn_frontend`, and native `hipdnn_backend` versions while it exposes backend APIs directly.
- `hipdnn-frontend` declares the minimum compatible `hipdnn-nanobind` artifact/schema version it can consume.
- Include exact built-against source revisions and native versions in `pyext-manifest.json`.
- Include the manifest in build artifacts and release evidence.
- Make extension builds and wheel builds reject mismatched native/extension/frontend inputs before publishing.

## Follow-up developer documentation

This RFC is the decision record. After agreement, create a follow-up documentation PR that distills the stable guidance into a developer-facing page.

Recommended destination:

- `docs/packaging/custom_python_bindings.md`, because the guidance primarily covers Python packaging, ROCm SDK add-ons, ABI artifacts, and wheel release ownership.

If maintainers prefer the development section, use:

- `docs/development/custom_python_bindings.md`

Either way, the developer document should link back to this RFC and avoid restating the full alternatives discussion.

## Related documents

- `docs/rfcs/README.md` — RFC process and metadata expectations.
- `docs/rfcs/RFC0003-Build-Tree-Normalization.md` — build tree/source organization background, including forward compatibility with language binding layers.
- `docs/rfcs/RFC0010-Test-Scripts-Migration.md` — precedent for moving ownership to the repository where code and tests co-evolve.
- `docs/development/build_system.md` — build/runtime dependency distinction and CMake `find_package()` dependency resolution.
- `docs/packaging/python_packaging.md` — ROCm Python packaging model, `rocm_sdk`, and framework build/runtime initialization guidance.
- `docs/development/artifacts.md` — TheRock artifact components and runtime/devel split.
