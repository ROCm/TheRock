---
author: Brian Harrison (bharriso)
created: 2026-06-23
modified: 2026-08-06
status: draft
discussion: https://github.com/ROCm/TheRock/issues/6048
---

# Custom Python Bindings Layering

> **RFC number is a placeholder.** This document is filed as `RFC00XX` while it
> is under review. Several RFCs are in flight concurrently and the numbering
> space is contended — `RFC0013` is claimed by both this PR and
> [PR #6904](https://github.com/ROCm/TheRock/pull/6904) (ROCm Core Docker
> Standards), and `RFC0012` was similarly double-claimed earlier. The final
> number will be assigned immediately before merge, once the surrounding RFCs
> have landed and the next free number is unambiguous. The same placeholder
> convention is used by
> [PR #6034](https://github.com/ROCm/TheRock/pull/6034) and
> [PR #6118](https://github.com/ROCm/TheRock/pull/6118).

This RFC proposes the ownership and packaging model for ROCm library Python bindings that are not generated C API wrappers. hipDNN is the motivating example, but the model is intended to apply to any ROCm library that needs a custom Python binding layer over component-specific C, C++, or higher-level APIs.

## Summary

ROCm should separate native library ownership, generated C API bindings, custom Python binding extension builds, and final Python wheel packaging.

The proposed target state is:

1. **Native component repositories remain native-first.** `rocm-libraries/projects/<component>` owns native libraries, headers, CMake package exports, plugins, and native tests.
1. **TheRock / `rocm_sdk` publishes normal ROCm SDK native artifacts.** TheRock does not own custom Python binding version policy, extension ABI matrices, or final binding wheel release metadata.
1. **Custom binding projects live in `rocm-bindings`.** For hipDNN, `rocm-bindings/hipdnn/python/backend` owns the native Python extension build, and `rocm-bindings/hipdnn/python/frontend` owns the final wheel.
1. **`hip-python` is proposed to be relocated into the same bindings repository.** It remains the generated low-level C API binding area for HIP and ROCm library C APIs. `hipdnn_backend.h` is one example generator input, not a custom hipDNN frontend subproject.
1. **Binding wheels ship through the ROCm wheel indices defined by [RFC0012](./RFC0012-Repo-Structure.md), on their own cadence,** as an add-on to the ROCm Core SDK rather than as part of it.

This document describes the proposed target state. Current hipDNN sources still keep binding sources, package metadata, and tests under `projects/hipdnn/python`.

## Decision requested

Reviewers are asked to approve the ownership boundaries and artifact contracts in this RFC:

- custom binding projects live in `rocm-bindings`, with extension builds co-located with final wheel projects under a per-component `python/` directory;
- TheRock/`rocm_sdk` publishes normal native ROCm artifacts, not custom binding wheels or Python ABI matrices;
- relocated `hip-python` owns generated C API bindings;
- custom frontend wheels are ROCm SDK add-ons with explicit runtime, device, ABI, and validation contracts;
- binding wheels publish to the RFC0012 central `whl/` / `whl-next/` indices from a dedicated area, on a cadence the bindings repository owns.

Exact CI implementation, upload automation, wheel backend hooks, and durable developer how-to content are follow-up work after this RFC is accepted.

## Background: how the ROCm Python surfaces fit together

There are several distinct Python surfaces in the ROCm ecosystem and they are easy to conflate. This section fixes the vocabulary before the layering discussion.

| Surface                                                                              | What it is                                                                                                                                                                              | Where it is defined                                                         | Nature of the Python code                                                                             |
| ------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `rocm`, `rocm-sdk-core`, `rocm-sdk-libraries`, `rocm-sdk-device-*`, `rocm-sdk-devel` | The ROCm SDK wheels. Native binaries and libraries, plus the minimal Python needed to locate and preload them (`rocm_sdk`) for consumers such as PyTorch.                               | [`docs/packaging/python_packaging.md`](/docs/packaging/python_packaging.md) | Thin locator/loader glue over native payloads. Not an API binding.                                    |
| `hip-python` and the `rocm-bindings-*` wheels                                        | Generated 1:1 Cython bindings over the ROCm **C** APIs (HIP, math libraries, systems libraries, compiler). Machine-generated from headers.                                              | [PR #5609](https://github.com/ROCm/TheRock/pull/5609) (in discussion)       | Generated. Tracks the C API mechanically; no hand-authored Pythonic surface.                          |
| Custom component bindings (this RFC)                                                 | Hand-written Pythonic APIs over a component's C++ or higher-level API, where a 1:1 C wrapper is not the product. hipDNN's `Graph` / `TensorAttributes` frontend is the motivating case. | This RFC                                                                    | Hand-authored, component-owned, evolves with the component's Python UX rather than with its C header. |

The three are complements, not alternatives. A component may have all three: native libraries in the SDK, generated C API bindings in `hip-python`, and a custom Pythonic frontend covered by this RFC.

The high-level questions this raises, and this RFC's answers:

- **New packages or existing packages?** New packages. A custom frontend is neither an SDK wheel (it is not a native ROCm artifact) nor a generated binding (it is hand-authored over a C++ API). Folding it into `rocm-sdk-*` reintroduces the failure documented in [#5678](https://github.com/ROCm/TheRock/issues/5678); folding it into `hip-python` puts a hand-authored, component-owned API surface inside a generated one.
- **Who drives the development model?** The component team, inside `rocm-bindings`. TheRock and the SDK provide the native artifacts the binding builds against and loads at runtime; they do not own the binding's Python version matrix, ABI strategy, or release metadata. See [Proposed layering](#proposed-layering).
- **What are the dependencies between them?** A custom frontend wheel depends on the ROCm SDK **runtime** wheels for native libraries. It does **not** require `hip-python`. It **may** depend on `hip-python` when it wants generated C API access alongside its custom surface — see [Mixing generated and custom bindings](#mixing-generated-and-custom-bindings). Nothing in this RFC makes `hip-python` a mandatory dependency of a custom binding, and nothing forbids it.
- **Self-contained or SDK-dependent?** SDK-dependent. `pip install hipdnn-frontend` pulls the ROCm runtime wheels through `Requires-Dist` rather than vendoring ROCm shared libraries. See [`hipdnn-frontend` runtime dependency loading](#hipdnn-frontend-runtime-dependency-loading).

## Motivation

The current hipDNN Python binding prototype keeps the native extension build, the Python package metadata, the wheel packer, and the tests in `rocm-libraries/projects/hipdnn/python`, and its wheel was for a period assembled by TheRock-side CI helpers.

That has worked for experimentation, but it creates unclear ownership for production packaging:

- Native hipDNN builds would need to know Python wheel policy and Python ABI support.
- TheRock would be tempted to repack staged Python files into a wheel as a CI/testing convenience. It did, and the result was reverted in [PR #6425](https://github.com/ROCm/TheRock/pull/6425).
- Python API additions would require updates to extension code, wheel metadata, and Python tests that live in different ownership layers.
- Generated backend C API wrappers and custom Pythonic frontend APIs could be confused as one deliverable.

The concrete cost of the unclear boundary is already on record. [#5678](https://github.com/ROCm/TheRock/issues/5678) shipped a prebuilt `hipdnn_frontend_python.cpython-312-*.so` inside `rocm-sdk-devel` with a CI-local `/home/runner/...` RPATH and a cp312 lock, breaking `rocm-sdk test` on Python 3.13. That is precisely the class of defect that arises when a Python ABI artifact is treated as an ordinary native artifact by a layer that does not own Python ABI policy.

The target model puts the binding-specific release decisions in the bindings repository, where the extension build, wheel build, tests, and supported Python ABI matrix can evolve together.

## Goals

- Define where custom Python binding projects live.
- Define which layer owns native libraries, extension artifacts, final wheels, and generated C API bindings.
- Keep `rocm-libraries` native-first.
- Keep TheRock focused on normal ROCm SDK native artifacts.
- Let `rocm-bindings` own Python version support, ABI tagging, extension artifacts, wheel packaging, tests, and release/upload policy.
- Make build-time dependency resolution distinct from runtime dependency loading.
- Define the release channel for binding wheels in terms of the approved [RFC0012](./RFC0012-Repo-Structure.md) repository structure.
- Provide a repeatable pattern for other ROCm library custom bindings.

## Non-goals

- Define the final repository name or governance for `rocm-bindings`. This RFC uses `ROCm/rocm-bindings` as a placeholder; the repository does not exist yet.
- Define the full implementation plan for relocated `hip-python`; this RFC only proposes that it lives in the same bindings repository and owns generated C API bindings. The `hip-python` integration plan is [PR #5609](https://github.com/ROCm/TheRock/pull/5609).
- Change hipDNN native API ownership.
- Specify final upload buckets, release automation, or exact CI workflows.
- Require all custom bindings to use nanobind. hipDNN does, and this RFC uses hipDNN as the concrete example.
- Constrain non-Python language bindings. The directory layout reserves room for them; their contracts are out of scope.

## Proposed layering

| Layer                                       | Owns                                                                                                                                              | Does not own                                                                                    |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| `rocm-libraries/projects/<component>`       | Native library sources, headers, CMake package exports, plugins, native tests                                                                     | Custom Python extension sources, final Python wheel metadata, Python binding release policy     |
| TheRock / `rocm_sdk`                        | Native ROCm build orchestration, native artifact split, ROCm SDK runtime/devel/device packages                                                    | Custom Python binding version matrix, binding extension artifacts, final binding wheel metadata |
| `rocm-bindings/<component>/python/backend`  | C++/nanobind extension sources, extension build, ABI-specific extension artifacts, artifact manifest, producer-side tests                         | Final user-facing wheel metadata and installed-wheel tests                                      |
| `rocm-bindings/<component>/python/frontend` | Final Python wheel, import package, runtime ROCm initialization, package metadata, wheel tags, installed-wheel tests, docs, release/upload policy | Native library implementation, generated C API bindings, TheRock artifact split                 |
| relocated `hip-python`                      | Auto-generated Python bindings over ROCm C APIs across projects                                                                                   | Custom Pythonic frontend APIs such as hipDNN `Graph` and `TensorAttributes`                     |

For hipDNN, the target project split is:

```text
ROCm/rocm-bindings/
  hipdnn/
    python/
      backend/
        CMakeLists.txt               # native Python extension build
        src/                         # C++ binding sources
        pyext-manifest.json.in       # artifact manifest template
        tests/                       # producer-side build/load/manifest tests

      frontend/
        pyproject.toml               # final published wheel metadata
        src/hipdnn_frontend/         # checked-in Python import package
        tests/                       # installed-wheel/API/GPU tests
        samples/                     # source samples; install only by explicit policy
        tools/                       # artifact staging/build helpers

  hip-python/                        # proposed relocated generated C API binding project
    ...                              # generated Python bindings for HIP and library C APIs
```

The layout groups by component first and language second (`<component>/python/...`) so that a future Rust or C# binding for the same component is a sibling of `python/` rather than a parallel top-level tree. `backend` and `frontend` name the *role* in the binding stack — the native extension producer and the user-facing wheel — and deliberately avoid naming the binding technology, so a component that uses pybind11, Cython, or raw CPython C API instead of nanobind fits the same layout unchanged.

Note the term collision: hipDNN's own native libraries are already called `hipdnn_backend` and `hipdnn_frontend`. In this layout `python/backend` and `python/frontend` refer to the *binding* roles, not to those native libraries. For hipDNN specifically the two happen to align — the binding frontend wraps the native frontend — but that is coincidence, and a component whose binding names differ from its native library names is not required to rename anything.

`python/backend` intentionally has no `pyproject.toml` in the target layout. It is a CMake/native artifact producer, not a Python wheel project. The `pyproject.toml` belongs to `python/frontend`, which builds the installable wheel from a staged extension artifact.

The distribution/project name `hipdnn-frontend` and import package name `hipdnn_frontend` intentionally differ. Python distributions commonly use hyphens; Python import packages use underscores.

The native extension is not checked into the `python/frontend` source tree. It is built by `python/backend`, staged into a temporary wheel build tree, and then appears in the built wheel and installed package. If typing artifacts are generated, `python/frontend` owns packaging `.pyi` stubs and `py.typed` consistently.

## Artifact flow

```text
rocm-libraries/projects/hipdnn
  -> native hipDNN libraries, headers, CMake configs

TheRock / rocm_sdk
  -> normal ROCm SDK runtime/devel artifacts containing native hipDNN outputs

rocm-bindings/hipdnn/python/backend
  -> Python-native extension artifacts built with find_package() against installed ROCm/hipDNN devel artifacts

rocm-bindings/hipdnn/python/frontend
  -> installable wheels that stage the matching extension artifact
```

TheRock does not need to know the `hipdnn-frontend` Python version matrix. It only needs to publish the normal native hipDNN runtime/devel artifacts through `rocm_sdk` or equivalent native install artifacts.

`python/backend` decides which extension artifacts to build for the supported Python ABI surface. `python/frontend` consumes one matching artifact per wheel.

## Release channel and cadence

Binding wheels are an **extension on top of ROCm**, in the same sense as `rocm-examples` or RDC: they consume released ROCm artifacts and ship on their own cadence. They are not part of the ROCm Core SDK wheel set, and TheRock does not add `rocm-bindings` as a submodule or build binding wheels in its nightly pipeline.

This is a deliberate reversal of the earlier direction. TheRock briefly built and packaged hipDNN Python bindings ([PR #5429](https://github.com/ROCm/TheRock/pull/5429)); that wiring was removed in [PR #6425](https://github.com/ROCm/TheRock/pull/6425) on the grounds that bindings belong in a separate repository. This RFC records the resulting model rather than reopening it.

### Placement in the RFC0012 repository structure

[RFC0012](./RFC0012-Repo-Structure.md) is approved and defines the publish surface: two central PEP 503 indices per stream, `<stream>.repo.amd.com/rocm/whl/` and `<stream>.repo.amd.com/rocm/whl-next/`, fed by every wheel-producing area in that stream. RFC0012 does not currently name a bindings area, and this RFC must fill that gap explicitly rather than leave the destination implied.

**Proposal:** binding wheels publish from a per-project folder under `extras/`, and feed the central indices like every other extras project:

```text
<stream>.repo.amd.com/rocm/extras/rocm-bindings/
  whl/
  whl-next/
```

`extras/` is the right home under RFC0012's own definition — projects released independently, on their own cadence, rather than pinned to the Core SDK release train. The `omnistat/` entry is the existing precedent for a wheel-only extras project with no native packages. Per-project subfolders are retained under `extras/` so bucket permissions can be granted per project, which matters here because the bindings repository will have a different owner set than the SDK.

Consequences of that placement, all inherited from RFC0012 rather than invented here:

- Binding wheels appear in the same central `whl/` and `whl-next/` indices as `rocm`, so `pip install --index-url .../rocm/whl/ hipdnn-frontend` resolves the binding and its ROCm Core SDK dependencies in a single resolution pass.
- The ROCm major is carried by the `+rocm<major>` PEP 440 local-version tag on a single-named wheel, not by the package name. `hipdnn-frontend-0.2.0+rocm7` and `hipdnn-frontend-0.3.0+rocm8` are entries on the same project page.
- `whl-next/` carries the explicit-device-extras variant; `whl/` republishes with `device-all` added automatically. Binding wheels do not ship device payloads themselves, so they inherit whatever the ROCm Core SDK dependency resolves to.

**Open items for reviewers**, since these are the parts RFC0012 does not settle:

1. Whether the folder is `extras/rocm-bindings/` (one folder for the whole bindings repository) or `extras/<component>-bindings/` (one per component). The former is simpler and matches the single-repository model; the latter gives per-component bucket permissions. Recommendation: start with the single folder and split only if permissions demand it.
1. Whether an RFC0012 amendment is needed to name bindings in the `extras/` enumeration, or whether the existing "projects released independently" definition is sufficient without a doc change. Recommendation: a one-line addition to RFC0012's `extras/` list once this RFC is accepted.
1. Which streams bindings publish to. Recommendation: `nightly` and `stable` initially; `dev` only if per-commit binding builds prove useful.
1. The relationship to the existing PyPI presence. `hip-python` has published to PyPI/TestPyPI since 2023 and would gain a `repo.amd.com` channel on relocation. Whether PyPI remains a mirror, becomes the primary for bindings, or is retired is a decision for the `hip-python` owners in [PR #5609](https://github.com/ROCm/TheRock/pull/5609), but it must be answered — two independent channels publishing the same distribution name is a defect, not a feature.

### Cadence and pinning

The bindings repository picks its own release cadence and its own ROCm Core SDK pinning rule, following the pattern RFC0012 sets for expansions. Concretely:

- A binding wheel declares the ROCm version range it supports through `Requires-Dist`, and the exact ROCm version it was built and validated against through the extension artifact manifest.
- The bindings repository may cut a release without a ROCm release, provided the declared range still holds.
- A ROCm release does not require a binding release. Consumers on an older binding stay on it until the declared range excludes their SDK.

## Compatibility across the repository split

Once bindings live in a separate repository from the native libraries, the dependency flow is `rocm-libraries` → TheRock → `rocm-bindings`. The bindings repository is a downstream consumer of the public API that `rocm-libraries` produces, and it builds against released artifacts rather than against source.

That has a direct consequence for where regressions should be caught. Anything the bindings layer detects is one of two things: a breaking change to the public API that slipped through, or a critical defect in the native library. In either case, finding it at the bindings layer points to a gap in component-level testing. The bindings are too far downstream to be the right place to *first* catch a native regression.

So the model is not "use the bindings repository as the safety net":

- **`rocm-libraries` carries the public-API test coverage.** The public API is the contract; forward/backward-compatibility discipline belongs on that surface, in the repository that owns it. A binding update must not be able to break silently because the native side had no coverage of the API it depends on.
- **Synchronization happens at release boundaries.** The binding validates against the ROCm release artifacts it declares support for, at the cadence described above.
- **Version declarations make skew loud rather than silent.** Every layer declares the minimum version of each direct versioned input, and builds fail on mismatch — see [Version skew across layers](#version-skew-across-layers).
- **Manual pre-integration is a backstop, not the mechanism.** A component team landing a risky native API change can build the binding against a local ROCm build before the artifacts publish (see the local override path in [Cross-repo API sequencing](#cross-repo-api-sequencing)). That de-risks a specific change; it is not a substitute for native-side coverage.

## Mixing generated and custom bindings

A component may need both a generated C API surface and a small amount of hand-written binding code. The concrete case raised in review is hipFile: one or two functions need special error handling in the C binding layer before returning to Python — capturing `errno` or `hipPeekAtLastError()` — which a mechanically generated 1:1 wrapper does not produce.

This is explicitly allowed. A user-facing Python API may depend on both a generated binding package and a custom extension. Nothing in this layering forces an all-or-nothing choice, and requiring a component to hand-write a complete binding just to special-case two functions would be a bad outcome.

The ordering preference, strongest first:

1. **Extend the generated binding.** If the behavior is general — error capture on a call that can fail — it likely belongs in `hip-python`'s hand-coded override mechanism, which already exists for exactly this purpose. This keeps one binding package per C API and avoids a second artifact. Requires agreement from the `hip-python` owners.
1. **Add a narrow custom extension alongside the generated binding.** When the override path is unavailable or the behavior is genuinely component-specific, the component adds a small extension under `rocm-bindings/<component>/python/backend` covering only the functions that need it, and a frontend wheel that re-exports the generated bindings for everything else. The frontend wheel is then the single user-facing import, with `Requires-Dist` on both the generated binding package and its own staged extension.
1. **Full custom binding.** Only when the component's user-facing API is fundamentally not a C API wrapper. hipDNN is this case: its product surface is the C++ frontend graph API, and a generated `hipdnn_backend.h` wrapper is a different, complementary deliverable.

In all three cases the frontend wheel is the user-facing package and owns the import surface, the ROCm runtime initialization, and the release metadata. Which of the three a component uses does not change its position in the layering table.

Packaging consequence for case 2: the frontend wheel has a `Requires-Dist` on the relevant `rocm-bindings-*` distribution in addition to the ROCm runtime wheels, and its extension artifact manifest must record the generated-binding version it was built against, exactly as it records native ROCm versions. Two independently versioned inputs means two declared ranges.

## Relationship to `hip-python`

[PR #5609](https://github.com/ROCm/TheRock/pull/5609) proposes the `hip-python` integration and is in discussion. The boundary this RFC asserts, from the custom-bindings side:

- `hip-python` and the `rocm-bindings-*` wheels are **generated C API bindings**. Where they list a component — including hipDNN — that denotes generated wrappers over that component's C headers.
- Generated bindings over `hipdnn_backend.h` are a legitimate and useful deliverable. They do **not** replace `hipdnn_frontend`, which is a hand-written extension over the C++ frontend graph API (`Graph`, `TensorAttributes`, operation attributes, execution helpers). The two are separate packages with different ownership, different release policy, and different ABI strategies.
- A component appearing in a generated-bindings package list is not a statement that the component's Python story is complete.

Two contracts must be reconciled between the two RFCs before either is implemented, because a user can install both into one environment:

1. **Runtime library resolution.** This RFC specifies explicit `Requires-Dist` on the ROCm runtime wheels plus `rocm_sdk.initialize_process(..., check_version=...)`. A binding that instead falls back silently from wheel-installed ROCm to a system `/opt/rocm` will, in a mixed environment, load a different ROCm than its sibling — the failure family already seen in [#5678](https://github.com/ROCm/TheRock/issues/5678) and [#6314](https://github.com/ROCm/TheRock/issues/6314). The two packages must agree on one resolution order and one version-check policy.
1. **Repository naming.** Both RFCs use the name `rocm-bindings`, for different contents. This RFC uses it for a repository hosting per-component binding projects plus a relocated `hip-python`. The generated-bindings RFC additionally uses `rocm-bindings-*` as a wheel-name prefix meaning "generated C API bindings". A repository whose name implies all bindings, containing wheels whose shared prefix implies only generated ones, will be misread. Resolving this is a naming decision for both sets of authors; this RFC does not presume the answer.

## Current hipDNN state

Current hipDNN keeps the extension build, the Python package, the wheel packer, tests, and samples under `rocm-libraries/projects/hipdnn/python`. The tree is already split by role, which makes the proposed relocation close to a move rather than a restructure:

```text
projects/hipdnn/python/
  README.md
  download_third_party_deps.py
  frontend_bindings/
    CMakeLists.txt
    src/                            # module.cpp + per-area binding sources
  frontend_wheel_package/
    pyproject.toml
    pack_frontend_wheel.py
    src/hipdnn_frontend/__init__.py
    samples/
    tests/
```

Current facts:

- `python/frontend_bindings/CMakeLists.txt` is a standalone CMake project (`cmake_minimum_required(VERSION 3.26)`, `project(hipdnn_frontend_bindings)`). It is no longer driven by scikit-build and no longer has a parent-CMake mode.
- It resolves `hipdnn_frontend`, `hipdnn_backend`, and `hip` through `find_package(... CONFIG)` against `CMAKE_PREFIX_PATH`, with explicit `FATAL_ERROR` diagnostics naming the searched path when a package is missing.
- It builds `hipdnn_frontend_python` with `nanobind_add_module(... STABLE_ABI ...)`, requires `Python 3.12 ... Development.SABIModule`, and links `hipdnn_frontend`, `hipdnn_backend`, and `hip::host`.
- It sets `INSTALL_RPATH "$ORIGIN"`, so the extension resolves `libhipdnn_backend` as a sibling when co-installed. This is the fix for the CI-path RPATH leak reported in [#5678](https://github.com/ROCm/TheRock/issues/5678).
- `python/frontend_wheel_package/pack_frontend_wheel.py` stages the built extension into an importable `hipdnn_frontend` package and delegates to `python -m build --wheel`, generating a `setup.py` that sets `py_limited_api = "cp312"` and forces a platform wheel via `has_ext_modules`.
- `python/frontend_wheel_package/src/hipdnn_frontend/__init__.py` already implements the runtime initialization pattern this RFC codifies: preload via `rocm_sdk.initialize_process(preload_shortnames=...)` when ROCm SDK wheels are present, otherwise fall back to `ROCM_PATH`/`HIP_PATH`/`ROCM_HOME` with `os.add_dll_directory()` on Windows.
- `python/frontend_wheel_package/pyproject.toml` declares `name = "hipdnn-frontend"`, `requires-python = ">=3.12"`, and pytest markers. It is development metadata and is not yet the release-wheel contract — notably it carries no `Requires-Dist` on the ROCm runtime.
- **TheRock no longer builds or packages hipDNN Python bindings.** [PR #6425](https://github.com/ROCm/TheRock/pull/6425) (merged 2026-07-13) removed the TheRock-side wheel packer, the frontend pytest script, the CI matrix entry, the `share/hipdnn/python/**` artifact rules, and the `third-party/nanobind` and `third-party/robin-map` subprojects. The transitional packer this RFC previously described as "to be retired" is already gone.

Target changes:

- Move the extension project from `rocm-libraries/projects/hipdnn/python/frontend_bindings` to `rocm-bindings/hipdnn/python/backend`.
- Move the wheel project from `rocm-libraries/projects/hipdnn/python/frontend_wheel_package` to `rocm-bindings/hipdnn/python/frontend`.
- Keep native hipDNN APIs and CMake exports in `rocm-libraries/projects/hipdnn`.
- Add the artifact manifest, the `Requires-Dist` ROCm runtime declaration, and the wheel-tag validation described below; these do not exist today.

Because the current tree is already a standalone CMake extension project plus a separate staging/packaging project, the relocation preserves the existing build topology. The pieces this RFC adds are the manifest contract, the runtime dependency declaration, and the validation matrix — not a new build model.

## Build-time dependency resolution and runtime loading

There are two distinct dependency paths.

### `python/backend` build-time dependency resolution

`python/backend` builds a native Python extension. Its dependencies are resolved at configure/build time like a normal CMake project, not through Python runtime loading.

The project should consume installed ROCm/hipDNN devel artifacts through `CMAKE_PREFIX_PATH`, `ROCM_PATH`, or the equivalent `rocm_sdk` devel prefix. The expected pattern should use the native hipDNN frontend package as the direct frontend dependency, and should also declare a direct backend dependency while the binding exposes backend APIs such as plugin-path configuration:

```cmake
find_package(nanobind REQUIRED CONFIG)
find_package(hipdnn_frontend CONFIG REQUIRED)
find_package(hipdnn_backend CONFIG REQUIRED)
find_package(hip CONFIG REQUIRED)
```

The installed `hipdnn_frontend` CMake config carries its own transitive backend/data-sdk requirements, but the current binding also includes backend headers and calls backend extension APIs directly. If those backend-only bindings are removed later, the direct backend dependency and manifest fields can be dropped.

In build orchestration terms, `python/backend` has build-time dependencies on the native hipDNN frontend/backend development artifacts and HIP development package. If represented in a TheRock-style graph, this is a build dependency on installed native artifacts, not a runtime dependency of the final `hipdnn-frontend` wheel.

The `python/backend` manifest must state the minimum native ROCm, hipDNN frontend, and hipDNN backend versions it supports, plus the exact versions/source revisions used to build the artifact.

### `hipdnn-frontend` runtime dependency loading

The custom frontend wheel should depend on ROCm SDK runtime/library packages rather than bundle ROCm shared libraries. Its wheel metadata must declare the ROCm runtime dependency and version policy, for example:

```text
Requires-Dist: rocm[libraries] == <matching ROCm package version>
```

The exact version selector is a release policy decision, but it must be explicit and must match the native ROCm version recorded in the extension artifact manifest. Under [RFC0012](./RFC0012-Repo-Structure.md) the ROCm major is carried by the `+rocm<major>` local-version tag, so the selector must be written to interact correctly with local-version segments rather than assuming a bare version comparison.

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
- use the ROCm version recorded in the staged extension manifest or final wheel metadata as the `check_version` policy;
- choose whether version mismatches warn or fail as an explicit release policy;
- use `ROCM_PATH`, `HIP_PATH`, or `ROCM_HOME` for non-wheel/native installs;
- call `os.add_dll_directory()` on Windows fallback paths;
- keep loader setup separate from public API re-exports.

The current hipDNN `__init__.py` already implements the wheel/native fallback and the Windows DLL-directory registration. What it does not yet do is pass `check_version`, which is the part that turns a silent version mismatch into a diagnosable error.

The fallback order matters and must be explicit rather than incidental: when ROCm SDK wheels are installed they win, and a system `/opt/rocm` is used only when they are not. Preferring or silently falling through to a system install in a wheel environment produces exactly the mixed-ROCm failures seen in [#5678](https://github.com/ROCm/TheRock/issues/5678) and [#6314](https://github.com/ROCm/TheRock/issues/6314).

This makes `hipdnn-frontend` an add-on to `rocm_sdk`: it consumes and extends the SDK, but is not part of the SDK's core wheel set.

## Wheel metadata and platform policy

`hipdnn-frontend` wheels contain a native extension staged from `python/backend`, so they must be built as platform wheels, not pure Python wheels.

The wheel build must ensure:

- the wheel is not tagged `py3-none-any`;
- the wheel's Python, ABI, and platform tags match the staged extension artifact, for example `cp312-abi3-<platform>` or `cp311-cp311-<platform>`;
- Linux platform tags are not over-claimed. A `manylinux_*` tag is only valid when the wheel satisfies the relevant audit/package-index policy;
- external ROCm shared-library dependencies are satisfied by declared ROCm runtime packages and validated by release tests.

The follow-up packaging guide should define the exact backend hook/tooling that enforces `Root-Is-Purelib`, `WHEEL` tags, `RECORD` contents, and audit checks.

## Python ABI and artifact strategy

Python extension modules are not generic shared libraries. They must be treated as Python ABI artifacts.

`rocm-bindings`, not TheRock, owns the supported Python version matrix for custom binding wheels.

### Relationship to the ROCm SDK Python support floor

[Issue #5701](https://github.com/ROCm/TheRock/issues/5701) records the policy for extensions **shipped inside the `rocm-sdk-*` wheels**: a runtime-required extension must be importable on every supported Python (3.10–3.14), because those wheels are version-agnostic and are import-tested as a set. A test-only extension may be a documented sharp edge or excluded from the runtime wheel.

Custom binding wheels are a different case, and the distinction is load-bearing:

- A binding wheel is its own distribution with its own `Requires-Python`. `pip` will simply decline to install it on an unsupported interpreter, which is a correct and legible outcome.
- Shipping a cp312-only extension **inside** a version-agnostic SDK wheel is not legible; it produces an unimportable file that the SDK's own tests then trip over. That is [#5678](https://github.com/ROCm/TheRock/issues/5678).

The split proposed in this RFC therefore removes the #5678 failure mode structurally: binding extensions never ride inside SDK wheels, so they can never present an unimportable artifact to `rocm-sdk test`.

That is a constraint satisfied, not a coverage question answered. A binding wheel whose floor is above the SDK's floor leaves users on 3.10 and 3.11 with a working ROCm SDK and no binding. hipDNN is in exactly this position today: `Development.SABIModule`, `nanobind_add_module(... STABLE_ABI ...)`, and `requires-python = ">=3.12"` mean the current artifact cannot serve 3.10 or 3.11 at all, because the Python Limited API floor for nanobind's stable-ABI output is 3.12.

So each binding project must make an explicit, recorded decision:

- **Match the ROCm SDK support floor** by building a CPython-minor artifact family covering every supported Python. Costs one extension artifact per minor per platform.
- **Declare a higher floor** and accept that consumers below it cannot install the binding. Requires stating the floor in the release notes and in `Requires-Python`, and confirming it against the product's Python support matrix.

Neither is wrong in general; leaving it implicit is. For hipDNN the current implicit answer is a 3.12 floor, and this RFC asks reviewers to make it an explicit one.

### Stable ABI artifact family

Use this when:

- the binding can use the Python Limited API / stable ABI;
- the support floor for that wheel family can be Python 3.12 or newer;
- one extension per platform can serve all Python versions at or above that floor.

For hipDNN, the current source already points in this direction:

- CMake requires Python 3.12 with `Development.SABIModule`.
- `nanobind_add_module(... STABLE_ABI ...)` is used.
- The generated wheel `setup.py` sets `py_limited_api = "cp312"`.

`python/backend` should produce a stable-ABI artifact per platform. Linux example:

```text
hipdnn-python-backend-artifacts/
  cp312-abi3-<platform>/
    hipdnn_frontend_python.abi3.so
    hipdnn_frontend_python.pyi          # if typing support is generated/shipped
    py.typed                            # include when shipping typing support
    pyext-manifest.json
```

Windows uses the platform's Python extension suffix, for example `.pyd`, while preserving the same Python/ABI/platform-tag contract.

`python/frontend` should produce a wheel like:

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

`python/backend` must then build one extension artifact per CPython minor and platform:

```text
hipdnn-python-backend-artifacts/
  cp3XY-cp3XY-<platform>/
    hipdnn_frontend_python.<platform extension suffix>
    pyext-manifest.json
```

`python/frontend` then builds matching wheels:

```text
hipdnn_frontend-<version>-cp310-cp310-<platform>.whl
hipdnn_frontend-<version>-cp311-cp311-<platform>.whl
hipdnn_frontend-<version>-cp312-cp312-<platform>.whl
```

Never package a normal `cpython-312` extension as if it were Python-minor-independent. This is the concrete defect in [#5678](https://github.com/ROCm/TheRock/issues/5678): a `cpython-312`-tagged extension shipped as the only Python version, which then failed to load on 3.13 on a private CPython symbol removed in that release.

## Extension artifact manifest

Every `python/backend` artifact should include a manifest. The manifest records the artifact identity and compatibility contract:

- producer and consumer project names;
- Python distribution, import package, and extension module names;
- Python tag, ABI tag, platform tag, and `Requires-Python` floor;
- ROCm runtime package requirement and platform-specific preload shortnames;
- minimum supported and built-against native ROCm, hipDNN frontend, and hipDNN backend versions;
- generated-binding package requirement and built-against version, when the frontend depends on one (see [Mixing generated and custom bindings](#mixing-generated-and-custom-bindings));
- native and binding source revisions;
- extension artifact digest;
- manifest schema version.

`python/frontend` must fail the wheel build when the selected artifact, package metadata, runtime preload policy, or native input versions do not match the requested wheel target.

Typing support is optional. If typing artifacts are generated, the manifest and wheel build must ensure `.pyi` stubs and `py.typed` are packaged together. The current hipDNN source does not ship these today.

## Validation and release workflow

CI must validate every Python ABI/CPython target in the supported artifact and wheel matrix.

At minimum, each matrix entry must:

- build the matching `python/backend` artifact;
- verify the artifact filename, ABI tag, manifest, and digest;
- build the matching `python/frontend` wheel;
- verify the wheel is non-pure and has the expected Python, ABI, and platform tags;
- install the wheel into a clean environment for that Python tag;
- import `hipdnn_frontend`;
- run CPU/API tests;
- run GPU tests when hardware is available.

Do not validate one Python environment and infer that the others work. For `abi3` wheels, test every supported CPython minor that the wheel claims to support.

Two checks specific to the add-on model, both derived from defects already observed:

- **Clean-environment install.** Install the wheel into an environment with no system ROCm on the loader path, so that a missing `Requires-Dist` or a broken preload fails the test rather than being masked by a developer machine's `/opt/rocm`.
- **RPATH audit.** Reject any staged extension carrying a build-machine-absolute RPATH. [#5678](https://github.com/ROCm/TheRock/issues/5678) found 40 of 404 shipped `.so` files in the `rocm-sdk-*` wheels carrying a `/home/runner/...` RPATH; binding artifacts must use `$ORIGIN`-relative RPATHs, which the current hipDNN `CMakeLists.txt` already sets.

## Alternatives considered

### Keep the binding project in the native hipDNN layer

The original option was to keep `hipdnn_frontend_python` in `rocm-libraries/projects/hipdnn/python` and have TheRock/native hipDNN builds produce Python extension artifacts for the frontend wheel project to consume.

This was rejected as the preferred target because it splits one logical binding change across layers:

- a native API addition lands in `rocm-libraries`;
- the binding must be updated in the native layer;
- Python package metadata and installed-wheel tests still live in `rocm-bindings`;
- CI must wait for native artifacts to propagate before the Python wheel/test project can validate the binding change end-to-end.

That split makes it hard to land the C++ binding, Python packaging, and Python tests as one reviewable unit.

It also pushes Python release policy down into the native layer. The native hipDNN build would need to know which Python versions, ABI families, and wheel artifacts `rocm-bindings` plans to support, then generate and ship extension artifacts for that support matrix on native builds. That is undesirable when the native library may be built even when no Python binding wheel is being released.

The preferred target moves the binding project into `rocm-bindings/hipdnn/python/backend` because the binding project owns:

- the supported Python version/ABI matrix;
- extension artifact naming and manifest schema;
- release artifact selection;
- compatibility with the final wheel project;
- producer-side tests that should run with the frontend wheel tests.

This keeps release artifacts isolated to the owning binding repository and lets binding changes, packaging changes, and tests move together. The split can be revisited if reviewers prefer tighter native-layer ownership, but the expected cost is more sequencing friction and more Python-version knowledge in the native build.

### Build binding wheels in TheRock's release pipeline

TheRock could add `rocm-bindings` as a submodule and build binding wheels alongside the SDK, giving one release train and no cross-repo sequencing.

This was tried and reverted. [PR #5429](https://github.com/ROCm/TheRock/pull/5429) wired hipDNN binding builds, packaging, and wheel-validation CI into TheRock; [PR #6425](https://github.com/ROCm/TheRock/pull/6425) removed all of it. The reasons the model did not hold:

- It puts the Python ABI matrix in the layer that does not own it, which produced [#5678](https://github.com/ROCm/TheRock/issues/5678) — a cp312-only extension leaking into `rocm-sdk-devel` and breaking `rocm-sdk test`.
- It forces every binding release onto the SDK cadence, including binding-only fixes.
- It requires TheRock to carry binding-only build dependencies (`nanobind`, `robin-map` were added as `third-party` subprojects, then removed).

The counter-argument is real: a separate repository means the sequencing friction described in [Cross-repo API sequencing](#cross-repo-api-sequencing). The mitigations there — declared minimum versions, fail-fast, local override paths — are the accepted cost.

### Directory layout alternatives

Two alternatives to `rocm-bindings/<component>/python/[backend,frontend]` were considered:

- `rocm-bindings/<component>/<component>-nanobind` and `<component>-frontend` (the layout in the first draft of this RFC). Rejected because `nanobind` is an implementation detail that would appear in the directory name of every component that happens to use it, and because it leaves no room for non-Python bindings.
- `rocm-bindings/python/<component>/[backend,frontend]`, grouping by language first. Rejected because a component's bindings in different languages are more likely to be reviewed and released together than all components' Python bindings are.

## Migration plan

1. **Agree on this RFC.** Decide the ownership boundaries, repository layout, release channel, and artifact contracts.
1. **Create the `rocm-bindings` repository.** It does not exist yet. Establish the `<component>/python/[backend,frontend]` layout and reserve `rocm-bindings/hipdnn/`.
1. **Relocate `hip-python`** into the same repository as the generated C API binding area, coordinated with [PR #5609](https://github.com/ROCm/TheRock/pull/5609).
1. **Move hipDNN binding sources.** Move `projects/hipdnn/python/frontend_bindings/` to `rocm-bindings/hipdnn/python/backend/` and `projects/hipdnn/python/frontend_wheel_package/` to `rocm-bindings/hipdnn/python/frontend/`.
1. **Add the release contract that does not exist today.** `Requires-Dist` on the ROCm runtime wheels, `check_version` in the loader, and an explicit decision on the Python support floor.
1. **Define and validate the manifest schema.** Make wheel builds fail on mismatched ABI, Python tag, platform tag, native input versions, or extension module names.
1. **Wire CI.** Build backend artifacts, build frontend wheels, install them into clean environments, and run CPU/GPU tests across the full declared matrix.
1. **Establish the publish path.** Create the `extras/rocm-bindings/` area per [Release channel and cadence](#release-channel-and-cadence) and amend [RFC0012](./RFC0012-Repo-Structure.md) to name it.
1. **Publish developer documentation.** After the RFC is accepted and implementation details settle, add a durable developer guide under TheRock docs, likely in `docs/packaging/`, and link it back to this RFC.

## Risks and mitigations

### Cross-repo API sequencing

A native hipDNN API addition lands first in `rocm-libraries`. The binding update cannot be fully built/tested until that native change is available through TheRock/ROCm SDK artifacts.

```text
rocm-libraries native API change
  -> TheRock / rocm_sdk artifact update
  -> rocm-bindings/hipdnn/python/backend update
  -> rocm-bindings/hipdnn/python/frontend wheel/test update
```

Mitigations:

- Record minimum native artifact version/source revision in the backend manifest.
- Fail fast when the required native API is absent.
- Provide a local override path so developers can point `python/backend` at a local hipDNN build before TheRock artifacts publish.
- Keep binding API additions, extension changes, and frontend tests in the same `rocm-bindings/hipdnn` review whenever possible.

### Backend/frontend handoff gap

If `python/backend` only publishes native extension artifacts and `python/frontend` owns installed-wheel tests, regressions can hide at the handoff boundary.

Examples:

- wrong ABI tag;
- wrong extension filename;
- missing package data;
- loader failures from missing ROCm preloads;
- missing symbols/classes;
- tests landing only after the extension is already available to consumers.

Mitigations:

- Keep both projects under `rocm-bindings/hipdnn/python`.
- Run an integrated CI path: build backend artifact -> build frontend wheel -> install wheel -> run frontend tests.
- Keep producer-side smoke tests in `python/backend` and full installed-wheel tests in `python/frontend`.
- Let `python/frontend` consume a freshly built local backend artifact for development.

### Release channel does not materialize

This RFC and [PR #5609](https://github.com/ROCm/TheRock/pull/5609) both assume a bindings release channel that does not exist: `ROCm/rocm-bindings` is not a repository, and [RFC0012](./RFC0012-Repo-Structure.md) does not name a bindings area. Meanwhile [PR #6425](https://github.com/ROCm/TheRock/pull/6425) has already removed the TheRock-side path. Binding work is currently between two homes, with the old one demolished and the new one unbuilt.

Mitigations:

- Treat the [Release channel and cadence](#release-channel-and-cadence) decisions as blocking for this RFC rather than as follow-up work.
- Land the RFC0012 amendment naming the bindings area alongside repository creation, so the publish destination exists before the first wheel is built.
- Until the channel exists, keep hipDNN bindings buildable from source in `rocm-libraries` and do not ship them in SDK artifacts. This is the current state and is a stable holding position, not a regression.

### Temporary packers becoming release policy

Transitional wheel packers are useful for CI experimentation, but they carry placeholder metadata and CPython-specific tag behavior. They must not become the release path by accident.

The TheRock-side instance of this risk is closed: [PR #6425](https://github.com/ROCm/TheRock/pull/6425) removed it. The remaining instance is `projects/hipdnn/python/frontend_wheel_package/pack_frontend_wheel.py`, which is development tooling and is not a release contract.

Mitigations:

- Keep final wheel metadata only in `python/frontend`.
- Do not reintroduce binding wheel packing into TheRock.
- Treat any packer outside `rocm-bindings` as development-only and label it as such in-tree.

### Version skew across layers

Each layer depends on direct versioned inputs: `python/backend` depends on native ROCm, hipDNN frontend, and hipDNN backend devel artifacts, and `python/frontend` depends on a compatible backend artifact plus ROCm runtime/library packages. If those minimum input versions are implicit, an artifact can build or install successfully but fail at import/runtime because the native API, extension ABI, or loader expectations do not match.

Mitigations:

- Every component artifact must declare the minimum version of each direct versioned input it supports.
- For hipDNN, `python/backend` declares minimum supported native ROCm, native `hipdnn_frontend`, and native `hipdnn_backend` versions while it exposes backend APIs directly.
- `python/frontend` declares the minimum compatible backend artifact/schema version it can consume.
- Include exact built-against source revisions and native versions in `pyext-manifest.json`.
- Include the manifest in build artifacts and release evidence.
- Make extension builds and wheel builds reject mismatched native/extension/frontend inputs before publishing.

### Divergent runtime resolution between binding packages

A user can install a custom frontend wheel and a generated `hip-python` binding into the same environment. If they disagree about how to find ROCm — one preferring wheel-installed libraries with a version check, the other silently falling back to a system `/opt/rocm` — the process can end up with two ROCm versions loaded, or with a binding bound to a ROCm the user did not install.

Mitigations:

- Standardize on one resolution order across all binding packages: ROCm SDK wheels first, then `ROCM_PATH`/`HIP_PATH`/`ROCM_HOME`, then platform defaults.
- Standardize on `rocm_sdk.initialize_process(..., check_version=...)` as the preload mechanism when SDK wheels are present.
- Make the version-mismatch policy explicit per package and visible in the manifest.
- Add a mixed-install CI case once a second binding package exists.

## Follow-up developer documentation

This RFC is the decision record. After agreement, create a follow-up documentation PR that distills the stable guidance into a developer-facing page.

Recommended destination:

- `docs/packaging/custom_python_bindings.md`, because the guidance primarily covers Python packaging, ROCm SDK add-ons, ABI artifacts, and wheel release ownership.

If maintainers prefer the development section, use:

- `docs/development/custom_python_bindings.md`

Either way, the developer document should link back to this RFC and avoid restating the full alternatives discussion.

## Related documents

- [`docs/rfcs/README.md`](/docs/rfcs/README.md) — RFC process and metadata expectations.
- [`docs/rfcs/RFC0003-Build-Tree-Normalization.md`](/docs/rfcs/RFC0003-Build-Tree-Normalization.md) — build tree/source organization background, including forward compatibility with language binding layers.
- [`docs/rfcs/RFC0008-Multi-Arch-Packaging.md`](/docs/rfcs/RFC0008-Multi-Arch-Packaging.md) — device package split that binding wheels inherit through their ROCm runtime dependency.
- [`docs/rfcs/RFC0010-Test-Scripts-Migration.md`](/docs/rfcs/RFC0010-Test-Scripts-Migration.md) — precedent for moving ownership to the repository where code and tests co-evolve.
- [`docs/rfcs/RFC0012-Repo-Structure.md`](/docs/rfcs/RFC0012-Repo-Structure.md) — approved repository/index structure on `repo.amd.com`; defines the `whl/` and `whl-next/` indices this RFC publishes into.
- [`docs/development/build_system.md`](/docs/development/build_system.md) — build/runtime dependency distinction and CMake `find_package()` dependency resolution.
- [`docs/packaging/python_packaging.md`](/docs/packaging/python_packaging.md) — ROCm Python packaging model, `rocm_sdk`, and framework build/runtime initialization guidance.
- [`docs/development/artifacts.md`](/docs/development/artifacts.md) — TheRock artifact components and runtime/devel split.
- [PR #5609](https://github.com/ROCm/TheRock/pull/5609) — HIP Python integration RFC (generated C API bindings); in discussion.
- [PR #6425](https://github.com/ROCm/TheRock/pull/6425) — removal of hipDNN Python bindings from TheRock; establishes the separate-repository direction this RFC builds on.
- [Issue #5678](https://github.com/ROCm/TheRock/issues/5678) — hipDNN frontend extension shipped in `rocm-sdk-devel` with a dead RPATH and cp312 lock.
- [Issue #5701](https://github.com/ROCm/TheRock/issues/5701) — recorded policy for Python extensions shipped inside ROCm SDK wheels.

## Revision history

- 2026-06-23: Initial draft (Brian Harrison)
- 2026-08-06: Address review feedback. Add the background section on how the ROCm Python surfaces relate; adopt the `<component>/python/[backend,frontend]` layout; add the release channel/cadence section grounded in RFC0012; add sections on cross-repo compatibility, mixing generated and custom bindings, and the relationship to `hip-python`; reconcile the Python ABI strategy with the recorded SDK policy in #5701; refresh the hipDNN current-state section for the restructured source tree and the merged TheRock revert (#6425); hyperlink related documents. Filed under a placeholder RFC number pending assignment at merge.
