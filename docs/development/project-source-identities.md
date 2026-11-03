# Project source identities for classes 3–5

`THEROCK_PROJECT_SMREV=ON` enables reviewed, project-scoped source identities for
Git subprojects on native Linux. Input declarations live beside each target in
its `CMakeLists.txt`; the Python generator receives the configured inputs from
CMake. The option defaults to OFF.

## Scope

| Class | Configured targets | Reviewed declarations | Source selection |
| --- | ---: | ---: | --- |
| 3: rocm-systems | 34 | 34 | Project tree and required shared trees/files |
| 4: rocm-libraries | 31 | 35 | Project tree, shared sources and generators; four alternative primitive-library targets are included |
| 5: separate Git checkouts | 7 | 7 | Complete tracked repository tree |

Class 5 includes `amd-llvm`, `amd-comgr`, `hipcc`, `hipify`, `rocm-cmake`,
`rocm-half`, and `libhipcxx`. The LLVM targets share the complete LLVM source
scope and retain their individual configuration and dependency fingerprints.
Classes 1–2 retain their fetched markers, including the SIMDe/OpenBLAS fix.
Classes 6–7 retain their existing handling. A monorepo target without a reviewed
source declaration fails configuration when the option is ON.

## Declare inputs beside the build

AMD SMI's declaration in `core/CMakeLists.txt` includes:

```cmake
therock_cmake_subproject_declare(amdsmi
  FPRINT_PROJECT_SOURCE
  FPRINT_PROJECT_INPUTS "shared/cuid" "shared/sha256"
  FPRINT_PROJECT_OPTIONAL_INPUTS ${THEROCK_PROJECT_ROOT_INPUTS}
  FPRINT_PROJECT_ENVIRONMENT
    ${THEROCK_PROJECT_ROCM_VERSION_ENVIRONMENT}
    ${THEROCK_PROJECT_PACKAGE_ENVIRONMENT}
    "DESTDIR" "HOME" "ROCM_PATH" "ROCM_RPATH"
  FPRINT_PROJECT_VERSION_QUERIES "latest:amdsmi_pkg_ver*" "latest:rsmi_so_ver*"
  EXTERNAL_SOURCE_DIR "${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/projects/amdsmi"
  # Existing build arguments follow.
)
```

The selected source directory supplies the default scope, so AMD SMI's main
source path is not repeated in a separate catalog. Its source identity includes
`projects/amdsmi`, `shared/cuid`, and `shared/sha256`, plus the declared optional,
version and environment inputs. Paths are relative to the containing Git root.

| Argument | Meaning |
| --- | --- |
| `FPRINT_PROJECT_SOURCE` | Opt this declaration into project source identity when the global option is ON. Incompatible with `FPRINT_SOURCE_HASH`. |
| `FPRINT_PROJECT_SCOPE` | Replace the default scope with a repository-relative parent tree. Use `.` for COMGR/hipcc to include the entire LLVM repository, or a parent project for nested tests/samples. The scope must cover the selected source directory. |
| `FPRINT_PROJECT_INPUTS` | Additional required source trees or files. |
| `FPRINT_PROJECT_OPTIONAL_INPUTS` | Additional paths whose absence or presence is recorded. |
| `FPRINT_PROJECT_ENVIRONMENT` | Environment variable names whose values enter the identity. |
| `FPRINT_PROJECT_ENVIRONMENT_PATTERNS` | Full-match regular expressions for dynamically named environment variables, such as HIPIFY's CUDA sample roots. |
| `FPRINT_PROJECT_VERSION_QUERIES` | Ordered `latest:<glob>`, `nearest:<glob>`, `nearest-annotated:<glob>`, `exact`, or `timestamp:<path>` queries. `latest` selects the highest version-sorted tag; `nearest` uses reachable annotated/lightweight tags; `nearest-annotated` uses annotated tags only. |
| `FPRINT_PROJECT_UNCACHEABLE_REASON` | Generate a marker but invalidate the final fingerprint until the stated inputs are modeled. |

`cmake/therock_project_smrev.cmake` defines reusable lists for common optional
root paths and environment names. Targets explicitly select those lists; they
are not applied to every target automatically. Required and optional inputs,
environment names and version rules preserve the reviewed scope. This change
does not narrow those scopes or remove currently unset environment inputs.

Normal CMake conditions and variables can select active inputs. Changes to these
declarations cause CMake to reconfigure. The existing consumer-graph parser
recognizes the new arguments and continues treating `BUILD_DEPS`, `RUNTIME_DEPS`
and `COMPILER_TOOLCHAIN` as dependency relationships.

## Generation and validation

1. `therock_cmake_subproject_declare()` stores the input declarations as target
   properties. Activation calls `therock_project_smrev()` when source hashing
   is not selected.
2. The helper uses the configured source directory and Git root. Known monorepo
   roots receive stable logical names; separate repositories under TheRock use
   their repository-relative checkout paths. The helper passes expanded inputs
   to `build_tools/generate_project_smrev.py`.
3. The generator validates source state, resolves committed tree/blob IDs and
   declared version/environment values, and writes
   `build/source-identities/.<target>.smrev` atomically. Unchanged bytes retain
   their modification time.
4. It also writes `<target>.request.json` containing the configured source,
   logical repository, source entry, output path and complete input description.
   These are generated build files, not a manually maintained input catalog.
   `requests.txt` lists the requests for this configuration.
5. Build guards use those requests to recalculate and compare identities. They
   also run after subproject build/stage operations before recording success.
   Input drift fails the operation and removes the owned marker. Reconfigure
   before using request files created by a different request format.
6. Existing CMake code combines the source identity with configuration, selected
   file hashes and dependency fingerprints. Artifact aggregation is unchanged.

The helper records the marker lookup directory in `FPRINT_SOURCE_DIR` and reads
the validated marker directly. The actual source directory used by the build is
unchanged. Source-preparation markers, including LLVM's repository/revision
marker, are preserved. Generated markers and requests stay in the build tree.

Git `HEAD` locates committed objects and detects concurrent changes; its commit
ID is not inserted as the repository-wide source identity. Validation rejects
tracked modifications, relevant untracked files, hidden index flags, missing
required inputs, unsupported path traversal and mismatched nested checkouts.
Ignored files must be derived outputs, not independent undisclosed source inputs.
Nested repository presence, pins, tags and clean state remain accounted for.

## Version policy and limits

Reused artifacts retain the producer's commit/hash/count/branch labels. This
policy does not promise that rebuilding at another commit produces identical
version-label bytes. The code does not rewrite headers or restored artifacts.
Tag-derived release values and semantic timestamps remain inputs. History-based
queries require complete Git history. Do not change or initialize nested
repositories concurrently with a build.

The environment declarations are conservative: some fields apply only to certain
configurations. `null` means unset; unset, empty and a value remain distinct.
External tool/SDK paths do not identify their installed bytes. Completeness of
those inputs still requires the toolchain/environment audit.

`profiler-hub` and `mirage` generate markers but retain invalid final fingerprints
because their external fetch inputs are not fully modeled. Invalidity propagates
to consumers. KPACK split-artifact fingerprint emission and build-path
normalization remain separate work. This change does not enable cache restore
consumption, CI reuse or compilation avoidance.

The supported scope is native Linux and the reviewed repository layouts. Source
overrides for the known monorepos must retain those layouts and input paths.
Separate repositories outside TheRock and additional optional/platform-specific
targets require explicit support and review.

## Apply and validate the follow-up

Apply on the checkout containing the project-source-identity implementation:

```bash
git apply --check /path/to/therock-project-smrev-cmake-declarations.patch
git apply /path/to/therock-project-smrev-cmake-declarations.patch
python3 -m unittest discover -s build_tools/tests -p 'project_smrev_test.py' -v
python3 -m pytest build_tools/tests/cmake_consumer_graph_test.py -q
git diff --check
```

The source identity tests use temporary Git repositories and builds. CMake/Ninja
integration tests exercise declaration forwarding, conditions, broader scopes,
missing declarations, invalid argument combinations, all 76 declared targets,
source changes and build guards. Inspect the summary for skipped tests. The graph
tests require the project's pytest and cmake-parser dependencies. Git 2.31+,
CMake 3.25+ and Ninja are required by the integration fixtures; a complete TheRock
build may require newer CMake.

Review and commit the changed files before configuring the real checkout.
Other source routes still reject tracked modifications in the enclosing TheRock
repository. Keep submodule experiments and unrelated files out of that commit.
Then use the existing build configuration:

```bash
cmake -S . -B build -DTHEROCK_PROJECT_SMREV=ON
python3 build_tools/generate_project_smrev.py \
  --check-list build/source-identities/requests.txt
cmake --build build --target therock-check-project-identities
```

For unchanged source inputs, marker contents should match the pre-refactor
contents. Final `.fprint` values may change once because CMake declaration files
participate in fingerprinting. Repeat configuration must be stable. Use a
separate disposable checkout for committed project, shared-input and unrelated
changes. Compile/link logs and a separate consumer experiment are needed to
establish build avoidance and actual cache reuse.

## Review baseline

| Repository | Reviewed source revision |
| --- | --- |
| `.` | `a0e1df89a47ef3bf3e954569e35232e0f9072ba2` |
| `rocm-systems` | `430d3d8f489ee9fbc13764610902a7d344b72368` |
| `rocm-libraries` | `f4e24d58b3f313f1a31068def1e35376e43c2f49` |
| `base/rocm-cmake` | `10155d7272ea1bf79f6b5a9dbc339657af1aa372` |
| `base/half` | `207ee58595a64b5c4a70df221f1e6e704b807811` |
| `compiler/amd-llvm` | `36c1564ff815158e4fdff0880ba90b5d628aa045` |
| `compiler/hipify` | `d364ef407b2c3e20c6ceb3575bd852f0c7cebb8c` |
| `math-libs/libhipcxx` | `88f23f01fd7000c1ad25492311693e8e7ccabe31` |

These revisions record the review baseline; they are not source-identity inputs.
