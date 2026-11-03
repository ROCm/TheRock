# Project source identities for classes 3–5

This opt-in local Linux implementation replaces repository-HEAD source identity
for registered Git targets with a generated, project-scoped `.smrev`. It targets
the supplied TheRock revision `a0e1df89a47ef3bf3e954569e35232e0f9072ba2`.
The source revisions reviewed are recorded in `build_tools/project_smrev_inputs.json`.

## Scope

| Class | Configured targets | Declared policies | Source selection |
| --- | ---: | ---: | --- |
| 3: rocm-systems | 34 | 34 | Project tree plus declared shared source trees/files |
| 4: rocm-libraries | 31 | 35 | Project tree plus declared shared sources and generators; includes four alternate primitive-library declarations |
| 5: separate Git checkouts | 7 | 7 | Complete tracked repository tree, with checked nested Git sources |

The seven class-5 targets are `amd-llvm`, `amd-comgr`, `hipcc`, `hipify`,
`rocm-cmake`, `rocm-half`, and `libhipcxx`. The first three share the LLVM
repository identity and retain their individual configuration and dependency
fingerprints. This patch does not narrow LLVM to independently reviewed subtrees.

Classes 1–2 retain the fetched-marker mechanism, including the existing
SIMDe/OpenBLAS wrapper fix. Classes 6–7 retain their existing handling. Windows,
WSL-specific wrappers, source overrides with a different repository layout, and
newly introduced monorepo targets need an explicit policy review. An unregistered
monorepo target fails configuration while this option is enabled.

## How it works

1. `cmake/therock_subproject.cmake` asks `cmake/therock_project_smrev.cmake` for a
   registered target's source identity during activation.
2. `build_tools/project_smrev_inputs.json` selects its source paths, optional
   common inputs, version queries, and environment variables.
3. `build_tools/generate_project_smrev.py` validates the actual source checkout,
   reads committed tree/blob IDs, and atomically writes
   `build/source-identities/.<target>.smrev`. Unchanged content keeps its mtime.
4. The existing fingerprint calculation uses that marker's contents as its
   `SOURCE` input. Configuration/input-file hashing and build/runtime/toolchain
   dependency handling continue through the existing CMake implementation.
   Existing artifact aggregation is unchanged.
5. A build prerequisite checks the configured identities again. Each managed
   subproject build and stage operation also checks its identity before recording
   its success stamp. Drift fails the operation and removes the owned marker;
   there is no repository-HEAD fallback for a registered target.

Markers live in the build tree to avoid modifying source repositories or their
source-preparation markers. The helper records the effective `FPRINT_SOURCE_DIR`
as `build/source-identities/<target>`, beside its generated marker. CMake reads
the validated marker directly: running the default reader there would inspect
the enclosing TheRock Git repository instead of the actual source repository.
The source used to configure or compile the project is not moved.

Git `HEAD` is used to locate committed objects and detect concurrent source
changes. Its commit ID is not inserted into this source identity. Tree IDs cover
tracked filenames, contents, symlink targets and executable modes. Dirty tracked
files, untracked files inside selected scopes, hidden index flags, missing required
inputs and mismatched nested checkouts are rejected. Ignored generated files are
not inputs; this assumes they are derived outputs, not undisclosed build inputs.

## Input examples

| Target/profile | Selected source inputs |
| --- | --- |
| `amdsmi` | `rocm-systems/projects/amdsmi`, `shared/cuid`, `shared/sha256` |
| `hip-clr`, `ocl-clr` | `rocm-systems/projects/clr`, `projects/hip`, `projects/hrr` |
| `rocBLAS` | `rocm-libraries/projects/rocblas`, `shared/tensile`, `shared/ctest` |
| `hipBLASLt` | `rocm-libraries/projects/hipblaslt`, `shared/origami`, `shared/rocroller`, `shared/stinkytofu`, `shared/mxdatagenerator`, `shared/ctest` |
| DNN providers | Their `rocm-libraries/dnn-providers/<project>` directory, `dnn-providers/cmake`, and `shared/ctest` |
| LLVM targets | The entire tracked `compiler/amd-llvm` tree |

Paths after the first entry in a row are relative to that row's repository.
The FFT projects' own `shared/` directories are inside their project trees and
are already covered. The repository-level `shared/ctest` is an additional input.

The catalog deliberately includes a conservative set of common root files and
CMake directories. Missing optional paths are represented explicitly, so adding
one changes the identity. Environment entries were collected from the reviewed
recipes and shared generators; some only apply to particular configurations.
They are a conservative set, not a claim that every field affects every build.
`null` means the variable is unset; unset, empty and a value are distinct.
HIPIFY's dynamically named CUDA sample variables are matched by a declared
pattern. No complete environment dump is stored. Paths to external tools or SDKs
do not identify their bytes; validating those installations remains part of the
toolchain/environment audit.

## Version and provenance policy

Reused artifacts retain their original producer's commit/hash/count/branch labels.
This implements the agreed producer-provenance policy; it does not promise that
a fresh build at another commit would have identical version-label bytes.
The patch never rewrites headers, Python version files, or restored artifacts.

Release-tag queries are evaluated according to the selected profile. AMD SMI uses
the highest version-sorted matching tag names. Projects using `git describe`
record the selected tag rather than the entire repository tag list. CLR's
project commit timestamp remains an input because it can determine HIP's numeric
patch version. History-dependent queries require a non-shallow checkout. Nested
repositories conservatively record their own tag refs and whether they are
populated, as well as their pinned commit.

Existing source-preparation markers, including LLVM's two-line repository/revision
marker, are preserved. Their separate uses for version generation remain intact.
Do not populate/update nested repositories concurrently with a build. If a build
initializes a nested checkout, the identity check requires reconfiguration before
that changed source state can be accepted.

## Known limits

This is an opt-in source-identity patch, not acceptance of every artifact as safe
for remote reuse. It does not connect restored artifacts to consumer builds.
The full Linux build and its output fingerprints still need validation in a
complete checkout. All 72 configured class-3/4/5 targets have policies; complete
coverage of every optional upstream feature is not established.

Two registered targets generate markers but intentionally have invalid final
fingerprints: `profiler-hub` can fetch SQL schemas from a floating external ref;
`mirage` can fetch external HotSwap compiler/runtime sources. Their normal builds
remain available. Cacheability requires representing those fetched inputs first.
Invalid fingerprints propagate through the existing dependency rules.

Build-path normalization is a separate change. These marker identities omit
checkout/build locations, but other CMake fingerprint inputs can still contain
absolute paths. Broad invalidation can also come from real shared inputs,
conservative source scopes, or remaining source routes outside classes 3–5.

## Apply and validate

Use the clean `users/erman-gurses/test-fingerprint-stability` checkout containing
the SIMDe/OpenBLAS fix. Do not apply this over the earlier experimental
`generate_project_smrev.py` implementation without reconciling it first.

```bash
git apply --check /path/to/therock-project-smrev-classes-3-5.patch
git apply /path/to/therock-project-smrev-classes-3-5.patch

python3 -m unittest discover \
  -s build_tools/tests -p 'project_smrev_test.py' -v
git diff --check
```

The tests use temporary repositories and builds. The CMake tests require CMake
3.25+, Ninja and Python 3.9+; the TheRock checkout itself may require a newer
CMake. Inspect the test summary for skipped tests. Git 2.31+ is required by the
CMake Git-directory query.

Review and commit the patch before the real configure experiment: source routes
outside these classes may still invalidate their fingerprints when TheRock's
tracked working tree is dirty. Keep source preparation complete and tracked
source checkouts clean. If a history-dependent query reports a shallow checkout,
fetch the required history in that repository before retrying.

Use the existing configuration and add the option; this does not start compilation:

```bash
cmake -S . -B build -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DTHEROCK_AMDGPU_FAMILIES=gfx110X-all \
  -DTHEROCK_PROJECT_SMREV=ON

python3 build_tools/generate_project_smrev.py \
  --check-list build/source-identities/requests.txt

cat build/source-identities/.amdsmi.smrev
cmake --build build --target therock-check-project-identities
```

Repeat configuration with identical inputs and compare markers and `.fprint`
files. In a separate disposable checkout, test a committed AMD SMI edit, an
unrelated project edit, and a `shared/cuid` edit. Check source markers separately
from dependency-propagated fingerprints. Build logs are required to establish
which projects actually compile. After the configure checks pass, run the desired
real build using `cmake --build build` and inspect its logs.

Disable this route with `-DTHEROCK_PROJECT_SMREV=OFF` and reconfigure. Existing
generated markers are ignored while disabled; no source marker is overwritten.
