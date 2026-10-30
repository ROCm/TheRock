# Experimental DASan build

## Fresh normalized full-stack build

### Retry only verification of an existing delivery

When all build and packaging stages passed but downloading or verifying the
SDK failed, dispatch only the corrected verification path:

```bash
gh workflow run multi_arch_ci_asan.yml --repo ROCm/TheRock \
  --ref users/safaizal/dasan-stack-01a189df -f dasan_validate_run_id=41073351087
```

This runs no build or packaging stages. It selects the existing full-tests SDK
from that original run's combined delivery ZIP using bounded HTTP Range reads,
checks its ZIP CRC and original master SHA-256, and runs all 27 unchanged tests
with the original lit bundle. The runtime-only SDK is not downloaded. Transfer
logs and provenance distinguish the original build run/SHA from this new
verification run/tools SHA. The historical delivery has no unpacked-size field;
disk preflight explicitly reserves 300 GiB for extraction plus the selected
archive and 15 GiB for scratch space. No test or debug payload is omitted.

### Build a new stack from source

Dispatch the full-stack path in **ROCm/TheRock**, using the current branch:

```bash
gh workflow run multi_arch_ci_asan.yml --repo ROCm/TheRock \
  --ref users/safaizal/dasan-stack-01a189df -f dasan_full_stack=true
```

This uses the normalized stage targets and artifact tooling from the
[rockrel reference run](https://github.com/ROCm/rockrel/actions/runs/40793100440),
but the workflow run, source checkout, artifact ownership, and downloads all
belong to `ROCm/TheRock`. Native build stages use
`aws-linux-scale-rocm-large`, matching the actual reference jobs. WSL uses
the reference runner group `ROCR-Runtime-libhsakmt-WSL`.

The compiler/runtime, emulation, runtime tests, math/ML, communication,
storage, debug tools, data-center tools, profiler applications, computer
vision, media, and WSL stages all build from source. Profiler core libraries
belong to compiler/runtime; ML libraries and hipDNN providers belong to the
normalized math stage. No earlier run's binaries or compilation cache are
reused. Stage dependencies are transferred only within the new run.
Emulation waits for this run's compiler so its C++ code uses the DASan compiler.

The `linux-release-dasan` preset applies device address sanitizer flags,
with explicit bootstrap-tool exclusions for LLVM, HIPIFY, and hipcc.
It overrides the package preset's `THEROCK_MINIMAL_DEBUG_INFO` to `OFF`:
the current minimal-debug toolchain generation defines configuration flags
before CMake supplies optimized defaults, dropping `-O2` and breaking AMD SMI's
`_FORTIFY_SOURCE` checks. Normal optimized defaults and explicit debug flags
are retained in the DASan path.
The WSL bridge has no GPU device code and uses its standard platform-support
package preset. Per-stage provenance records source pins, patches, configuration,
and instrumentation evidence; Rust and bootstrap/system dependencies are
identified separately from instrumented C/C++ code.

The final package job checks complete artifact coverage, assembles standard
install-prefix tarballs, checks the host/compiler and packaged RCCL device
code, and publishes each `.tar.gz` as a separate raw GitHub Actions artifact,
named exactly as that archive. The small `dasan-stack-sdk-metadata-gfx950`
artifact contains the master checksums, provenance, and download instructions.
Coverage requires distributable archives, including OpenMPI. Six topology
nodes do not produce separate Linux archives: Windows-only `core-hipinfo`,
build dependencies `fmt`, `grpc`, and `spdlog`, and `sysdeps-libmnl`/`sysdeps-libnl`
whose payloads belong to `sysdeps`. The source manifest records these exceptions;
missing distributable archives still fail packaging.
The delivery includes the ordinary SDK archive, a complete archive including
tests, SHA-256 checksums, provenance, and download instructions. Stage binaries
also remain in the standard run-scoped artifact store; stage logs alone are
not treated as binary artifacts.

The following job independently downloads only the full-tests archive and the
unchanged lit bundle. It verifies every metadata checksum and the exact selected
archive SHA-256 entry, then runs all 27 DASan host/HIP/OpenMP
tests on a multi-GPU gfx950 runner. Build success, package validation, and GPU
validation are distinct results; inspect all three before relying on the SDK.

The raw single-file transport avoids downloading a ZIP containing both SDK
variants. It checks space for the selected archive, its extracted payload, and
15 GiB of scratch space before transferring. Downloads report progress every
30 seconds, resume partial transfers only for the same artifact identity, and
fail on incorrect byte ranges, sizes, or checksums. The transfer budget is
150 minutes within a 240-minute validation job. The complete runtime and tests
SDKs remain separate downloads; the runtime archive is not needed by GPU validation.

Applications using the fully sanitized HIP/ROCr libraries must link the shared
host runtime with `-fsanitize=daddress -shared-libsan` and make the compiler's
shared runtime directory discoverable at execution. The delivery README gives
the environment setup. Mixing a static host DASan runtime with these sanitized
shared libraries produces recursive HSA interception and is unsupported.
The compiler stage checks shared HIP/ROCr initialization before library builds.
GPU validation uses shared host linkage, including GPU test host helpers;
the three original host tests and standalone static/shared initialization checks
retain their original linkage coverage. All 27 source tests and assertions are
retained, but this full SDK's GPU tests do not claim static-host-runtime coverage.

## ROCm systems integration rebuild

The current build pins `ROCm/rocm-systems` branch `users/searlmc1/dasan` at
`4b6ee66563c1f331d9d3568c73b3f41c0ed5a5f2`, including Joseph's
[ROCr changes](https://github.com/ROCm/rocm-systems/pull/12975) and Larry's
[RCCL changes](https://github.com/ROCm/rocm-systems/pull/14544).
The historical validation below describes the previous source combination.

RCCL uses `BUILD_DEVICE_ADDRESS_SANITIZER=ON` and `ENABLE_DEVICE_LINKER=ON`
under `THEROCK_SANITIZER=DASAN`. Its previous compatibility patch disabling
the device linker is removed so the per-translation-unit DASan runtime path
is used. The packaged RCCL device-code audit remains enabled.

The targeted update rebuilds and validates the compiler/runtime, then builds
`comm-libs` against that installation while reusing the original math artifacts
from run `40545400942`. The other component stages do not need rebuilding.
The compiler/runtime workflow also rebuilds LLVM; its compiler-rt DASan runtime
needs a loader capability-query compatibility patch for this ROCr revision.
ROCr reports loader minor version zero through the general major-version query
while exposing loader 1.4 through the exact-version query. DASan must use the
exact-version query to discover code-object placement, or global-memory
instrumentation is ineffective. The unchanged GPU lit tests gate this update.
The container image and GPU targets retain the baseline configuration.

```bash
gh workflow run multi_arch_ci_asan.yml --repo ROCm/TheRock \
  --ref users/safaizal/dasan-systems-01a189df \
  -f dasan_systems_update=true -f dasan_dependencies_run_id=40545400942
```

The original [compiler/runtime run](https://github.com/ROCm/TheRock/actions/runs/40443090683)
and [math run](https://github.com/ROCm/TheRock/actions/runs/40545400942) took
12m48s and 2h52m34s respectively for their build steps. The original
[comm-libs run](https://github.com/ROCm/TheRock/actions/runs/40586632932)
took 2h41m32s for its build step; its Ninja log records 2h32m45.3s for
RCCL's build stamp. The other component stages are runtime-tests, storage-libs,
profiler-apps, media-libs, emulation, dctools-core, and cv-libs.
These are comparison baselines; the RCCL improvement must be measured by
the new run rather than inferred from workflow success.

## Validated one-off build (2026-10-27 UTC)

The pinned source combination below, with both DASan compatibility patches and
HIP frame pointers, produced a portable installation in the pinned CI image.
The packaged installation passed all 27 unchanged DASan lit tests on an
eight-GPU MI350X (`gfx950`) machine: 3 host, 21 HIP, and 3 OpenMP tests. This
includes `heap-overflow`, `host-overflow`, `quarantine`, and `multi-device`,
using the default unwinder; `fast_unwind_on_malloc=0` was not used.

Both `gfx942` and `gfx950` passed instrumented compilation and linking with the
installed HIP headers and device libraries. GPU execution was validated only
on `gfx950`. Static and shared host DASan initialization also passed.

The accepted flag is `-fsanitize=daddress`. The compiler rejects
`-fsanitize=dasan` at this revision.

The archive `therock-dasan-gfx942-gfx950.tar.xz` has SHA-256
`db59aa600917080ff555204b0485c57c66d0afe7acb46deed16a4ce903168ec2`.
It contains the merged installation, relocatable lit sources/configuration,
validation runner, source/configuration manifests, and both LLVM patches.

## Configuration

This one-off configuration builds LLVM, compiler-rt, HIPIFY, ROCr, and HIP in
the same manylinux container as the Linux `compiler-runtime` CI stage. It uses
the `THEROCK_ENABLE_DASAN` CMake option and excludes math libraries, profilers,
media libraries, and other unrelated components.

The pinned image is
`ghcr.io/rocm/therock_build_manylinux_x86_64@sha256:cf4f6d9909056906e4a1c0bc4c245658a0af4c6a6174082a86adfa2d518b7aeb`,
matching `.github/workflows/multi_arch_build_portable_linux_artifacts.yml`.

| Source                     | Requested branch/revision | Pinned commit                              |
| -------------------------- | ------------------------- | ------------------------------------------ |
| ROCm/llvm-project          | `amd/dev/msearles/dasan`  | `b07aa864f5b2bf648e6a61e2902cbfde83192481` |
| ROCm/HIPIFY                | `d364ef4`                 | `d364ef407b2c3e20c6ceb3575bd852f0c7cebb8c` |
| ROCm/SPIRV-LLVM-Translator | `5817038`                 | `5817038657a5d3166f753389ff7f4740bd8d7c16` |
| jhuber6/rocm-systems       | `loader_va`               | `f535db62a0dc73f2d56d4b93d9bbd4f3155e113b` |

## Build and resume

### GitHub Actions

The experimental branch adds an `enable_dasan` manual-dispatch input to
`multi_arch_ci_asan.yml`. It selects a dedicated DASan compiler/runtime path
using the same pinned CI container. The normal ASan path remains the default.
The compiler/runtime mode builds both `gfx942` and `gfx950`. The math mode
below reuses its installation and targets `gfx950`.

```bash
gh workflow run multi_arch_ci_asan.yml --repo ROCm/TheRock \
  --ref users/safaizal/dasan-ci-01a17b3a -f enable_dasan=true
```

The CPU job builds the pinned sources, validates the installed host runtimes
and both compile/link targets, and uploads the installation, source manifests,
logs, and unchanged lit source bundle as GitHub Actions artifacts. Separate
multi-GPU jobs run all 27 DASan tests on `gfx942` and `gfx950`, including OpenMP
and the multi-device test. The test jobs use the installed artifacts and do
not set `fast_unwind_on_malloc=0`. Each architecture uploads its own results;
a failed or unsupported required test fails that job. Artifacts are retained
for 14 days. This workflow does not publish a ROCm release.

### Math libraries using an existing DASan installation

Use the same branch and provide the successful compiler/runtime run as the
baseline. This downloads `dasan-installation-gfx942-gfx950` from that run and
builds the math artifact group for `gfx950` with `THEROCK_SANITIZER=DASAN`.
The compiler spelling is `-fsanitize=daddress`, including HIP language
compilation. GPU targets retain their normal spelling without `:xnack+`.

```bash
gh workflow run multi_arch_ci_asan.yml --repo ROCm/TheRock \
  --ref users/safaizal/dasan-ci-fix-01a17b04 \
  -f enable_dasan=true -f dasan_math_libraries=true \
  -f baseline_run_id=40443090683
```

The reused compiler and runtime retain the instrumentation of the baseline
run. Math libraries are newly instrumented. The math mode excludes ML libraries
and uploads build and configuration logs even on failure to
`s3://therock-ci-artifacts/<run-id>-linux/logs/math-libs/gfx950/`.

### Local or remote build

Use a dedicated Linux checkout containing the DASan enablement changes. Source
files for the top-level enablement must be committed before launching, including
new CMake hooks and scripts: the manifest's diff fingerprint only covers tracked
files. Commit the experimental gitlink pins after preparation as well when
preserving a complete reproducible checkout.

Source preparation updates its four gitlink pins and its local rocm-systems submodule
URL, fetches only the `compiler-runtime` source set, and applies TheRock's
normal compatibility patches. It refuses already initialized experimental
submodules because `fetch_sources.py` resets their contents.

The launcher then applies the local `patches/dasan/llvm-project` compatibility
patches, including on an already prepared checkout. The supplied LLVM branch
uses an obsolete ROCr executable-allocation enum and bit value; the selected
ROCr revision expects `HSA_AMD_MEMORY_POOL_EXECUTABLE_FLAG` with value `1 << 2`.
The first patch updates both the call site and fallback enum definition. The
second requests executable permission explicitly when mapping the reserved
address space; allocating executable-capable memory alone does not make the
KFD mapping executable.

The third patch makes `hsa-missing.c` portable to the CI image's older glibc:
it enables `_GNU_SOURCE` before including `dlfcn.h` and explicitly links
`libdl`. Both static and shared sanitizer executions and their FileCheck
expectations remain intact. The bundle copies and hashes these patched test
sources without further edits.

The launcher detects an already applied ordered prefix and applies only the
remaining patches with TheRock's normal `git am` identity and fixed committer
date. It refreshes LLVM's `.smrev` using the original source pin and the digest
of the complete ordered patch series, preserving ROCr's patch metadata. This
allows a new compatibility patch to be added without resetting sources or
reapplying earlier patches. Dirty compiler sources, conflicts, or patches
applied out of order stop the build for inspection.

```bash
python3 build_tools/dasan_build.py \
  --output-dir /absolute/path/to/unique-dasan-output \
  --prepare-sources --sudo \
  --gpu-targets 'gfx942;gfx950' --jobs 192 --link-jobs 8
```

The job counts above suit the 384-thread, 1 TiB build host used for this
experiment. Reduce them on smaller hosts. Omit `--sudo` if Docker is accessible
directly. `--gpu-families gfx94X-dcgpu` selects a family instead of individual
targets. `--prepare-only` fetches and records sources without starting a build.

For subsequent iterations, omit `--prepare-sources`. Existing CMake build
directories and ccache are reused. A manually prepared checkout is also
supported: each experimental source must descend from the specified pin and
have no tracked uncommitted changes. Commit local source fixes before invoking
the script so `sources.json` identifies them. Append additional CMake options
after `--` when necessary.

```bash
python3 build_tools/dasan_build.py \
  --output-dir /absolute/path/to/unique-dasan-output --sudo
```

The script writes `sources.json`, `configure.json`, `status.json`, `build.log`,
and `docker.log` under the output directory. Logs append across iterations.
The sources manifest records effective commits after compatibility patches,
the top-level commit and tracked diff fingerprint, and the container digest.
`status.json` distinguishes preparation, completed build, and failure; build
completion does not imply runtime test success.

The launcher enables LLVM/compiler-rt tests explicitly while disabling
`THEROCK_BUILD_TESTING` and `THEROCK_BUILD_COMGR_TESTS`. The selected artifact
features exclude HIP's general test suite, ROCr runtime tests, KFD tests, and
OpenCL. This keeps the validation scope focused on the experimental compiler.

Artifacts are under `build/artifacts/`. The merged installation is
`build/dist/rocm/`, including `lib/llvm/bin/clang`, the Clang resource directory,
`libamdhip64`, and `libhsa-runtime64`. Compiler-rt host and AMDGPU sanitizer
runtime files must be checked in both staged artifacts and the merged tree.

## Validation boundaries

At this pinned compiler revision, the sanitizer's command-line spelling is
`-fsanitize=daddress`; the LLVM team's requested `-fsanitize=dasan` spelling
must be tested and reported separately. Do not interpret accepting a spelling
as evidence that host/device instrumentation or runtime linking works.

The pinned compiler uses the canonical runtime target `amdgpu-amd-amdhsa`, as
in the LLVM team's recipe. Its older `amdgcn-amd-amdhsa` spelling remains a
legacy alias. TheRock's option integrates compiler-rt into its existing host
and AMDGPU runtime configuration with the canonical spelling.

The build host may have no AMD GPU, `/dev/kfd`, or `/dev/dri`. It can produce
gfx942 and gfx950 binaries and run compiler checks, but DASan execution tests
require a GPU machine with a compatible driver. The Docker build does not
request GPU devices. Record compile, link, artifact, and GPU runtime checks
separately; do not report skipped or unsupported tests as passed.

After the build, run the focused validator inside the same container, with
the source and output directories mounted at the build-time locations. Set
`BUILD` to the build directory and `HOST_DASAN_SUITE` to the configured host
compiler-rt `test/dasan` directory. Locate its `lit.site.cfg.py` if necessary:

```bash
BUILD=/therock/output/build
find "$BUILD/compiler/amd-llvm/build" -path '*/test/dasan/lit.site.cfg.py'
# Select the host runtime's test/dasan directory from the results above.
HOST_DASAN_SUITE=/path/to/host/compiler-rt/test/dasan
python /therock/src/test_tools/dasan_smoke.py \
  --rocm-path "$BUILD/dist/rocm" \
  --llvm-lit "$BUILD/compiler/amd-llvm/build/bin/llvm-lit" \
  --lit-suite "$HOST_DASAN_SUITE" \
  --output-dir /therock/output/validation
```

The validator audits installed sanitizer resources, checks driver spellings,
requires instrumented IR and successful links for both gfx942 and gfx950, and
requires all three host lit tests to pass. It writes `summary.json`, detailed
logs, and lit JSON reports to the validation directory. GPU tests are explicitly
reported as unrun by default. On a GPU machine, pass `--gpu-arch gfx942` or
`--gpu-arch gfx950` to require the 21 HIP tests; only the multi-device test may
be unsupported unless `--require-multi-device` is also passed. To exercise all
27 source tests on a machine with eight gfx950 GPUs, append:

```bash
--gpu-arch gfx950 --gpu-device-count 8 --require-multi-device --openmp
```

Set the architecture and device count to the actual validation hardware.
`--openmp` enables the three OpenMP tests and uses a generated lit site overlay
to supply runtime capabilities that a CPU-only build could not discover.
Without that flag, OpenMP tests remain explicitly unrun. The validator also
executes installed static and shared host initialization smoke programs.

The LLVM team's initial observation was approximately 28 lit tests; the pinned
source contains 27 (3 host, 21 HIP, and 3 OpenMP). They reported three
remaining failures (`heap-overflow`, `host-overflow`, and `quarantine`) after
the ROCr loader patch. Their proposed explanation is missing frame pointers
in the HIP runtime. Reproduce the default tests first and preserve results.
Then compare a HIP build with frame pointers and/or
`DASAN_OPTIONS=verbosity=1:fast_unwind_on_malloc=0`, recording the exact changed
options. The lit configuration overwrites `DASAN_OPTIONS` with `verbosity=1`,
so an outer shell override does not change unmodified lit runs. Rerun the
lit-emitted failing binary commands manually for this diagnostic comparison.
The slow-unwinder workaround is diagnostic evidence, not proof that the
default fast-unwinder configuration works.

Expect iterations at source compatibility, compiler-rt installation/packaging,
and GPU validation boundaries. Preserve source pins, logs, and caches so each
iteration can target the failing component instead of repeating a full build.

## Additional component stages in cloud CI

Dispatch `multi_arch_ci_asan.yml` on the DASan branch with
`dasan_component_stage` and a `baseline_run_id` providing
`dasan-installation-gfx942-gfx950`. Supported stages are `emulation`,
`runtime-tests`, `comm-libs`, `cv-libs`, `storage-libs`, `debug-tools`,
`dctools-core`, `profiler-apps`, `media-libs`, and `hipdnn`.
For example:

```bash
gh workflow run multi_arch_ci_asan.yml --repo ROCm/TheRock \
  --ref users/safaizal/dasan-ci-fix-01a17b04 \
  -f dasan_component_stage=comm-libs -f baseline_run_id=40443090683 \
  -f dasan_dependencies_run_id=40545400942
```

Each stage runs on `aws-linux-scale-rocm-large` in the pinned manylinux image,
imports the compiler/runtime SDK, and builds missing dependencies for gfx950
with `THEROCK_SANITIZER=DASAN` (`-fsanitize=daddress`). The SDK probe checks
host and device links and instrumented device IR before the stage starts.
The optional `dasan_dependencies_run_id` reuses available dependency artifacts
from a successful DASan math run. Imports are restricted to the selected stage's
dependencies; the requested components are built from source.
These jobs build and package components; they do not execute GPU tests.
The `runtime-tests` stage compiles its test binaries.

The job uploads merged and split installation archives as
`dasan-<stage>-gfx950`, with diagnostics in `dasan-<stage>-logs-gfx950`.
S3 logs are under
`<run-id>-linux/logs/<stage>/gfx950/` in `therock-ci-artifacts`.
Periodic uploads obtain fresh short-lived credentials through the job's
GitHub OIDC identity, and a final upload runs after success or failure.
Source revisions, configuration, SDK probe results, sanitizer command evidence,
and final status accompany the build log.
