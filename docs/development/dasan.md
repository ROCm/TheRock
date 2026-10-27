# Experimental DASan build

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
