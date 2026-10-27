# DASan non-math component CI

Dispatch `multi_arch_ci_asan.yml` with `enable_dasan_components=true` on the
component CI branch. This invokes `multi_arch_ci_dasan_components.yml` and does
not run the compiler-only DASan build or the ordinary host-ASan pipeline.

The baseline is the successful ROCm/TheRock run **40443090683**, revision
`91f84d411a32b0071c360360de9ac99897aae997`. Its compiler/runtime installation and
unchanged DASan lit bundle are downloaded through the authenticated Actions
artifact API. The build checks the baseline run result, embedded run ID, and
installation archive SHA-256 before use:

```text
78fa7a7911d09917532396ffdc8fbed0f35a761d9d1cde9f3bfcebf3510e09d1
```

The baseline contains a merged installation rather than bootstrap artifacts.
`build_tools/dasan_components/baseline-layout.json` restores stage ownership using
metadata from the matching pinned local build. It contains no local binaries.
Every imported payload comes from the CI archive, and every imported file must
match exactly one ownership rule. Prebuilt markers prevent source compilation
of those components. The small auxiliary metadata overlay is regenerated for
the new run. After building, all original payload files and symlinks are checked
against the baseline, and generation of an LLVM build is an error.

The build uses the same pinned manylinux Docker image, LLVM/HIPIFY/SPIRV and
`loader_va` source revisions. It adds the locally validated rocJITsu, Mirage,
profiler SDK and RDC fixes from `patches/dasan/components/`.

Included groups are the remaining common compiler/runtime components,
emulation, runtime tests, computer vision, storage, debug tools, data-center
tools, media, communications, profiler applications, and standalone hipDNN.
Math libraries, their dependent ML providers, and Windows/WSL components are
excluded. Both gfx942 and gfx950 are compiled. Components retain their normal
build flags; this is a DASan-capable SDK, not blanket sanitizer instrumentation.
The pinned compiler's flag is `-fsanitize=daddress`.

Packaging checks and rocJITsu/Mirage CPU tests run in the build job. Six gfx950
jobs execute installed runtime/DASan, communications, library, profiler SDK,
profiler-systems, and profiler-compute tests. Tests retain runner GPU isolation
and count devices through the selected HIP runtime. Component Python
requirements finish installing before CMake configures tests. There is no
slow-unwinder override. Capability skips remain visible in the reports.

The workflow uploads the expanded installation, its SHA-256, source and patch
provenance, baseline reuse audit, build logs, and test reports. Test jobs run with
`fail-fast: false`, so independent failures retain their evidence.

This one-off workflow deliberately pins one baseline. Once its Actions artifacts
expire, it fails rather than silently rebuilding or substituting local binaries.
Changing the baseline requires validating a new checksum and stage layout.
