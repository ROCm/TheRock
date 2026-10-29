# ROCm/TheRock selected DASan artifact tests

Owner session: `01a17bc3-5a47-7832-8012-6a7e5551f090`.

The workflow consumes exactly the ten successful TheRock producers pinned in
`producer-manifest.json`, verifying producer, run, commit, and SHA256 before
extracting. It preflights all requested stages on CPU and runs only test suites
present in the selected packages; absent suites remain failed coverage entries.
RDC/emulation availability does not count as tests. It never substitutes SDKs or
libraries from other builds.

Test sources are pinned to TheRock `52dfa419e245242f79e25ad1bc10011d9c3c1295`.
The FFT multi-GPU runner is vendored from `eee107e60047531cc765664faa20cb4cf2b5d1bd`
and its source hash is in `matrix.json`. RCCL, rocSHMEM, hipFFT/rocFFT multi-GPU,
and compiler multi-device lit27 use eight-gfx950 runners; other GPU suites use one.

`BUILD_VARIANT=release` keeps the stock test-selection logic on its full-suite path;
`SANITIZER_MODE=dasan` records the active sanitizer. The selected shared DASan
runtime is preloaded for component tests and compile-time tests use
`-fsanitize=daddress -shared-libsan`. Compiler runtime validation retains its
static/shared load-order checks without a forced preload. No tests may pass with
an empty selection. CI artifacts retain logs and source provenance for 30 days.

## Repairs after run 40809027703

The launcher preserves Python's library directory and inherited library paths
after the selected DASan and SDK paths. Artifact transfers retry up to three
times, resume only on a matching HTTP range response, and retain byte counts and
observed hashes without signed URLs. SHA256 pins remain mandatory; a complete
ZIP with a different digest fails. Artifact installation has a 180-minute limit.

Coverage distinguishes setup failures, incomplete producer artifacts, missing
test payloads, and components without an executable suite. Missing payloads
continue to fail the coverage gate. The ten pinned artifacts are unchanged.
Future math, comm, storage, profiler, and RDC producers enable test building;
profiler builds audit required manifests and installed CTest payloads before
packaging. New producer artifacts require explicit provenance and selection.

Validation: five transfer regression tests, Python compilation, workflow YAML
parsing, launcher path precedence checks, profiler payload audit checks, and
replay of the original plan (81 requested, 12 runnable, 69 unavailable).
