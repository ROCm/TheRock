# Pinned rocSHMEM functional probe

This default-off diagnostic closes a coverage gap in build-profile I-005. In
TheRock run 41261635745, the original comprehensive rocSHMEM job reported success
with all 296 selected cases skipped. The manual family runner override took
precedence over the component's configured eight-GPU route. The installed wrapper
also searched PATH for `amd-smi` and `rocminfo`, while the generic component runner
added only library directories. Its explicit driver diagnostics found one GPU and
VMM support, but the wrapper reported zero GPUs and unavailable VMM.

The diagnostic uses the configured `linux-gfx942-8gpu-ossci-rocm` route and existing
digest-pinned image/security settings. It checks the actual HSA and HIP counts,
then runs the original sanity policy and original comprehensive rocSHMEM driver.
Sanity uses its original separate f7 OpenCL-runtime image and security options;
rocSHMEM keeps its original a13 base image. The actual run40 sanity component is
retained as a regression fixture. These contexts must not be combined: the base
image lacks sanity's system OpenCL loader, as failed probe41272130530 showed.
Only the test subprocess gains the installed SDK's `build/bin` on PATH. Existing
skip conditions, backends, tests, tolerances, and CTest commands remain intact.
A successful result additionally requires nonzero executed tests and all 296
original selected cases accounted for. Hardware-dependent skips remain visible.

The producer is fixed to run 41261635745/source 60faab8d3. Native binaries are
checked against independently downloaded published archives and exact per-image
inventories covering 31 architectures, including gfx1250-strict. The installed
functional binary's SHA differs from its build-tree record; the contract uses
the actual published installed bytes, without normalization. Only executed
gfx942 behavior gains runtime evidence; the other images retain static checks.
Before/after native and resolved-loader file hashes must agree. This is selected
file/provider proof, not a complete SDK or all-architecture runtime proof.

No source build, shared-cache write, release publication, CDash submission, or
Quartz notification runs. Evidence uploads go only to GitHub Actions artifacts.
The early HIP count query can warm runtime state, and metadata hashing adds work:
none of this diagnostic's timings establish a performance gain. The frozen
full baseline/candidate workload is unchanged.

Dispatch the existing Multi-Arch CI entry on the isolated runtime branch with
`profile_rocshmem_runtime_probe=true` and `profile_rocshmem_expected_sha` equal to
the reviewed frozen diagnostic commit. All other inputs use their defaults.
Repository, actor, branch, workflow, checkout SHA, producer, and runner guards
fail closed. Normal setup/summary/build paths are excluded only in this mode.
