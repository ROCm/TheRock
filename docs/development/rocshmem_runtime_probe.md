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
skip conditions, backends, tests, tolerances, and original executable arguments remain intact.
A successful result additionally requires nonzero executed tests and all 296
original selected cases accounted for. Hardware-dependent skips remain visible.

The corrected context probe41272939789 executed279cases successfully, skipped7,
and failed10 while allocating the first symmetric-heap buffer. Its ten original
1GiB-volume commands inherit loop10/batch0: pinned tester_arguments.cpp leaves
these defaults unchanged, Tester resolves batch10, and the wave/workgroup
geometry factors cancel to10GiB per local/remote buffer. Two buffers require20GiB
payload per rank; the fixed6GiB heap cannot satisfy even the first buffer.
The stock systems patch changes only ROCR flag.h, and the common control patch
changes only three CMake files. The inspected functional C++ sources remain at
the pinned a10 base; no whole functional source-tree hash was captured in CI.

This isolated capacity discriminator changes exactly ten generated add_test heap
environment fields6GiB->24GiB. It binds the complete original installed CTest file
SHA, retains original/modified copies and hashes, and rejects any other graph
change. Original binaries, arguments, loop counts, volume, test properties and
the other286 selected cases stay unchanged. The original296 case identities are
checked individually: all279 previously passing and ten allocation-failing cases
must now pass; only the seven originally skipped cases may still skip. Their
recorded original outcomes and complete original installed CTest graph are fixtures.

All ten affected commands use SLR NP2. Pinned test_driver.cpp169–186 selects
hipSetDevice(rank); slr.hpp forks ranks0/1. Original CTest runs --parallel1, so
the affected cases use one rank per logical GPU0/1 without test overlap. A separate
HIP child queries every one of eight devices after cleanup and immediately before
the original test command, requiring at least32GiB free each:24GiB heap plus8GiB
outside-heap margin. The child exits before the original SLR program forks.
Free/total memory and return codes are retained before/after. These are
point-in-time observations; other processes can allocate later, and the4GiB
within-heap margin above20GiB payload is a hypothesis to test. No model-only
capacity assumption, lower work volume, tolerance waiver or performance claim is made.
The allowlisted ambient heap-allocator setting is recorded and must be unset/empty.
Pinned default_allocator.hpp then selects HIPAllocatorFinegrained, whose device
allocation uses hipExtMallocWithFlags/hipDeviceMallocFinegrained. No allocator or
backend is switched; the four original VMM tests retain their own original override.

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
