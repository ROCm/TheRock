# Leading-build-stack source support (not enabled)

`build_tools/normal_leading_stack.py` supplies source and output gates for the
explicit normal-CI selector `leading-build-stack`. The workflow wires it together
with I-029 before the first controlled normal cohort. `none` remains the default,
and `rocfft-private-batch` remains a separately selectable RTC-only mode.
Jobs/dependencies, stock tests, GPU policy, cache regimes, CI-only destinations
and Quartz guards remain unchanged. No normal cohort is launched by these files;
full-COMM repetition/full-DEB and normal end-to-end acceptance remain separate
requirements. This support code claims no new speedup or GPU correctness result.

## COMM source contract

Run apply only after stock source fetch/patching and before common identity
capture/top-level configure. The original TheRock `comm-libs/CMakeLists.txt` and
all three affected rocSHMEM CMake files must match their pinned product LF SHA256s.
The rocm-systems base is `a10dca087065a82f15ea89dc7d6fdd3c6bb2c246`.
The actual HEAD may include the stock ROCR patch: require that base as an ancestor,
record the actual HEAD/branch/parents/shallow state, and verify the unchanged
`.rocm-systems.smrev` against the original stock patch-series fingerprint.

Apply the identical reviewed three-file rocSHMEM patch in BOTH arms. Apply the
same ROOT cache-option/pass-through patch with default1 in BOTH arms. Candidate
then changes only that single ROOT default field from1 to2. Neither arm advances
source HEAD/history/branch or rewrites the stock `.smrev`; there is no new workflow
CMake argument splicing, global HIP flag, architecture filter or source-version
normalization. Record every applied target/source hash and patch payload hash.
Only ROOT's default field differs; generated compiler commands must be compared
by the normal input/cache gates rather than assumed equivalent.

The helper accepts only the exact reviewed LF payloads:

| Payload under `patches/build-profile/normal-stack` | SHA256                                                             |
| -------------------------------------------------- | ------------------------------------------------------------------ |
| `therock-native-link-control.patch`                | `e9c8de9b3f3804fe88186a4b7453bf4fa43c6388d8524171347a65d1361e621a` |
| `rocshmem-native-link-control.patch`               | `8b21193ef9f4287f522b8bd8648e7487a183165422460e6ffa8eb84e02676746` |
| `bounded-elf-probe.patch`                          | `777edbf4581659833190bc047c4e6314ba25381b7bf72f0bfd07b7a1a05b98b1` |

COMM scope is the generic Linux `comm-libs` job with an empty family argument;
the generated rocSHMEM target set must remain all31 original architecture IDs,
including `gfx1250-strict`. Source apply is case-specific and current-run-specific;
reuse/adoption, a changed selected original file, an unexpected payload or a
changed stock source marker fails before acceptance.

## Native ELF source contract

Apply before native-package source identity capture/build, in the Linux DEB or
RPM job. Baseline keeps the original module; candidate applies exactly the
validated whole-file-to-four-byte probe patch to
`build_tools/packaging/linux/runpath_to_rpath.py`. Record both original and
resulting source hashes, the actual ROOT Git identity and the patch hash.
No worker, compressor, package selection, timestamp or artifact input changes
are introduced here. A shorter actual conversion is already measured separately;
its effect on complete package assembly must still pass the active full-DEB
comparison and later normal end-to-end checks.

## Calls and acceptance

The normal workflow supplies `NORMAL_VALIDATION=true`, CI release destinations,
TheRock Actions/dispatch/current run metadata and the matching dispatch product
and case. Default-off invocation needs no event or source mutation. Enabled calls
reject foreign repositories, local execution, other product/case inputs and a
changed global HIP parallel-job setting. Run the same case throughout a job:

```bash
python build_tools/normal_leading_stack.py apply-comm --enable --case baseline --output build/logs/normal-validation/leading-stack-comm
# Stock top-level configure and COMPLETE normal COMM build/checks.
python build_tools/normal_leading_stack.py validate-comm --enable --case baseline --output build/logs/normal-validation/leading-stack-comm

python build_tools/normal_leading_stack.py apply-elf --enable --case baseline --output output/normal-validation/leading-stack-elf
# Stock COMPLETE native package build/checks.
python build_tools/normal_leading_stack.py validate-elf --enable --case baseline --output output/normal-validation/leading-stack-elf
```

Candidate changes only the case argument. The main preconfigure hook applies
COMM source only in its generic comm-libs job. Native apply runs before native
identity/build capture, with required validation immediately after successful
package build. RTC preparation/validation is independently scoped to gfx103X
math. Other jobs do not attempt to validate absent product receipts. Each validation requires the current run/attempt/case, unchanged
source identities and payloads. COMM additionally requires ROOT and nested native
widths1/2, unchanged global HIP width1, successful generated parallel capability
for candidate, and all31 original targets. Both expanded native final-link argv
must have the expected width and no compile/dry-run/response-file ambiguity.
Each target also needs an executed Ninja-log record and a regular ELF output;
retain original command text/argv, Ninja records and native file hashes.
Compilation edges may not gain the native link override. ELF validation is an
unchanged-source gate only, not package-content acceptance.

Workflow gates require successful build and CURRENT validation exit before
acceptance, keep all stock checks, and always retain partial source/product logs.
Compare source/SDK identities, actual cache keys/outcomes, all output inventories,
actual allocations and enabled correctness checks between arms. Do not normalize
binaries, recover missing old bytes by assertion, skip tests or infer GPU numerical
correctness from static hashes. Graph/source/file-hash gates do not replace the
stock tests or full parity review. Post-build hashing/inspection overhead stays
outside the stage build timer and inside observed job elapsed.

## Local host/source evidence

```bash
python -W error::ResourceWarning -m unittest build_tools.tests.normal_leading_stack_test -v
```

The committed JSON source fixture preserves exact original file bytes, including
upstream trailing newlines, without applying repository formatters to copied
source code. Fixtures reuse the exact pinned three rocSHMEM CMake files,
original ROOT/ELF sources, and both actual expanded native link commands with
original SHA256 provenance. Git fixtures add an explicitly inert unrelated stock
patch history to exercise ancestor/HEAD/marker preservation; they are not the
product base commit. Only the internal fixture API can substitute that base;
the CLI always requires a10. Inert Ninja graphs query original commands without
executing a compiler, and tiny marker bytes stand in for native outputs solely
to test hash/format gates. A host CMake generation fixture uses the actual patched
tools/tests files: all37 compile commands remain byte-identical; only the two
final link edges gain one `--offload-jobs=2`, retaining31 architectures. Invalid,
unsupported and conflicting settings fail generation. No target is compiled,
linked, installed or executed by that generation fixture. ELF tool decisions use
mocks and do not substitute for the completed actual-conversion evidence.
