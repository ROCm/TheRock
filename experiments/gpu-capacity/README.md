# Read-only gfx950 allocation diagnostic

Prepared for root review; **not dispatched by this task**. These files request the existing ordinary `linux-gfx950-1gpu-ccs-ossci-rocm` label, never the reserved ASan sandbox. They perform no build, SDK install, GPU compute, numerical testing, cleanup, device configuration, release or deployment. Only a small JSON/log artifact is uploaded.

The source-supported manual allocation route is documented in [the runner investigation](../../evidence/full-validation/runner-availability-findings.md). At frozen TheRock HEAD `5b7cb88373362ab80dc0f19605915e534f4158aa`, `test_artifacts.yml` honors an explicit nonempty `test_runs_on` for a human manual dispatch. `fetch_test_configurations.py` carries that input and falls back to ordinary GPU labels when its ASan sandbox entry is empty. `test_component.yml` applies the usual sanitizer test environment independently of pool selection. These source facts justify checking the ordinary manual route without bypassing the reserved sandbox policy; they do not establish successful ASan execution.

The workflow copies the exact pinned Linux GPU base container options from `fetch_test_configurations.py`: root user, memlock/nofile limits, seccomp setting, video/numeric groups, `/dev/kfd`, `/dev/dri`, GPU-isolation env file, `ROCR_VISIBLE_DEVICES` and `KUBE_CPU_REQUEST`. Its no-ROCm image is the default pinned in `test_component.yml`:

```text
ghcr.io/rocm/no_rocm_image_ubuntu24_04@sha256:a13e8370a2d1d33a910e47669cdf075dbe9554685e78592f22935ec7c7c35e39
```

Source and instrumentation are separate checkouts. The probe requires Linux, `GITHUB_ACTIONS=true`, repository `ROCm/TheRock`, and the exact frozen source SHA. It records its own SHA256 and the instrumentation commit. It reads CPU model/affinity, visible cgroups/limits, device-node metadata, KFD/DRM topology and only the explicit GPU-isolation/environment allowlist. Host topology can contain more devices than the job's runtime isolation permits; a node count is not an allocated-GPU count. No credentials or arbitrary environment are dumped.

If `rocminfo` already exists on the container PATH, `--version` is limited to five seconds and enumeration to15 seconds, each with a1MiB log cap. No tool is installed. An unavailable tool is recorded rather than treated as failed allocation. This existing tool is not a pinned-HEAD product or an ASan/numerical validation. The Python code itself never opens or configures a GPU device. The requested image digest is recorded; root should also retain the container-initialization log and any resolved platform-specific digest. No Docker socket is accessed.

## Root's execution procedure

After reviewing these concrete files, root can create a new uniquely named owned TheRock branch from the frozen source, place this workflow at a dispatchable workflow path, and include the two diagnostic assets at their existing paths. Keep the resulting immutable commit and branch in a separate manifest. Use the existing per-command `sa-faizal` API helper; do not modify the active comparison branch or change the global GitHub account. No dispatch command has been issued here.

Monitor the new owned run and save raw run/jobs API responses and logs. The workflow's **10-minute job timeout starts after allocation**; it does not limit queue waiting. Root must separately enforce a **10-minute queued-without-allocation limit**, then cancel only this owned capacity-probe run and record its raw queued status. That diagnostic cancellation rule does not apply to the ongoing baseline/candidate validation cohort.

Interpret outcomes carefully:

| Observation | Meaning |
|---|---|
| Still queued with no assigned runner after10 minutes | Allocation was not obtained during this window; not proof of permanent access denial or nonexistent capacity |
| Explicit runner-group/repository/workflow access rejection | Record the exact rejection as access evidence; do not generalize a queue delay into this category |
| Assigned runner, then image/container/device-mount initialization failure | Capacity was allocated; setup failed before the probe, requiring setup diagnosis |
| Probe passes and expected nodes exist | Runner/container/device visibility is demonstrated; no GPU compute or ASan/product correctness is established |
| Probe exits2 for missing expected nodes | A runner was allocated but expected KFD/render device visibility is missing; not a numerical-test result |
| rocminfo missing/fails/times out | Existing diagnostic-tool availability/runtime observation only; interpret alongside node and initialization evidence |

A successful allocation diagnostic does not supply a matching SDK. Full candidate GPU validation still needs installable artifacts tied to the fixed source and separate candidate patch, standard runtime/sanitizer setup, driver sanity and required numerical coverage. Do not substitute this probe for those gates.

## Local verification

Only syntax and help checks are permitted locally; actual probing is guarded to Rock CI:

```powershell
python -m py_compile experiments/gpu-capacity/probe.py
python experiments/gpu-capacity/probe.py --help
```

No result is claimed until the reviewed dedicated run actually executes.

Preparation checks passed: Python compilation/help and a static token-for-token comparison of container options against the pinned source lists. No hardware probe was run locally. The upload action is pinned to `ea165f8d65b6e75b540449e92b4886f43607fa02`, resolved read-only from the official `actions/upload-artifact` `v4` tag during preparation. Checkout uses the pinned TheRock workflow's existing action commit.
