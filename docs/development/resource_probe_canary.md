# Resource probe A/B canary

`.github/workflows/resource_probe_canary.yml` is a manually dispatched,
non-publishing validation workflow. It measures three or more paired baseline
and probed observations for these source-declared targets:

* `compiler-runtime`
* `comm-libs`
* `math-libs:gfx110X-all`
* `math-libs:gfx94X-dcgpu`

Pair order alternates between baseline/probe and probe/baseline. Every
observation uses isolated source, build, telemetry, and sink directories.
`ccache` is disabled so the second member of a pair cannot inherit compiled
objects from the first. The probed member uses the `diagnostic` detail profile.

The A/B boundary covers source fetch, inbound-artifact fetch, configure, build,
applicable build tests, packaging tests, and two local release-path equivalents.
The equivalents stream artifacts and logs into an isolated local sink and
record checksums. They use no credentials and perform no network writes. Exact
artifact manifests from the baseline and probed builds must match.
Each member also emits a phase timing table containing only the variant, phase,
start/end timestamps, duration, and exit code. This supplies the unprobed wall
times needed to quantify probe perturbation without collecting commands.

The canary deliberately does **not** call `artifact_manager.py push`,
`post_stage_upload.py`, a release bucket, or a publication workflow. Therefore
it does not measure S3/GitHub upload latency, credential setup, remote service
behavior, or publication. The normal opt-in release workflow measures the real
artifact-push and log-upload commands, with log-upload telemetry written outside
the directory being uploaded and collected afterwards by an always-run step.

The workflow is default-off because it is available only through
`workflow_dispatch`. `source_run_id` must identify a compatible completed run
whose inbound artifacts can bootstrap the selected stages.

## Windows full-release canary

`.github/workflows/windows_resource_probe_canary.yml` is a separate,
default-off canary for the Windows core release path. Windows-native stage jobs
and Windows Python packaging are pinned to `azure-windows-scale-rocm`; tarball
packaging and publication orchestration remain Linux jobs. Use the same pinned
commit for six runs and counterbalance order: pair 1 baseline/probed, pair 2
probed/baseline, and pair 3 baseline/probed. Canary mode disables ccache. The
probed runs select every configured Windows stage (`compiler-runtime`,
`runtime-tests`, every `math-libs` family, `debug-tools`, and `media-libs`),
Windows Python dependency, fetch, build, and upload phases, tarballs, and final
publication logic.

Canary mode makes the final release-bucket publication a dry run and disables
downstream test and framework workflow dispatches. To exercise the real release
transport path, it still performs dev-channel stage artifact, log, tarball, and
Python-package uploads. These are external writes to run-scoped dev artifact
locations and require the normal CI credentials.

After both members of each pair finish, dispatch
`.github/workflows/windows_resource_probe_compare.yml` with their run IDs and
the one pinned commit expected for the six-run cohort. The comparison rejects
mismatched commits, labels, pair numbers, or AB/BA order;
compares relative path, size, and SHA-256 manifests; and uploads sanitized
GitHub job/step timing for baseline-versus-probed perturbation analysis.

On an unmerged feature branch, GitHub does not register a newly added manual
workflow. Use the existing `.github/workflows/multi_arch_release.yml` entrypoint
at that branch ref with its Windows canary mode, pair, variant, and selector
inputs. It emits the same label and manifest evidence. The dedicated canary and
comparison entrypoints become directly dispatchable after they exist on the
default branch.
