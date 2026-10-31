# Targeted build profiling experiment

Historical evidence: https://github.com/ROCm/rockrel/actions/runs/41147801272.
Rockrel is read only. All experimental execution uses **ROCm/TheRock**
`multi_arch_ci.yml`. Never dispatch or rerun rockrel.

The source baseline is TheRock HEAD
`5b7cb88373362ab80dc0f19605915e534f4158aa`.
Historical TheRock source: `36d2f5f107665ddbb2b69d3d0f90ff7276ef85d8`.
This branch adds diagnostics, pins external CI configuration, and holds build
runner labels at the historical large-runner labels. No compiler flags change.

Initially run only `compiler-runtime,math-libs`, Linux and Windows `gfx103X-all`.
Set stage reuse off; Python, PyTorch, JAX, native Linux and native Windows package
inputs false. CI artifact/log uploads are needed. Release publication and
deployment workflows are not invoked. Do not compare reduced workflow elapsed
time to the full historical release as an optimization result.

Questions: does the long segmented-sort compilation use one core, multiple
compiler workers, or wait on resources? What effective CPU allocation exists?
Which project and command dependencies constrain the tail?

`profile_build.py` wraps the normal stage build, preserving ccache. It samples
process commands/cwd, CPU and resource counters once per second, and records
visible affinity/cgroup v2 limits plus sampler CPU. Sampled process CPU is a lower
bound; POSIX waited-child CPU is separate. Hidden host limits and Windows job
limits need host verification. Sampler CPU does not establish total perturbation.

`capture_build_graph.py` runs after the timed build and preserves generated Ninja
rules, command listings, response-expanded compilation databases, and Ninja logs.
Outputs under `build/logs` use the existing stage-log upload. Neither tool records
the environment or credentials. Match source manifests, actual runner capacity,
toolchain, cache state and configuration before comparing builds.

Subprocess/exit-code/capacity tests and real Ninja response-expansion tests pass
locally on Windows and also run in the Linux and Windows jobs. Instrumentation
alone establishes no speedup. Repeat controlled baseline/candidate runs, separate
cache conditions, and preserve artifact/test coverage before claiming a gain.
This is an experimental branch; it must not be merged as a production speedup.
