# Configured dependency impact

Run after applying the patch from the repository root:

```bash
python3 -m build_tools.github_actions.artifact_cache plan \
  --db /home/master/erman/cache.db --build build \
  --topology BUILD_TOPOLOGY.toml --changed-subproject amdsmi > amdsmi-plan.json
```

The plan command configures before reading therock_consumer_graph.json and
artifact_subprojects.json. Repeat --changed-subproject for multiple CMake targets.
This accepts subproject target names, not changed file paths or artifact names.
Existing --changed file-path handling remains separate.

dependency_impact contains changed roots, transitive consumers, affected artifact
groups, one shortest explanation path per root/artifact, and the requested
component/bundle identities matching those groups. Family filtering applies to
affected_requested_outputs; affected_artifacts describes the whole configured graph.
Some artifact groups have no cacheable fingerprint; they remain in the graph report.
Unmapped subprojects are reported separately.

Missing, malformed or inconsistent graphs cause an error; the command never
silently substitutes static topology. Standalone Python callers must supply both
metadata files from the same successful configuration. Case normalization bridges
the lowercase consumer graph and mixed-case artifact target names.

This is dependency impact, not observed compilation or build selection.
Cache candidates still depend on current fingerprints and read policy.
No stage is declared skippable, no files are restored, and no fingerprint source
algorithm is changed. Repository-wide source identities may still invalidate
unrelated artifacts. local_cache cache behavior is unchanged.

Validation: 11 focused tests passed. The uploaded real AMD SMI graph produced 25
affected artifact groups. Full checkout regression testing remains required.
