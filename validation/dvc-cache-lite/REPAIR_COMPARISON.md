# Follow-up small-file repair comparison

Prepared October 14, 2026 by toga-7-shark52. Publication and dispatch of this follow-up remain pending. It extends the successfully executed small-file investigation without adding any dependency or artifact downloads.

[The preceding native run](https://github.com/ROCm/TheRock/actions/runs/38617563147) reproduced ordinary Python cache-read PermissionError in both baseline and PR on hosted Windows 3.12.10 and AWS Windows 3.13.7. In 500 rounds each, reader errors were baseline 1079 / PR 733 (hosted) and baseline 558 / PR 500 (AWS). Published contents were correct and no staging files remained. Hosted baseline also had 30 publisher errors: failed os.replace followed by cache hashing that raised single-path Errno 13. These are controlled interleavings, not CI failure-rate estimates or exact attribution of the historical incident.

The held-handle probe narrows the mechanism: after renaming an open staged hardlink into the cache, a normal Python read of that newly renamed cache path fails until the DELETE-access handle closes. A native reader that allows DELETE sharing succeeds. Ordinary reads through the surviving source hardlink succeed, even during the conflict; the sharing rule is therefore more specific than a blanket prohibition across hardlink aliases.

This branch compares four variants using locally generated 2.9 KB files:

- `baseline`: verbatim baseline publication.
- `candidate`: verbatim PR #8807 publication.
- `direct-link`: experimental publication with os.link directly into the final cache name, no staging rename. Refuse unsupported hardlinks rather than silently falling back to an untested path. Accept an existing winner only after size/MD5 verification. This is not a full production solution: cross-volume and no-hardlink fallback still require design and validation.
- `locked-candidate`: unchanged PR publication with a shared lock honored by the reader and all publishers. A barrier starts all publishers before they acquire the lock. The lock covers the existence check and operation, so only one publisher wins; readers cannot overlap publication. This models full per-hash synchronization, not a lock around writes alone. Process-local locking does not cover independent processes or external readers.

Each job performs 500 rounds per variant with 12 publishers and one reader, recording reader and publisher exceptions, content verification and staging cleanup. The baseline/PR variants retain synchronized cache misses; the lock variant intentionally prevents concurrent cache operations. Review read counts so a zero-error result is not accepted without reader activity.

The same manual workflow requests hosted Windows and the affected AWS pool, with ten-minute limit/job, sparse checkout, preinstalled native Python, no package installs, no S3/DVC network use, and small JSON artifacts. The prior PR and prior validation branch remain unchanged. Local 100-round Linux checks for all four variants pass; native follow-up remains pending.

If direct-link publication passes the Windows controls, it offers a way to avoid this rename window even for readers that do not participate in a process-local lock. If only locking passes, finish the full cache-operation locking prototype and test actual small synthetic pull() workloads. Validate corrupt-cache eviction, repeated manifests, failed-download recovery, independent-hash concurrency and separate pull/process boundaries before selecting the production repair. Keep worker traceback diagnostics and do not close #8505 on this control alone.
