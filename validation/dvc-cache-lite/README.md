# Small-file Windows cache investigation

Prepared October 14, 2026 by toga-7-shark52 for TheRock #8807 / #8505. The cancelled full-workload plan is retained separately and must not be dispatched under this plan.

This branch contains two manual jobs: hosted Windows and `aws-windows-scale-rocm-prod-mix`. Each has a ten-minute limit. It checks out only this small validation directory and the workflow, uses preinstalled native Python 3.10 or newer, and installs no packages. A runner without a suitable interpreter fails explicitly rather than downloading one. There are no DVC/S3 downloads. Checkout/action bootstrapping still has ordinary small network transfers; result uploads are JSON only.

`windows_sharing_probe.py` tests reads through hardlink aliases while a native handle holds DELETE access, with zero-access/read handles, an independent copied file, a native reader allowing DELETE sharing, and recovery after closing the handle. Holding a handle models a possible rename window; it does not establish that the original CI run encountered it. Run once in the temporary volume and once beneath the cache parent.

`publication_stress.py` runs the unmodified baseline and PR `_store_in_cache` functions against locally generated 2.9 KB files. Each of 500 rounds per version synchronizes 12 publishers at the cache-miss check and runs a separate reader. Real native rename/replace calls execute; no filesystem failures are injected. Unused S3 imports have explicit stubs, and attempting to create an S3 client raises an error. Each round uses a new task-owned temporary directory, checks final contents/staging cleanup, then removes the data. JSON results preserve reader and publisher errors separately and are written incrementally.

Source snapshots are verbatim baseline `b04778ebc8c987fec84b902ec80fcb5a9974c4f2` and PR `b4236cc48911cd7250a4786a1e91778fb79a79f7`; results include fetcher SHA256 and actual Python/platform. Baseline source matches the fetcher used in the October 12 incident.

Local validation: 100 baseline plus 100 PR rounds with 12 publishers pass on Linux under the system interpreter without boto3 installed. This verifies harness operation, not Windows semantics. Formatting, pinned actionlint 1.7.10, Python compilation and Bandit 1.9.4 pass. Native execution remains pending. Inspect probe artifacts explicitly: probe errors are allowed to continue so independent publication stress still runs; workflow color alone is insufficient evidence.

A baseline reader/publisher PermissionError followed by candidate success would support #8807. Candidate errors would show that no-overwrite publication is insufficient under that controlled schedule. Zero errors in both versions remain inconclusive about a rare real-CI failure. Large fixture contents are unnecessary to investigate these filesystem rules, though these probes cannot recreate the original complete workload or network timing.
