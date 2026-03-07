# RFC-0010: Quartz: Central CI/CD Data Hub for the ROCm Ecosystem

- **Author:** Laura Promberger (HereThereBeDragons)
- **Created:** 2026-03-03
- **Modified:** 2026-03-03
- **Status:** Draft
- **Discussion:** https://github.com/ROCm/TheRock/discussions/3782

This RFC proposes Quartz, a central CI/CD data hub that collects TheRock build and test results into a database, distributes status notifications to downstream projects, accepts results reported back by downstream projects, and exposes all data via SQL for dashboard analytics.

## Overview

TheRock CI produces build and test results continuously, but there is no unified place to observe them, no mechanism for downstream projects to be automatically notified when a (nightly) build succeeds, and no way for downstream projects to report their test results back. Each project polls GitHub, scrapes logs, or relies on manual handoff.

Quartz closes this gap. It is implemented as GitHub Actions workflows in a dedicated `ROCm/quartz` repository. All ingestion, validation, and notification logic is written in Python. As a backend it will use a database (choice: ClickHouse Cloud). Four GitHub Apps handle authenticated data transport between repositories.

## Goals

1. **Collect TheRock CI results into a structured database** — per job, per architecture, per branch, for any TheRock branch (nightly, prerelease, PR, ...)
1. **Notify downstream projects automatically**
   — when new nightlies or prereleases are available, via push (workflow trigger) or pull (status.json polling)
   — when any other branch has finished the CI, via push (workflow trigger)
1. **Accept results back from downstream projects** — build and test outcomes against TheRock artifacts, stored in the database
1. **Power analytics dashboards** — Grafana queries database directly; no GitHub API calls at query time
1. **Stay within the GitHub ecosystem** — no external infrastructure required other than the database

## Non-Goals

- Replacing GitHub Actions as the CI/CD system for TheRock or downstream projects
- Real-time streaming analytics — data is available within seconds/minutes of a job completing, not milliseconds
- Mandatory adoption — downstream projects can participate at any level (full push+report, pull-only, or not at all)
- Replacing existing project-level dashboards — Quartz is additive

## Architecture

Quartz sits between TheRock CI and the rest of the ROCm ecosystem. TheRock jobs push results into Quartz as they complete. Quartz validates, stores, and routes that data in three directions: into the database for analytics, out to downstream projects as status notifications, and back from downstream projects when they report their own test results. All processing runs as GitHub Actions workflows in `ROCm/quartz` — there is no separate server or long-running process.

```
           ┌──────────────────────┐
           │   TheRock CI/CD      │
           │  (Build/Test Jobs)   │
           └──────────┬───────────┘
                      │ workflow_dispatch (GH App: Quartz Hauly)
                      │ (job + workflow data)
                      ▼
            ┌────────────────────┐    notify                 ┌───────────────────────┐
            │  Quartz Workflows  │──────────────────────────►│ Downstream Projects   │
            │                    │  workflow_dispatch        │ (vllm, rocm-examples, │
            │  - Validation      │  or status.json           │  rocm-systems, ...)   │
            │  - Transform       │  (GH App: Quartz Conveyor)└────────────┬──────────┘
            │  - Allowlist       │                                        │
            │  - Subscription    │                                        │
            └─────────┬──────────┘                                        │
                      │     ▲            report back                      │
                      │     └─────────────────────────────────────────────┘
                      │               workflow_dispatch
               INSERT │     (GH App: Quartz Hunt or Quartz Kibble-<project>)
                      │
                      ▼
            ┌──────────────────────────┐                  ┌──────────────────────────┐
            │       Database           │◄─────────────────│  Dashboards (Grafana)    │
            │  • therock_workflow_runs │    SQL queries   └──────────────────────────┘
            │  • therock_workflow_jobs │
            │  • downstream_*          │
            └──────────────────────────┘
```

Data flows:

- **TheRock → Quartz:** Each CI job dispatches job data and parent workflow metadata via `workflow_dispatch` (Quartz Hauly app). A final workflow step (`if: always()`) dispatches a completion signal so Quartz can mark unfinished jobs as `timed_out`.
- **Quartz → Database:** Python script validates and inserts via the Database HTTPS API.
- **Quartz → Downstream (push):** Quartz Conveyor app triggers a named workflow in the downstream project when a relevant status change occurs. Subscription declared in `config/subscriber.yml`.
- **Downstream → Quartz (pull):** Downstream projects may poll `release-nightly/<date>/status.json` or `prerelease/<version>/status.json` on a schedule. No installation required.
- **Downstream → Quartz (report back):** Downstream projects dispatch results via `workflow_dispatch` using Quartz Hunt (Tier 1, ROCm-internal) or Quartz Kibble-{project} (Tier 2, external). Quartz validates App ID against `config/allow-list/` before accepting.
- **Dashboards → Database:** Direct SQL queries. No GitHub API calls at query time.

## Repository Structure

The `ROCm/quartz` repository has five distinct areas of concern:

- **`.github/workflows/`** — the three workflow entry points, one per data direction. Workflow YAML is kept minimal: trigger definition, input parameters, and a call to the relevant Python script.
- **`config/`** — human-editable configuration. `subscriber.yml` lists which downstream projects receive push notifications. `allow-list/` maps each project's repository to its expected GitHub App ID; each project's own maintainers own their file via CODEOWNERS.
- **`scripts/`** — all business logic in Python. Validation, schema checking, allowlist enforcement, and database insertion happen here, not in YAML.
- **`release-nightly/` and `prerelease/`** — static JSON status artifacts committed to the repo after each build. Downstream projects that prefer polling over push notifications consume these directly via raw GitHub URLs.
- **`templates/`** — onboarding materials for downstream projects: example workflows for both push and pull subscription models, and an example workflow for reporting results back to Quartz.

```
ROCm/quartz/
├── .github/
│   ├── CODEOWNERS                       # config/allow-list/<project>.yml owned by each project's maintainers
│   └── workflows/
│       ├── receive-therock-data.yml     # Quartz Hauly: ingest TheRock job results
│       ├── notify-downstream.yml        # Quartz Conveyor: push status to subscribers
│       └── receive-downstream-data.yml  # Quartz Hunt/Quartz Kibble: ingest downstream results
│
├── config/
│   ├── subscriber.yml                   # Projects/workflows to notify (Quartz Conveyor targets)
│   └── allow-list/
│       ├── rocm-examples.yml            # Tier 1: repo → Quartz Hunt App ID
│       └── vllm.yml                     # Tier 2: repo → Quartz Kibble-vllm App ID
│
├── scripts/
│   ├── validate_allowlist.py
│   ├── validate_schema.py
│   └── insert_database.py
│
├── release-nightly/
│   ├── 20260215/
│   │   └── status.json
│   ├── latest.json                     # symlink to most recent nightly
│   └── latest_good.json                # symlink to most recent fully passing nightly
│
├── prerelease/
│   ├── 7.11.0/
│   │   └── status.json
│   └── latest.json                    # symlink to most recent prerelease
│
├── templates/
│   ├── subscriber-pull.yml            # Scheduled workflow to poll status.json
│   ├── subscriber-push.yml            # Workflow triggered by Quartz Conveyor
│   └── downstream-send.yml            # Downstream: dispatch results to Quartz
│
└── README.md                          # Onboarding guide pointing to templates/
```

## GitHub Apps and Authentication

Four GitHub Apps handle all authenticated data transport:

| App                         | Direction                  | Who uses it                                               |
| --------------------------- | -------------------------- | --------------------------------------------------------- |
| **Quartz Hauly**            | TheRock → Quartz           | TheRock CI jobs                                           |
| **Quartz Conveyor**         | Quartz → Downstream        | Downstream projects subscribing to push notifications     |
| **Quartz Hunt**             | Internal AMD/ROCm → Quartz | ROCm org projects (shared app, Tier 1)                    |
| **Quartz Kibble-{project}** | External → Quartz          | External community projects (one app per project, Tier 2) |

Tier 1 uses a single shared app across all ROCm-org projects — simpler onboarding, but a compromised credential affects all Tier 1 reporters. Tier 2 uses one app per external project, installed in that project's own org — narrower blast radius at the cost of more setup per project.

**Authentication:** Every incoming `workflow_dispatch` to Quartz must pass two independent checks:

1. **GitHub App token** — GitHub validates the token before accepting the dispatch; cannot be forged.
1. **Allowlist match** — `github.event.installation.id` must match the expected App ID for that repository in `config/allow-list/<project>.yml`.

A project claiming to be `ROCm/vllm` with the wrong App ID is rejected. A project not in the allowlist is rejected and should trigger a security alert.

**Allowlist governance:** External projects' CODEOWNER must approve changes to their own `allow-list` file, ensuring the external project controls their own entry independently of the Quartz team.

Note: The ROCm org currently has ~50 apps registered (limit: 100). Quartz Kibble apps are created in the external project's own org and do not count against this limit.

## Database Design

### Schema

Inspired by the PyTorch HUD — all workflow runs and jobs are captured as individual rows, with a structured schema plus an `extra_info` JSON field for data that does not fit fixed columns.

- `therock_workflow_runs` — one row per TheRock CI run
- `therock_workflow_jobs` — one row per job within a run

ClickHouse's ReplacingMergeTree engine is used for both tables: inserts are always appends, and deduplication happens in the background using `updated_at` as the version column (last write wins). Multiple job retries and status updates are safe without application-level locking.

### Race Conditions and Out-of-Order Messages

Jobs from the same workflow run arrive concurrently and potentially out of order. Each job writes to its own independent row. The workflow adds a current timestamp before sending to Quartz — newest timestamp wins regardless of processing order.

### Lost and Stuck Messages

A workflow job may fail to report due to a runner crash or network failure. The final TheRock workflow step (`if: always()`) dispatches a completion signal — Quartz marks any jobs that never reported as `timed_out`, guaranteeing all runs reach a terminal state.

If the database is unreachable, the GitHub Actions job fails and can be retried manually. A proper dead-letter queue providing automatic retry and replay is a future addition — see Scope and Deferred Work.

## Notification System

### Push (Quartz Conveyor)

Downstream projects install the Quartz Conveyor GitHub App and declare their subscription in `config/subscriber.yml`. When a relevant TheRock status changes (e.g. a new nightly passes all checks), Quartz triggers a workflow in the downstream project.

Projects that prefer not to appear publicly in `subscriber.yml` can store their details as a GitHub secret on the Quartz repository; `subscriber.yml` then references the secret.

*Note: Whether Quartz Conveyor uses `workflow_dispatch` (targets a single named workflow) or `repository_dispatch` (triggers all listening workflows) is to be decided during Phase 2 design.*

### Pull (status.json)

Quartz commits a `status.json` per nightly and per prerelease to the repository. Downstream projects may poll these files on a schedule:

- `release-nightly/<date>/status.json`
- `prerelease/<version>/status.json`

`latest.json` and `latest_good.json` point to the most recent nightly and the most recent fully passing nightly respectively. No installation required — any project can poll these files.

## Security Considerations

### GitHub App Permissions and Blast Radius

Apps installed on Quartz require `Actions: Write`, which unavoidably also permits disabling workflows, cancelling runs, and deleting run logs. There is no "dispatch-only" GitHub permission.

Blast radius is limited to `ROCm/quartz` only — apps are installed on Quartz alone. A compromised Quartz Hunt (Tier 1) app can inject false data for all ROCm-org projects and disable ingestion workflows, but cannot affect any other repository.

### Development Rules

- Internal Quartz processing workflows must not expose a `workflow_dispatch` trigger. Any workflow that does accept `workflow_dispatch` must enforce allowlist verification as its first step — every incoming dispatch must be treated as potentially malicious and carrying crafted payloads.
- Every incoming `workflow_dispatch` workflow must verify
  - the App ID against the allowlist as its first step.
    = if possible: verify that the request comes from a actively running workflow (see newest [meta data available for workflows](https://github.blog/changelog/2026-02-19-workflow-dispatch-api-now-returns-run-ids/) )
- Quartz workflows must have strict, minimal scope over what they can change in the repository and database. Every workflow must declare explicit `permissions:` to enforce this:
  - TheRock → Quartz (`receive-therock-data.yml`): `contents: write` (status.json commit), `actions: none`
  - Quartz → Downstream (`notify-downstream.yml`) and Downstream → Quartz (`receive-downstream-data.yml`): `contents: read`, `actions: none`
- All business logic must be in Python scripts, not in workflow YAML.

### Mitigations

- Pydantic schema validation on all incoming payloads before any database insert.
- GitHub App rate limit (5,000 API calls/hour per installation) provides natural spam protection; no additional mechanism is available in GitHub Actions.
- Workflow disable events should trigger an external alert. GitHub Actions cannot self-monitor a disabled workflow — an external webhook or watchdog in a separate repository is required.
- Anomaly detection should flag impossible values (e.g. test duration of 1 second, non-existent ROCm versions).
- Repo rules to safe guard
  - Branch protection rules
  - Protection rules who can edit what `CODEOWNERS`. `.github/workflows/**`, and `scripts/*`

## Implementation Phases

Quartz is delivered in five phases, each building on the previous.

**Phase 1 — TheRock release workflows → Quartz + status.json**
Create the `ROCm/quartz` repository and database, stand up the Quartz Hauly GitHub App, and implement the TheRock data ingest workflow. Scope is limited to nightly and prerelease workflows. Publish `status.json` artifacts so downstream projects can begin polling immediately.

**Phase 2 — Subscription: Quartz → Downstream + Dashboards**
Implement the Quartz Conveyor app and outbound notification workflow. Onboard the first downstream subscriber. Connect Grafana to ClickHouse Cloud for analytics dashboards.

**Phase 3 — All TheRock workflows → Quartz**
Expand data collection beyond release workflows to all other TheRock CI workflows (PR builds, branch builds, manual runs). The ingest pipeline from Phase 1 is reused; this phase covers all additions need to the python scripts and workflows.

**Phase 4 — Reporting Back: Downstream → Quartz**
Define the downstream callback schema and implement the Quartz Hunt (Tier 1) and Quartz Kibble (Tier 2) ingest workflows. Provide onboarding templates for downstream projects. Covered by a follow-up RFC — see Scope and Deferred Work.

**Phase 5 — Expand Notification system to PR-Subscriptions**
When a downstream project PR triggers a TheRock CI run, notify that project automatically on completion. Covered by a follow-up RFC — see Scope and Deferred Work.

## Alternatives Considered

### No central database (status.json only)

The original scope was a lightweight notification system: TheRock writes a `status.json` per nightly to the Quartz repo, downstream projects poll it on a schedule. This is simple and stays entirely within GitHub.

Rejected because it provides no historical data, no downstream reporting path, no cross-project analytics, and requires every consumer to implement its own polling logic. A database is required to support the dashboard and downstream feedback use cases.

### AWS API Gateway + Lambda + OIDC instead of GitHub Actions

The processing layer (validation, database insert, downstream notification) could be implemented as Lambda functions triggered by an API Gateway REST endpoint, with GitHub Actions OIDC replacing GitHub Apps for authentication.

Advantages: no 64 KB `workflow_dispatch` payload limit, no runner spin-up overhead (~30–60s per insert), managed dead-letter queue via SQS, Lambda auto-scales under burst load, no private keys to manage.

Rejected because it leaves the GitHub ecosystem entirely, adds AWS infrastructure (API Gateway, Lambda, IAM) that a small team must own, and still requires a GitHub App for outgoing notifications to downstream repos. The GitHub Actions model keeps all logic in one repo, is reviewable as YAML and Python, and is sufficient at current CI volume. Can be expanded at a later point if needed.

### AWS Redshift instead of ClickHouse Cloud

| Factor                                | Redshift                                                                                         | ClickHouse Cloud                                                                    |
| ------------------------------------- | ------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------- |
| **Deduplication / upserts**           | Requires MERGE + staging table + scheduled VACUUM. High-frequency upserts accumulate tombstones. | ReplacingMergeTree: engine-level deduplication, always an append, no VACUUM needed. |
| **High-frequency upsert performance** | Poor fit — columnar block rewrites mean updating one row ≈ updating 100,000 rows.                | Designed for continuous INSERT streams.                                             |
| **ENUM enforcement**                  | CHECK constraints exist but are not enforced. Application must validate.                         | Enum8/Enum16 enforced at insert time.                                               |
| **Zero-storage computed columns**     | No ALIAS column type.                                                                            | ALIAS columns — defined in schema, zero storage, computed at query time.            |
| **Automatic TTL / row expiry**        | Requires scheduled DELETE + VACUUM.                                                              | Native TTL per table, per partition, per column value. Runs in background.          |
| **Materialized views**                | Supported; complex queries fall back to full recompute on refresh.                               | Insert-triggered — always current, zero query-time cost, no refresh job needed.     |
| **Query latency**                     | 100–500 ms typical                                                                               | 10–50 ms typical                                                                    |

ClickHouse selected because ReplacingMergeTree is a natural fit for the high-frequency, concurrent job-status-update pattern central to Quartz. Redshift's MERGE + VACUUM model is operationally expensive for this access pattern on a small team.

### `repository_dispatch` or `workflow_call` instead of `workflow_dispatch`

| Approach                            | Decision | Reason                                                                                                                                                                                      |
| ----------------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `repository_dispatch`               | Rejected | Requires `contents:write` (broader than needed); triggers all listening workflows (event type based) in the repo rather than one specific workflow; no payload schema enforcement by GitHub |
| `workflow_call` (reusable workflow) | Rejected | Cannot verify App ID, cross-repo call limitations                                                                                                                                           |
| `workflow_dispatch`                 | Selected | Targets a specific workflow; requires only `actions:write`; inputs are declared in the workflow YAML and validated by GitHub; App ID verifiable via `github.event.installation.id`          |

## Scope and Deferred Work

This RFC covers the core architecture, authentication model, and the first two phases of the data flow: TheRock CI results flowing into Quartz (Phase 1) and Quartz notifying downstream projects (Phase 2).

The following are out of scope and will be addressed in a follow-up RFC once downstream project requirements are gathered:

- **Downstream reporting back (Phase 3):** The schema for downstream callback data, mandatory vs. optional fields, and the Quartz Hunt Tier 1 shared credential model (acceptable blast radius vs. per-project apps) will be defined once downstream projects have confirmed their reporting requirements.
- **PR-subscription flow (Phase 4):** How Quartz identifies and notifies the originating downstream project when a PR-triggered TheRock CI run completes.

## Summary

Quartz provides the ROCm ecosystem with a unified CI/CD data hub: TheRock results flow in, downstream projects are notified, downstream results flow back, and dashboards query the database directly. The design stays within the GitHub ecosystem, uses ClickHouse Cloud for its high-frequency upsert and insert-triggered materialized view support, and secures all data transport with a pseudo-"2FA" GitHub App + allowlist model.

## Revision History

- 2026-03-03: Initial draft (Laura Promberger)
- 2026-03-05: Address feedback, add URL to discussion, adjust GitHub App names, add using secrets for subscriptions (Laura Promberger)
