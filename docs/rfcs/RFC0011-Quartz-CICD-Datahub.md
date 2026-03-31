# RFC-0011: Quartz: Central CI/CD Data Hub for the ROCm Ecosystem

- **Author:** Laura Promberger (HereThereBeDragons)
- **Created:** 2026-03-03
- **Modified:** 2026-03-18
- **Status:** Draft
- **Discussion:** https://github.com/ROCm/TheRock/discussions/3782

This RFC proposes Quartz, a central CI/CD data hub that collects TheRock build and test results into a database, distributes status notifications to downstream projects, accepts results reported back by downstream projects, and exposes all data via SQL for dashboard analytics.

## Overview

TheRock CI produces build and test results continuously, but there is no unified place to observe them, no mechanism for downstream projects to be automatically notified when a (nightly) build succeeds, and no way for downstream projects to report their test results back. Each project polls GitHub, scrapes logs, or relies on manual handoff.

Quartz closes this gap. It is implemented as GitHub Actions workflows in a dedicated `ROCm/quartz` repository. All ingestion, validation, and notification logic is written in Python. ClickHouse Cloud is used as the database backend. Four GitHub Apps handle authenticated data transport between repositories.

## Goals

1. **Collect TheRock CI results into a structured database** — per job, per architecture, per branch, for any TheRock branch (nightly, prerelease, PR, ...)
1. **Notify downstream projects automatically**
   — when new nightlies or prereleases are available, via push (workflow trigger) or pull (status.json polling)
   — when any other branch has finished the CI, via push (workflow trigger)
1. **Accept results back from downstream projects** — build and test outcomes against TheRock artifacts, stored in the database
1. **Power analytics dashboards** — Dashboards query database directly; no GitHub API calls at query time
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
            ┌──────────────────────────┐                  ┌──────────────────────────────────────────┐
            │       Database           │◄─────────────────│  Dashboards (Grafana, TheRock HUD, ...)  │
            │  • therock_workflow_runs │    SQL queries   └──────────────────────────────────────────┘
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

The `ROCm/quartz` repository has six distinct areas of concern:

- **`.github/workflows/`** — the three workflow entry points, one per data direction. Workflow YAML is kept minimal: trigger definition, input parameters, and a call to the relevant Python script.
- **`config/`** — human-editable configuration. `subscriber.yml` lists which downstream projects receive push notifications. `allow-list/` maps each project's repository to its expected GitHub App ID; each project's own maintainers own their file via CODEOWNERS.
- **`scripts/`** — all business logic in Python. Validation, schema checking, allowlist enforcement, and database insertion happen here, not in YAML.
- **`release-nightly/` and `prerelease/`** — static JSON status artifacts committed to the repo after each build. Downstream projects that prefer polling over push notifications consume these directly via raw GitHub URLs.
- **`templates/`** — example workflow files for downstream projects: pull and push subscription models, and reporting results back to Quartz.
- **`docs/`** — guides for pull subscription, push subscription, and reporting back, populated incrementally across phases.

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
│   ├── validate_schema.py
│   ├── validate_allowlist.py
│   ├── insert_therock_data.py
│   ├── insert_downstream_data.py
│   ├── notify_subscribers.py
│   ├── generate_status_json.py
│   └── ...                              # additional scripts added per phase
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
├── docs/
│   ├── pull-subscription.md           # Guide: polling status.json (Phase 1)
│   ├── push-subscription.md           # Guide: Quartz Conveyor push model (Phase 2)
│   └── reporting-back.md              # Guide: reporting results to Quartz (Phase 3)
│
└── README.md                          # What Quartz is; links to docs/ and RFC-0011
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
1. **App ID match** — `github.event.installation.id` must match the expected App ID. For downstream projects this is declared in `config/allow-list/<project>.yml`. For Quartz Hauly (TheRock → Quartz), the verification mechanism — hardcoded App ID in the workflow vs. a dedicated allow-list entry — is to be decided during implementation.

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

Jobs from the same workflow run arrive concurrently and potentially out of order. Each job writes to its own independent row.

**Deduplication strategy:** ReplacingMergeTree uses a `version` integer column (not a timestamp) to determine which row wins. The version is computed as `run_attempt * 2 + signal_type`, where `signal_type` is 0 for START and 1 for FINISH. This guarantees FINISH always wins over START within the same attempt, and later attempts always win over earlier ones — with no dependency on runner clocks.

**Timestamps:** `started_at` and `completed_at` are not taken from the runner clock. `send_to_quartz.py` fetches them from the GitHub API using the `job_id`, ensuring they come from GitHub's infrastructure. For FINISH signals, Quartz fetches `completed_at` via the GitHub API when processing the dispatch — by the time a Quartz runner is assigned, the TheRock job has already completed and `completed_at` is available. If it is not yet recorded, Quartz retries the API call with a short backoff.

### Lost and Stuck Messages

A workflow job may fail to report due to a runner crash or network failure. The final TheRock workflow step (`if: always()`) dispatches a completion signal — Quartz marks any jobs that never reported as `timed_out`, guaranteeing all runs reach a terminal state.

If the database is unreachable, the GitHub Actions job fails and can be retried manually. A proper dead-letter queue providing automatic retry and replay is a future addition — see Scope and Deferred Work.

## Notification System

### Push (Quartz Conveyor)

Downstream projects install the Quartz Conveyor GitHub App and declare their subscription in `config/subscriber.yml`. When a relevant TheRock status changes (e.g. a new nightly passes all checks), Quartz triggers a workflow in the downstream project.

Projects that prefer not to appear publicly in `subscriber.yml` can store their details as a GitHub secret on the Quartz repository; `subscriber.yml` then references the secret.

*Note: Quartz Conveyor uses `workflow_dispatch` (targets a single named workflow) as the starting point. `repository_dispatch` (triggers all listening workflows) may be added on request.*

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
- Quartz workflows must have strict, minimal scope. The repository default workflow permission is set to `read-all` (`contents: read`) at the repo level. Individual jobs that require more elevate only what they need:
  - `receive-therock-data.yml`: `contents: write` (to commit `status.json`) and `issues: write` (to open a GitHub Issue on failure) at the job level, all others inherit `read-all`
  - `notify-downstream.yml`: no elevation needed, `read-all` default is sufficient
  - `receive-downstream-data.yml`: `issues: write` at the job level (to open a GitHub Issue on failure), all others inherit `read-all`
- All business logic must be in Python scripts, not in workflow YAML.

### Mitigations

- Pydantic schema validation on all incoming payloads before any database insert.
- GitHub App rate limit (5,000 API calls/hour per installation) provides natural spam protection; no additional mechanism is available in GitHub Actions.
- Workflow disable events should trigger an external alert. GitHub Actions cannot self-monitor a disabled workflow — an external webhook or watchdog in a separate repository is required.
- Anomaly detection should flag impossible values (e.g. test duration of 1 second, non-existent ROCm versions).
- Repo rules to safe guard
  - Branch protection rules
  - Protection rules who can edit what `CODEOWNERS`. `.github/workflows/**`, and `scripts/*`

## Scope and Deferred Work

This RFC covers the core architecture, authentication model, and the first two phases of the data flow: TheRock CI results flowing into Quartz (Phase 1) and Quartz notifying downstream projects (Phase 2).

Phases 3–5 are out of scope and will be addressed in a follow-up. Phase 3 requires detailed downstream project requirements to be gathered first. Phase 4 (expanding data collection beyond nightly and prerelease) will be scoped once Phase 1 and 2 are operational — which additional workflows to instrument depends on operational experience and on decisions around PR and manual builds that are entangled with Phase 5.

## Implementation Phases

Quartz is delivered in five phases, each building on the previous.

Across all phases, a dedicated test repository mocks TheRock and downstream project workflows to allow development and validation in a non-production environment before changes are applied to the real pipelines.

**Phase 1 — TheRock release workflows → Quartz + status.json**
Create the `ROCm/quartz` repository and database, stand up the Quartz Hauly GitHub App, and implement the TheRock data ingest workflow. Scope is limited to nightly and prerelease workflows. Publish `status.json` artifacts so downstream projects can begin polling immediately.

**Phase 2 — Subscription: Quartz → Downstream + Dashboards**
Implement the Quartz Conveyor app and outbound notification workflow. Onboard the first downstream subscriber. Connect the dashboards to use ClickHouse Cloud for analytics.

**Phase 3 — Reporting Back: Downstream → Quartz**
Define the downstream callback schema and implement the Quartz Hunt (Tier 1) and Quartz Kibble (Tier 2) ingest workflows. Provide onboarding templates for downstream projects. Covered in a follow-up — see Scope and Deferred Work.

**Phase 4 — All TheRock workflows → Quartz**
Expand data collection beyond nightly and prerelease to additional TheRock CI workflows. Which workflows to include (dev nightly, PR builds, manual runs) will be decided once Phase 1 and 2 are operational. Covered in a follow-up — see Scope and Deferred Work.

**Phase 5 — Expand Notification system to PR-Subscriptions**
When a downstream project PR triggers a TheRock CI run, notify that project automatically on completion. Covered in a follow-up — see Scope and Deferred Work.

## Phase 1 Implementation Plan

*Detailed implementation plans are provided for Phase 1 and Phase 2 only. Phases 3–5 will be detailed in a follow-up. The granular implementation tasks will be tracked as GitHub Issues in the respective repositories and organised in a GitHub Project.*

### A. Infrastructure

- Create `ROCm/quartz` repository *(completed)*
- Register and install GitHub Apps (Quartz Hauly, Quartz Conveyor, Quartz Hunt) *(completed)*
  - Store secrets of them in relevant repositories *(completed)*
- Create test repository with mock TheRock and downstream workflows for development and validation in a non-production environment *(completed)*
- Quartz repository access control: branch protection on `main`, CODEOWNERS for `.github/workflows/`, `scripts/`, `config/allow-list/`, explicit `permissions:` on all workflows
- Database setup:
  - Provision ClickHouse Cloud service
  - Create database and deploy schema: `therock_workflow_runs`, `therock_workflow_jobs`
  - Create read/write service account for Quartz workflows
  - Store database credentials as secrets in the Quartz repository

### B. Quartz Repository: Database Ingest

- `receive-therock-data.yml`: accepts `workflow_dispatch` from TheRock, verifies Quartz Hauly App ID (`github.event.installation.id`) as first step, calls Python scripts. Permissions: `contents: write`, all others `none`
- `scripts/validate_schema.py`: Pydantic validation of the incoming payload; rejects malformed or out-of-range values before any database write
- `scripts/insert_therock_data.py`: inserts rows into `therock_workflow_runs` and `therock_workflow_jobs`; on failure, marks the workflow run as failed

### C. Quartz Repository: Updating/Creating status.json

- Updated on every incoming TheRock signal via `scripts/generate_status_json.py`
- `scripts/generate_status_json.py`: reads current `status.json` as a Python dict, applies the new job's data, writes back as JSON, commits. Uses a git retry loop (pull → update → push, up to 5 retries with random 1–3 s backoff) to handle concurrent updates. A `status.json` commit that exhausts all retries opens a GitHub Issue and marks the workflow run as failed
- Structure: per platform → architecture → pipeline (rocm / pytorch / jax) → job list
- Published to:
  - `release-nightly/<date>/status.json`
  - `prerelease/<version>/status.json`
  - `latest.json` → most recent nightly
  - `latest_good.json` → most recent fully-passing nightly *(definition of "fully passing" to be decided: e.g. all architectures pass, or a required subset; PyTorch and JAX included or ROCm only)*
- Add pull subscriber onboarding template: `templates/subscriber-pull.yml` — scheduled workflow polling `status.json`

### D. TheRock Repository Changes

- Develop and validate dispatch steps against the test repository before applying to production TheRock workflows
- Instrumented workflows: nightly and prerelease
- `send_to_quartz.py`: aggregates and dispatches job data to Quartz via the Quartz Hauly app. A single script handles all contexts via arguments (pipeline: rocm/pytorch/jax, job type: build/test, signal: start/finish). Fetches `started_at` and `completed_at` from the GitHub API using `job_id` — no runner clock used. Sets `version = run_attempt * 2 + signal_type` for deduplication
- Add dispatch step calling `send_to_quartz.py` to each ROCm, PyTorch, and JAX build and test job: fires on job start and on job finish (`if: always()`)
- Add workflow-level completion signal: a final step with `if: always()` dispatches after all jobs — Quartz uses this to mark any unreported jobs for this `run_id` as `timed_out`

## Phase 2 Implementation Plan

### A. Quartz Conveyor: Downstream Subscription and Notification

- Dispatch mechanism: `workflow_dispatch`; `repository_dispatch` maybe available on request
- `notify-downstream.yml`: triggered on relevant TheRock status changes, reads `config/subscriber.yml`, dispatches to each subscriber via Quartz Conveyor app.
- `scripts/notify_subscribers.py`: reads subscriber config, sends `workflow_dispatch` to each subscribed workflow
- Add push subscriber onboarding template:
  - PR template and GitHub App installation instructions and implications to be added as subscriber
  - `templates/subscriber-push.yml` — workflow triggered by Quartz Conveyor

### B. Validation on Test Repository

- Validate full push notification flow end-to-end using the test repository mocking TheRock and a downstream subscriber
- Onboard real subscribers (starting with rocm-examples) only after successful validation

### C. Dashboards

- Create read-only ClickHouse service account for dashboards
- Create ClickHouse materialized views for pre-aggregated dashboard queries (pass rates per architecture, nightly history, build duration trends)
- Connect dashboards to ClickHouse using the read-only account
- Create new dashboards or update existing ones to use ClickHouse as the data backend, querying the materialized views

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

## Summary

This RFC proposes Quartz, a central CI/CD data hub that gives the ROCm ecosystem a unified view of TheRock build and test results, automatic notification of downstream projects, and a path for downstream projects to report their own results back. It covers the full architecture and security model, and provides detailed implementation plans for Phase 1 (TheRock data ingest and `status.json`) and Phase 2 (downstream notifications and dashboards). Phases 3–5 are defined at a high level and will be detailed in a follow-up once early operational experience with Quartz and downstream project requirements are in hand.

## Revision History

- 2026-03-03: Initial draft (Laura Promberger)
- 2026-03-05: Address feedback, add URL to discussion, adjust GitHub App names, add using secrets for subscriptions (Laura Promberger)
- 2026-03-16: Add Phase 1 and Phase 2 implementation plans; update deduplication strategy to version integer with GitHub API timestamps; resolve dispatch mechanism to `workflow_dispatch`; various consistency fixes (Laura Promberger)
- 2026-03-18: Add missing scripts to repository structure; add GitHub Issues and GitHub Project tracking note; minor fixes (Laura Promberger)
- 2026-03-19: Refactor workflow permissions to read-all default with job-level elevation; move Scope and Deferred Work before Implementation Phases; rewrite Summary; swap Phase 3 (downstream reporting) and Phase 4 (all TheRock workflows); add `docs/` directory to repository structure (Laura Promberger)
- 2026-03-31: Clarify dashboards are not Grafana-specific — any dashboard related the TheRock CI is in scope (Laura Promberger)
