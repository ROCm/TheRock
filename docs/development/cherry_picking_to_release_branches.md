# Cherry-picking commits to release branches

Release maintainers can use TheRock's
[Cherry-pick to release](https://github.com/ROCm/TheRock/actions/workflows/cherry-pick-to-release.yml)
workflow to apply reviewed default-branch commits from TheRock,
`ROCm/rocm-systems`, or `ROCm/rocm-libraries` to a release branch in the same
repository. Use a manual process when a change requires conflict resolution.

A live run creates an unsigned commit chain on a new `shared/cherry-pick/*`
branch and opens a pull request against either
`release/therock-X.Y` or `release/bkc/therock-X.Y-YYYYMMDD`.

## Running the workflow

1. Open the workflow and select **Run workflow**.
1. Select the repository containing the commits and release branch.
1. Enter commit SHAs in application order, separated by spaces or commas.
1. Enter the release branch and a JIRA key when required by the table below.
1. Choose whether to preview or publish. You can leave `dry_run` enabled to
   validate the request and preview the generated commits and pull request
   without publishing anything. A dry run is optional; a live run performs the
   same validation before pushing. Set `dry_run` to `false` to create the
   branch and pull request.

The release type and optional JIRA key determine the destination branch:

| Release branch                      | JIRA key                 | Destination branch                                                                              |
| ----------------------------------- | ------------------------ | ----------------------------------------------------------------------------------------------- |
| `release/therock-10.2`              | Required (`AIRDEL-1234`) | `shared/cherry-pick/therock-10.2-airdel-1234`                                                   |
| `release/bkc/therock-10.2-20261013` | Optional, provided       | `shared/cherry-pick/bkc-therock-10.2-20261013-airdel-1234`                                      |
| `release/bkc/therock-10.2-20261013` | Optional, omitted        | `shared/cherry-pick/bkc-therock-10.2-20261013-<first seven characters of the first commit SHA>` |

A regular release run might select `rocm-libraries`, commit `1a2b3c4`,
`release/therock-10.2`, and `AIRDEL-1234`. A ticketless BKC run might select
`rocm-systems`, commits `1a2b3c4, 5d6e7f8`, and
`release/bkc/therock-10.2-20261013`, leaving the JIRA field empty.

## Behavior and failure handling

Every requested commit must be reachable from the selected repository's
current default branch and have exactly one parent. This excludes root commits,
which have no parent, and merge commits, which combine two or more parent
histories. Commits are applied in the requested order, including repeated
inputs. Conflicts, empty picks, and existing destination branches fail the run.
An ancestor that was later reverted can be reapplied; a change whose resulting
tree is already present fails as empty.

A dry run reports the prospective commits and pull request without publishing.
A live run pushes the generated branch and opens the pull request. Validation,
conflict, and empty-pick failures push nothing. If the push outcome cannot be
confirmed or pull request creation fails after a successful push, the job
reports the destination branch for manual inspection and recovery.

The generated commit message preserves the source author and original message:

```text
Cherry-picks commit <full sha> into the ROCm <release label> branch.

JIRA ID: AIRDEL-1234

<original commit message>
```

Ticketless BKC cherry-picks omit the `JIRA ID` line and its following blank
line.

## Running the script directly

`build_tools/github_actions/cherry_pick_to_release.py` defaults to dry-run
mode. `CHERRY_PICK_DRY_RUN=true` or `false` sets the default explicitly.
`--dry-run` and `--no-dry-run` override that environment setting. A live run
requires the selected App token and TheRock workflow provenance. A local dry
run can omit both; its prospective pull request body then omits the triggering
operator paragraph. Run the script with `--help` for the input options and
corresponding environment variables.

## Access and branch safety

Triggering the workflow requires write access to TheRock. The workflow
publishes with repository-scoped GitHub App credentials: `systems-assistant`
for TheRock and rocm-systems, and `assistant-librarian` for rocm-libraries. The
person triggering the workflow does not need write access to rocm-systems or
rocm-libraries.

The workflow expects each App installation to have Contents, Pull requests,
and Workflows write permission. Target repository rulesets must allow the App
to create `shared/cherry-pick/*` branches containing unsigned commits.

A live run never pushes directly to a release branch. It creates the
destination branch only if that branch is still absent, so retries or
concurrent requests fail instead of replacing an existing branch.
