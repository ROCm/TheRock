# Cherry-picking commits to release branches

Release maintainers can use TheRock's
[Cherry-pick to release](https://github.com/ROCm/TheRock/actions/workflows/cherry-pick-to-release.yml)
workflow to cherry-pick commits from the default branch of `ROCm/rocm-systems`
or `ROCm/rocm-libraries`, or from TheRock's own default branch. The workflow
creates an unsigned commit chain on a new `shared/cherry-pick/*` branch in the
selected repository and opens a pull request against either
`release/therock-X.Y` or
`release/bkc/therock-X.Y-YYYYMMDD`.

## Running the workflow

The workflow starts in dry-run mode. Select the target repository and release
branch, enter commit SHAs in the order they should be applied, and provide a
JIRA key such as `AIRDEL-1234`. A JIRA key is required for regular release
branches and optional for BKC branches. Set `dry_run` to `false` only for the
coordinated live cherry-pick.

Every requested commit must be reachable from the target repository's current
default branch. The workflow rejects merge commits, commits already present on
the release branch, conflicts, empty cherry-picks, and an existing destination
branch. Validation and cherry-pick failures push nothing. If the push outcome
cannot be confirmed or pull request creation fails after a successful push,
the job reports the destination branch for manual inspection and recovery.

The generated commit message has this form:

```text
Cherry-picks commit <full sha> into the ROCm <release label> branch.

JIRA ID: AIRDEL-1234

<original commit message>
```

Ticketless BKC cherry-picks omit the `JIRA ID` line and its following blank
line.

Destination branch names:

| Release and ticket                                 | Destination branch                                                                              |
| -------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| `release/therock-10.2`, `AIRDEL-1234`              | `shared/cherry-pick/therock-10.2-airdel-1234`                                                   |
| `release/bkc/therock-10.2-20261013`, `AIRDEL-1234` | `shared/cherry-pick/bkc-therock-10.2-20261013-airdel-1234`                                      |
| `release/bkc/therock-10.2-20261013`, no ticket     | `shared/cherry-pick/bkc-therock-10.2-20261013-<first seven characters of the first commit SHA>` |

## Running the script directly

`build_tools/github_actions/cherry_pick_to_release.py` also defaults to dry-run
mode. `CHERRY_PICK_DRY_RUN=true` or `false` sets the default explicitly.
`--dry-run` and `--no-dry-run` override that environment setting: use
`--dry-run` to build and report without publishing, or `--no-dry-run` to push
the branch and create a pull request. The script requires the same App
credentials and TheRock workflow provenance as the workflow. Run it with
`--help` for the input options and corresponding environment variables.

## Access and repository configuration

Operators need permission to dispatch workflows in TheRock. The existing
`systems-assistant` performs operations in TheRock and rocm-systems, while
`assistant-librarian` performs operations in rocm-libraries. An operator's
personal target write access is not used. Before enabling live runs, confirm
that each App installation has Contents, Pull requests, and Workflows write
permission and that target rulesets allow the App to create
`shared/cherry-pick/*` branches containing unsigned commits. The workflow
never pushes directly to a release branch.

Branch creation uses an early existence check followed by a push with an
explicit empty-value lease (`--force-with-lease=refs/heads/<branch>:`). This
allows creation only while the destination branch is absent. The workflow
also requires Git to confirm that a new branch was created before opening a
pull request. An existing branch at the identical commit is rejected as well.
These checks protect the destination even when another workflow or a manual
push creates it during the run.
