# ROCm Libraries Gardeners

This documents the mechanics of
[gardening](https://github.com/ROCm/TheRock/blob/main/docs/rfcs/RFC0002-MonoRepo-Gardener-Rotations.md)
for the ROCm Libraries. If you haven't read the above doc, please start there.

## Becoming a member

Gardeners will need to be members of the [Compute Library Gardeners team](https://github.com/orgs/ROCm/teams/compute-library-gardeners).
Please contact an owner to become a gardener.

## Communications channel

We will be leveraging a shared Teams channel that contains all gardeners as well as core
infrastructure team members. You will be added to this channel once you become a member.

For anyone who wants to reach a gardener please email:
[rocm-libraries-gardeners](mailto:rocm-libraries-gardeners@amd.com)

## Mechanics of Gardening

Your primary job is to keep the mono-repo shippable. In order to facilitate this we've made
status badges for all relevant CI available here:
https://github.com/ROCm/rocm-libraries?tab=readme-ov-file#monorepo-status-and-ci-health.
Effectively your job is to ensure all status badges are green. All of these status
badges are clickable which will allow you to deep-dive on any failures quickly. If any
CI is missing, please file an issue leveraging the "gardener" tag, ping on the teams chat,
or preferably, add it yourself. You'll probably be tagged to review the PR if someone
else gets to it first.

## Notes on Privileges

Developers will not be able to bypass pre-submit checks in this repository unless an admin or
gardener pushes it through. This is being done intentionally to ensure we keep the quality of
the tree green. This also means that you will be asked to push changes through without
additional context. Your duty is to ensure you keep the tree green (or make it greener) so gardeners will need to understand the context before approving
any of these changes. Changes
that are ok:

- Reverts to fix broken things.
- Fast-forward fixes where reverts are unclear
- Fixes unrelated to code health (docs, etc)

On a case by case basis you should consider critical customer fixes, but these should be considered
as a group and likely admins should be approving the majority of those.

As an example to include an admin: *we have a critical feature but develop is broken and it is unrelated to our changes*

## Scope of Gardeners and Developers

In scope:
- Gardeners are responsible for ensuring develop (post-submit) checks remain green.
- If a post-submit check is red, the gardeners should review the failing CI system and triage the issue.
- No matter the issue, gardeners should notify the larger gardening team at least once per day about any post-submit failures.
- If the issue is related to a failure in the CI system (not a code change), the gardener should note the issue,
  verify whether existing PRs are facing the same problem, and notify the appropriate CI team, escalating the issue if required.
- If the issue is related to a code change, the gardener should isolate the error message, and notify the
  appropriate component owners with a link to the log (reference the [CODEOWNERS](../.github/CODEOWNERS) file).

Not in scope:
- Gardeners are not responsible for fixing code changes that break post-submit checks.
- Gardeners are not responsible for monitoring the health of every open PR.

Developer responsibilities:
- If developers find CI system failures in their PR (pre-submit) checks they should notify the gardener on rotation and the appropriate CI team.

### Beyond the Responsibilities

Gardeners should generally aim to be efficient at operating the CI/CD systems and doing first pass triage and routing.
Especially for people new to the role, this will involve more reaching out for help and coordinating resolution, but as experience increases,
it is natural to take a more active role in helping to route and do first pass triage oneself.
While going the extra mile on this is not a requirement of the role, efficient gardeners should aim to develop a proficiency with the
tools and their colleagues such that their judgment reduces the overall toil to the team. Often people who develop these skills find it
more effective to look a little bit more deeply at failures and route for resolution properly in one step.

This kind of investment is deeply valued for the overall health of the team and is encouraged.

### CI Teams

CI | Main primary contact | Team
---- | ------- | ---------
Math CI | eidenyoshida | [ROCm/rocm-math-lib-ci-team](https://github.com/orgs/ROCm/teams/rocm-math-lib-ci-team)
External (Azure) CI | jayhawk-commits | [ROCm/external-ci](https://github.com/orgs/ROCm/teams/external-ci)
TheRock CI | geomin12 | [ROCm/therockinfra](https://github.com/orgs/ROCm/teams/therockinfra)

## Gardener Rotation

[Confluence doc for Gardener Rotation](http://u.amd.com/rocm-libraries-gardeners)

It is the responsibility of the current gardeners to update the table when the gardeners rotate.

### Log

Filling in this section is optional while on rotation. While this level of
organization and tracking is not expected from all members, seeing the incident
history and actions taken in one location can be useful. However, for bugs that you can't immediately address
please file a new GH issue and label it with the "gardener" label.

You can see current list of [gardener known bugs](https://github.com/ROCm/rocm-libraries/issues?q=is%3Aissue%20state%3Aopen%20label%3Agardener)

Date | Library | Issue overview | Link to details | Resolved?
---- | ------- | -------------- | --------------- | ---------
6/30 | | | | ✅
