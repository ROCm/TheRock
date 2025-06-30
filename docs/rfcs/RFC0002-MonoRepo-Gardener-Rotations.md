# Gardeners

Gardeners[^1] are responsible for keeping their component areas continually shippable.
Build, test, and nightly release workflows should be kept passing ("green").
When tests fail ("red") or builds fail ("black"), corrective action like identifying
and reverting culprit commits should be taken quickly, so that developers can remain
productive and bugs are caught closer to commit times.

In order to facilitate this work, gardeners have some super powers afforded to them
that other developers do not such as:

- The ability to bypass any checks in the CI.
- Authority to request that any developer on the project who is implicated in a breakage/\* degradation of service/resolution stop what they are doing and participate.
- In the event of major breakages, the ability to escalate and pull higher level management in to resolve issues.

Note this concept isn't novel and can be referred to as "build cops" or "build sheriffs"
in other projects. Good judgement and commitment are critical to being a good gardener.

## Overview

### Responsibilities and priorities

1. When regressions occur that break CI workflows
   - Communicate that something is broken
   - Identify the culprit commit(s)
   - Notify the authors/reviewers of those commits
   - Revert those changes
   - If it's not obvious how to, facilitate getting changes in to get back to green (see 2)
1. For changes that need to bypass presubmit checks (to get back to green), serve as a
   central point of contact to help land those changes using your judgement.
1. File issues for (or add) presubmit tests that could have caught regressions that were fixed
   in 1/2.
1. Encourage and contribute to best developer practices
1. Engage with the infrastructure team for anything that you don't know (they are
   also part of the gardener chat)
1. Keep notes for the next gardener in a shared log or communicate via our gardener chat

The gardener is expected to be the primary contact while "on-duty" for
any changes that break workflows, during their business hours. We will endeavor to have
gardeners in every time zone that matters so that we can have one available whenever
they are needed.

This is _not_ an "on-call" position and responses outside of working hours are
not expected. In the event that a failure occurs while there is no gardener available,
a list of non on-duty gardeners will be available who have shared superpowers and similar duties.

> [!IMPORTANT]
> It's an expectation that on-duty gardeners adhere to an SLO to start responding to issues
> within 2 hours. This means that gardeners will need to routinely check for issues at least
> that often while they are working.

> [!NOTE]
> The gardener does not need to personally fix every issue/incident
> themselves, but they should be proactively monitoring project health and
> reactively responding to reports as they come in. When responding to an issue,
> the gardener may delegate to local project developers that follow up
> to ensure that PRs get reverted.

### Prerequisites for gardening

You will need to be part of the corresponding GitHub gardener team - this will be documented
per repo.

### Team member rotation

Initially rotations will be hand selected. We are working on integrating with tools
to make the current on-call more discoverable and alerts to be sent automatically to
them.

## Communication norms

When responding to an incident, communicate early and often:

1. Acknowledge ("ack") an issue where it is reported and announce the actions
   you are taking or delegating clearly. Do this in the corresponding channel
   you will be provided access to.
1. When an issue or incident affects multiple developers or will take time to
   resolve, consider announcing that to affected developers and filing a
   tracking issue on GitHub.

Primary communication about the changes in a commit should happen on the pull
request that commit came from. This ensures that members of the open source
community can see the discussion and avoid duplicating effort.

Secondary communication like real time discussion can occur on Discord, Slack,
or Teams. Initially we will be leveraging Teams for these discussions.

## Reacting to issues

### Monitoring and alerting

TBD - We will work to create alerting on corresponding channels.

### Identify the source of the issue

When CI is failing view the most recent job that passed.
Check PRs that were submitted between the two. If any PR bypassed presubmits
it's an obvious candidate. Otherwise, this your candidate list. If no obvious
change in this list, consult the gardener chat for more help if you can't
figure it out yourself.

It's also possible that it's not PR related and either broken infra or flaky test;
see the following two sections.

### Flaky tests

If a test fails periodically, some options are:

- De-flake the test
- Disable the test or make it opt-in
- Remove the test
- Have the test run multiple times

### Unhealthy CI machines

We use both standard GitHub-hosted runners and a growing number of self-hosted
runners across our projects. Sometimes the self-hosted runners need maintenance
or troubleshooting to keep them running jobs successfully. See the section
below for information about each type of self-hosted runner we use.

### Reverting vs fixing-forward

When a commit breaks a build, the first thought should be to revert that commit.

- Reverts are generally safe, fast to make, and easy to review.
- Fix-forwards can be tempting but carry additional risks and are necessarily
  urgent to review.
- When the shared `develop` branch is affected by a failing workflow, every
  developer working off of that branch is affected. Reverting a change puts the
  burden for fixing the issue back on just the original author of the change.

If multiple commits are implicated in broken workflows, cleanly reverting them
all sequentially is less risky than reverting manually.

## Other references

Google SWE Book:

Similar rotations and playbooks on other projects:

- https://github.com/flutter/flutter/blob/master/docs/infra/Flutter-Framework-Gardener-Rotation.md
- https://github.com/kubeflow/testing/blob/master/playbook/buildcop.md
- https://drake.mit.edu/buildcop.html

Developer policies:

- https://llvm.org/docs/DeveloperPolicy.html#patch-reversion-policy
- https://github.com/oppia/oppia/wiki/Revert-and-Regression-Policy

[^1]: https://abseil.io/resources/swe-book/html/ch23.html#ci_at_google
