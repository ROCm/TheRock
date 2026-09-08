# Skills

This directory holds reusable "skills" — structured guidance and tooling that an AI coding
assistant can load on demand, and that humans can read directly. The
skills here focus on **pull-request quality**: making the high-quality path the easy path so PRs land
traceable, tested, and safe to merge.

These skills are **advisory**. They help you author, review, and pre-merge gate a PR, but they do not
approve or block on their own and never post to GitHub or Jira without your explicit approval.

## PR-quality skills

| Skill                                        | Use it when                                                                                                                                                                                                                                                                      |
| -------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`rocm-pr-quality/`](rocm-pr-quality/)       | Authoring, reviewing, or pre-merge gating **any** ROCm-library PR. This is the library-agnostic base — always start here. See [`SKILL.md`](rocm-pr-quality/SKILL.md) and the detailed rubric in [`reference.md`](rocm-pr-quality/reference.md).                                  |
| [`therock-pr-quality/`](therock-pr-quality/) | A PR touches **TheRock build repo**: the superbuild / `cmake/` / `THEROCK_ENABLE_*` options, submodules and patches, `artifact-*.toml` / `BUILD_TOPOLOGY.toml`, or reusable CI workflows. Extends the base — read the base first. See [`SKILL.md`](therock-pr-quality/SKILL.md). |

Each skill exposes three actions you invoke in one sentence:

- **Author assist** — "help me write a PR worth reviewing."
- **Review assist** — "help me review this PR well."
- **Pre-merge gate** — "is it actually safe to press merge right now?"

Component repositories add their own thin overlays on top (for example, `hipblaslt-pr-quality` in
`rocm-libraries`). Overlays may tighten but never weaken the base rules.

## Related automated check

[`therock_pr_bot/`](therock_pr_bot/) is the **enforcement** counterpart to the advisory skills: an
automated policy check (branch-name convention, Conventional Commits PR title, and similar) defined in
[`policy.yml`](therock_pr_bot/policy.yml). [`FAQ.md`](therock_pr_bot/FAQ.md) explains what each check
means and how to fix a failure.

## See also

- [`CONTRIBUTING.md`](../CONTRIBUTING.md) — contribution workflow and PR process.
- [`docs/development/style_guides/`](../docs/development/style_guides/) — the canonical Bash, CMake,
  GitHub Actions, and Python style guides the skills defer to.
