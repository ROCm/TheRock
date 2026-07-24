# TheRock PR Bot — Policy FAQ Doc

**TheRock PR Bot** is an automated Pull Request (PR) gatekeeper.
On every Pull Request, it runs a set of policy checks — branch naming,
title/description, forbidden files, unit tests, and required CI checks —
then posts a single results table comment summarising what passed or failed.
PRs that fail key checks are flagged with a **`Not ready to Review`** label
until the issues are resolved.

It helps save time on the **first level of PR review** by automating the basic
checks a reviewer would otherwise perform manually, reducing first-pass review
time. Reviewers can filter out **`Not ready to Review`** PRs from their pending
review list and only start reviewing once that label has been removed — i.e.
once all policy checks have passed.

This document explains what each policy check means, why it exists, and how to fix a failure.

> **Note:** This is **NOT an AI Bot and does not use any LLMs**. It is a
> deterministic, rule-based checker driven entirely by `policy.yml`.

______________________________________________________________________

## 🌿 Branch Name

**What does it check?**
Your branch name must follow the agreed naming convention so PRs are easy to trace back to a contributor and topic.

**Allowed formats**

| Pattern                         | Example                                             |
| ------------------------------- | --------------------------------------------------- |
| `users/<username>/<anything>`   | users/dgaliffi/fix/remove-build-boost-option        |
| `users/<username>/<anything>`   | users/frepaul/ROCm-end-user-project-workflow        |
| `shared/<anything>`             | shared/add-runner-health                            |
| `<single-segment-name>`         | bump-rocm-libraries-936a6c7                         |
| `<single-segment-name>`         | ZIP-packaging-RFC                                   |
| `dependabot/<anything>`         | dependabot/github_actions/github-actions-3dfd2199fc |
| `revert-<pr-number>-<anything>` | revert-5217-users/derobins/add_hipfile_support      |

Rules:

- A recognised **prefix** must be present (`users/`, `shared/`, `dependabot/`, `revert-…`) — or the branch must be a single segment.
- **Uppercase letters are allowed** (acronyms and module names are common, e.g. `ROCm`, `SMP`, `RFC`).
- For `users/`, the `<username>` segment may contain letters (upper or lower), digits, and hyphens.
- **Anything after the prefix is allowed**, including nested `namespace/feature` paths (e.g. `users/dgaliffi/fix/remove-build-boost-option`).

**How to fix**
Rename your branch before opening the PR:

```bash
git branch -m old-name users/<your-username>/<topic>
git push origin -u users/<your-username>/<topic>
```

______________________________________________________________________

## 📝 PR Title

**What does it check?**

> **Note:** In the results table, the title and description checks are reported together as a single **PR Title/Description** row. Any title *or* description failure shows up there.

**Length rules (only)**

- Minimum: **10** characters
- Maximum: **100** characters

> The title is validated by **length only**. There is no enforced format
> (e.g. Conventional Commits) and no forbidden-word list — any wording is
> accepted as long as it is 10–100 characters long.

**How to fix**
Edit the PR title on GitHub (top of the PR page → pencil icon) so it is between 10 and 100 characters, e.g.:

```
Add token refresh support
Correct codeql workflow trigger
```

______________________________________________________________________

## 📄 PR Description

**What does it check?**
The PR body (description) must be at least **30 characters** long **and** reference a tracking item (JIRA ID or ISSUE ID).
An empty or one-line description makes it hard for reviewers to understand the context.

**Required tracking reference** — include **one** of the following. Type the line
**exactly as shown, without surrounding backticks**:

| Type             | Example                                                |
| ---------------- | ------------------------------------------------------ |
| JIRA ID          | JIRA ID : TESTAUTO-6039                                |
| JIRA ID          | JIRA ID - #330                                         |
| JIRA ID          | JIRA ID #330                                           |
| ISSUE ID         | ISSUE ID : TESTUTO-3334                                |
| ISSUE ID         | ISSUE ID - TESTAUTO-3433                               |
| ISSUE ID (link)  | ISSUE ID : https://github.com/abc/abc_repo/issues/1234 |
| Closing keyword  | Closes #10                                             |
| Closing keyword  | Fixes octo-org/octo-repo#100                           |
| Closing keyword  | Resolves: #123                                         |
| GitHub issue     | #123                                                   |
| GitHub issue URL | https://github.com/abc/abc_repo/issues/123             |

> **Note:** For `JIRA ID` / `ISSUE ID`, the separator is **optional** and may be `:` or `-` (`ISSUE ID #330`, `ISSUE ID : #330`, and `ISSUE ID - #330` all work). Each accepts a JIRA key (`PREFIX-<number>` — any project), a number (with or without `#`), or a link.
>
> **Closing keywords** (case-insensitive, optional colon): `close` / `closes` / `closed`, `fix` / `fixes` / `fixed`, `resolve` / `resolves` / `resolved` — followed by `#<number>` or `<org>/<repo>#<number>`.

**How to fix**
Edit the PR description and explain:

- *What* changed and *why*.
- A tracking reference from the table above (required) — e.g. a `JIRA ID :` / `ISSUE ID :` line, a `Closes #123`, or a plain `#123`.
- Testing steps if applicable.

______________________________________________________________________

## 📏 PR Size

**What does it check?**
Large PRs are hard to review thoroughly.

> **Note:** PR size limits are **not currently enforced** by `policy.yml`
> (there are no `max_files_changed` / `max_total_changes` /
> `max_single_file_changes` values configured). This section is guidance only
> and the bot does not fail a PR on size today.

**Recommended guidance**
Split your work into smaller, focused PRs. Each PR should ideally do one thing:

- One feature, one fix, or one refactor — not all three at once.
- Move large auto-generated or vendored file changes to a separate PR.

______________________________________________________________________

## ⛔ Forbidden Files

**What does it check?**
Certain file types must never be committed to the repository because they can expose secrets or introduce security risks.

| Pattern                                                  | Reason                                                         |
| -------------------------------------------------------- | -------------------------------------------------------------- |
| `**/*.pem`                                               | TLS/SSL certificates — must not be stored in source control    |
| `**/*.key`                                               | Private keys — must not be stored in source control            |
| `**/.env`                                                | Environment files — often contain secrets/passwords            |
| `**/*.exe`                                               | Windows executables — binary blobs with no review value        |
| `**/*.crt`, `**/*.cer`, `**/*.der`                       | Certificates — must not be committed                           |
| `**/*.p12`, `**/*.pfx`                                   | Keystores / certificate bundles — contain secrets              |
| `**/*.csr`                                               | Certificate signing requests — should not be in source control |
| `**/id_rsa`, `**/id_dsa`, `**/id_ecdsa`, `**/id_ed25519` | SSH private keys                                               |
| `**/*.gpg`, `**/*.asc`                                   | GPG keys / signatures — must not be committed                  |

**How to fix**
Remove the file from your commit:

```bash
git rm --cached path/to/secret.pem
echo "*.pem" >> .gitignore
git commit --amend
```

If a secret was already committed, rotate it immediately and follow your organization's incident response process.

______________________________________________________________________

## 🧪 Unit Test

**What does it check?**
PRs that change real source code must include at least one accompanying unit test.

**Rules**

- **Doc / config-only PRs are exempt.** If your PR only touches files like
  `.md`, `.txt`, `.yml`, `.yaml`, `.ini`, the check **passes automatically** — no test required.
- **Code PRs require a test.** If your PR changes source files such as
  `.py`, `.cpp`, `.cc`, `.c`, `.h`, `.js`, `.ts`, `.go`, `.java`, it must also
  include changes to a test file (a new test, or edits to an existing one).

**What counts as a test file?**

- Basename matches one of: `test_*`, `testing_*`, `*_test.*`, `*_tests.*`, or `Test*`
  - ✅ `test_parser.py`, `testing_parser.py`, `parser_test.cpp`, `parser_tests.cpp`, `TestUtils.cpp`
  - ❌ `test.py` (does NOT have the `test_` prefix)

| Pattern     | Example             |
| ----------- | ------------------- |
| `test_*`    | `test_parser.py`    |
| `testing_*` | `testing_parser.py` |
| `*_test.*`  | `parser_test.cpp`   |
| `*_tests.*` | `parser_tests.cpp`  |
| `Test*`     | `TestUtils.cpp`     |

**Path-based recognition**
Any file located under a `test/gtest/` directory is also treated as a unit
test, regardless of its filename — e.g.
`projects/miopen/test/gtest/unit_conv_solver_ConvWinoRageRxS.cpp`.

**How to fix**
Add a unit test for the code you changed, named `test_<something>`:

```bash
# example for Python
touch tests/test_my_feature.py
```

______________________________________________________________________

## 🔎 pre-commit

**What does it check?**
All pre-commit hooks defined in `.pre-commit-config.yaml` must pass.
This typically includes linting, formatting (black, isort), and other code-quality checks.

**How to fix**
Run the checks locally, let them auto-fix where possible, then commit the result:

```bash
python -m pip install pre-commit
pre-commit install
pre-commit run --all-files --show-diff-on-failure
git add -u
git commit -m "chore: apply pre-commit fixes"
```

______________________________________________________________________

## 🔎 CodeQL

**What does it check?**
GitHub's [CodeQL](https://codeql.github.com/) static-analysis engine scans the code added in this PR for known security vulnerabilities.
The bot fails this check when CodeQL reports **critical**, **high**, or **error**-severity alerts.

Common findings include:

| Alert                       | Meaning                                                   |
| --------------------------- | --------------------------------------------------------- |
| `py/command-line-injection` | User input passed unsanitised into a shell command        |
| `py/sql-injection`          | User input concatenated into a SQL query                  |
| `py/flask-debug`            | Flask app started with `debug=True` on a public interface |
| `js/xss`                    | User input rendered unescaped into HTML                   |

**How to fix**

1. Open **Security → Code scanning alerts** on the repository page.
1. Read the alert details and the suggested fix.
1. Apply the fix (validate/sanitise input, use parameterised queries, disable debug mode, etc.).
1. Push the fix — CodeQL will re-run and the alert will be resolved.

> **Tip:** GitHub Advanced Security AI comments directly on the offending line with a suggested code change.

______________________________________________________________________

## 🌿 Bump PRs (Automated Dependency Updates)

**What is a "Bump PR"?**

A **Bump PR** is an automated pull request that updates dependencies (e.g. from Dependabot or a bot like `assistant-librarian`). These PRs are routine, high-volume, and do not follow the standard PR conventions.

**Why did my Bump PR skip policy checks?**

When a PR is detected as a bump update from a configured bot account (e.g. `@assistant-librarian[bot]`), **all policy checks are auto-approved**. This includes:

- Branch name validation
- PR title (length) check
- JIRA/ISSUE ID reference requirement
- Unit test requirement
- And all other policies

This keeps automated bots from being blocked by human-oriented policy gates and prevents spam of "Not ready to Review" labels.

**How does the bot know it's a Bump PR?**

The PR author's login is checked against a configured list of bump bot accounts. Currently recognized:

- `assistant-librarian` (and `assistant-librarian[bot]`)
- `systems-assistant` (and `systems-assistant[bot]`)

If a different bot opens dependency-bump PRs in your repo, request that the maintainers add it to `bump_bot_authors` in `policy.yml`.

______________________________________________________________________

## General Questions

**Why did my PR get the "Not ready to Review" label?**

The label is added when:

1. **Unit Test check fails** — your PR changes source code but has no accompanying test file.
1. **JIRA/ISSUE ID reference is missing** — your PR description does not include a tracking reference.

All other policy failures (branch name, title format, description length, forbidden files, etc.) do not add the label; they are still reported in the table but do not block the PR.

**What is the "Not ready to Review" label?**

When **PR Title/Description**, **Unit Test**, or **Forbidden Files** fails, the bot adds a **`Not ready to Review`** label to the PR so it is clearly gated.
The label is removed automatically once all policy checks pass.
Other failures (Branch Name, PR Size, Draft PR, pre-commit, CodeQL) do **not** add the label.

**How are pre-commit and CodeQL shown?**

These run as separate CI workflows. The bot waits for them and folds their results into the same table — `pre-commit` and a single combined `CodeQL` row. The CodeQL row fails if CodeQL reports any error / critical / high severity alert.

**The bot timed out — what do I do?**

If `pre-commit` or CodeQL takes longer than 15 minutes, the bot times out.
Push an empty commit to re-trigger the workflow or close and reopen PR:

```bash
git commit --allow-empty -m "ci: retrigger policy check"
git push
```

**How do I re-run the bot after fixing issues?**

Push any commit (including `--allow-empty`) to the PR branch.
The `synchronize` event triggers a fresh policy check automatically.

______________________________________________________________________

## 🙋 Wish to Override the Policy Process and get unblocked?

Contact CODEOWNERS or supporters channel - (DevOps - Support or Help)

## 🙋 For any policy related feedback?

please reach out to the **ROCm Policy Council**.

📧 **Drop a mail to:** `rocm-repo-policy@amd.com` (ROCm Policy Council DLL)

Include your PR link, the check(s) you want overridden, and a short
justification so the council can review your request.

______________________________________________________________________
