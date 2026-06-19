# TheRock PR Bot — Policy FAQ Document

This document explains what each policy check means, why it exists, and how to fix a failure.

______________________________________________________________________

## 🌿 Branch Name

**What does it check?**
Your branch name must follow the agreed naming convention so PRs are easy to trace back to a contributor and topic.

**Allowed formats**

| Pattern                                  | Example                                               |
| ---------------------------------------- | ----------------------------------------------------- |
| `users/<username>/<feature-or-bug-name>` | `users/chi/ucicd_setup_visible_devices`               |
| `<namespace>/<feature-or-bug-name>`      | `samathew121/docker`                                  |
| `<single-segment-name>`                  | `bump-rocm-libraries-936a6c7`                         |
| `dependabot/<ecosystem>/<dependency>`    | `dependabot/github_actions/github-actions-3dfd2199fc` |
| `revert-<pr-number>-<original-branch>`   | `revert-5217-users/derobins/add_hipfile_support`      |

Rules:

- `username` — lowercase letters, digits, hyphens only.
- `branch name` — lowercase letters, digits, hyphens, underscores only.

**How to fix**
Rename your branch before opening the PR:

```bash
git branch -m old-name users/<your-username>/<topic>
git push origin -u users/<your-username>/<topic>
```

______________________________________________________________________

## 📝 PR Title

**What does it check?**
The PR title must follow **Conventional Commits** style so the changelog and release notes can be generated automatically.

> **Note:** In the results table, the title and description checks are reported together as a single **PR Title/Description** row. Any title *or* description failure shows up there.

**Required format**

```
type(optional-scope): short description
```

**Allowed types**

| Type       | When to use                              |
| ---------- | ---------------------------------------- |
| `feat`     | A new feature                            |
| `fix`      | A bug fix                                |
| `docs`     | Documentation only changes               |
| `style`    | Formatting, whitespace — no logic change |
| `refactor` | Code restructure — no feature or fix     |
| `perf`     | Performance improvement                  |
| `test`     | Adding or fixing tests                   |
| `build`    | Build system or dependency changes       |
| `ci`       | CI / workflow changes                    |
| `chore`    | Maintenance tasks                        |
| `revert`   | Reverting a previous commit              |

**Length rules**

- Minimum: **10** characters
- Maximum: **80** characters

**Forbidden words** — titles containing `WIP` or `do not merge` are blocked.

**How to fix**
Edit the PR title on GitHub (top of the PR page → pencil icon) to match the format, e.g.:

```
feat(auth): add token refresh support
fix(ci): correct codeql workflow trigger
```

______________________________________________________________________

## 🚫 Draft PR

> **Status: 🔜 To Be Enabled.** This check is planned but **not yet enforced**.
> It currently shows as *To Be Enabled* in the results table and never fails a PR.

**What it will check (once enabled)**
Draft PRs (and PRs with `WIP` / `do not merge` in the title) will be blocked from passing policy, so reviews are only requested on finished work.

**How to prepare**
Click **"Ready for review"** on the GitHub PR page once your work is complete, and remove any `WIP` prefix from the title.

______________________________________________________________________

## 📄 PR Description

**What does it check?**
The PR body (description) must be at least **30 characters** long **and** reference a tracking item (JIRA ID or ISSUE ID).
An empty or one-line description makes it hard for reviewers to understand the context.

**Required tracking reference** — include **one** of the following lines:

| Field      | Example                                                  |
| ---------- | -------------------------------------------------------- |
| `JIRA ID`  | `JIRA ID : TESTAUTO-6039`                                |
| `JIRA ID`  | `JIRA ID - #330`                                         |
| `JIRA ID`  | `JIRA ID #330`                                           |
| `ISSUE ID` | `ISSUE ID : TESTUTO-3334`                                |
| `ISSUE ID` | `ISSUE ID - TESTAUTO-3433`                               |
| `ISSUE ID` | `ISSUE ID : https://github.com/abc/abc_repo/issues/1234` |

> **Note:** The separator is **optional** and may be `:` or `-` (`JIRA ID #330`, `JIRA ID : #330`, and `JIRA ID - #330` all work). Each field accepts a JIRA key (`PREFIX-<number>` — any project), a number (with or without `#`), or a link.

**How to fix**
Edit the PR description and explain:

- *What* changed and *why*.
- A `JIRA ID :` or `ISSUE ID :` line referencing the tracking item (required).
- Any related issue numbers (`Fixes #123`).
- Testing steps if applicable.

______________________________________________________________________

## 📏 PR Size

**What does it check?**
Large PRs are hard to review thoroughly. Three limits are enforced:

| Limit                   | Value | Reason                                         |
| ----------------------- | ----- | ---------------------------------------------- |
| Max files changed       | 50    | Avoids PRs that touch too many unrelated areas |
| Max total changes       | 2000  | Keeps the overall diff reviewable              |
| Max changes in one file | 700   | Flags files that may need splitting            |

**How to fix**
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
  add a test file.

**How a test file is recognised**

| Pattern    | Example          |
| ---------- | ---------------- |
| `test_*`   | `test_parser.py` |
| `*_test.*` | `parser_test.py` |

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

## General Questions

### What is the "Not ready to Review" label?

When **PR Title/Description**, **Unit Test**, or **Forbidden Files** fails, the bot adds a **`Not ready to Review`** label to the PR so it is clearly gated.
The label is removed automatically once all policy checks pass.
Other failures (Branch Name, PR Size, Draft PR, pre-commit, CodeQL) do **not** add the label.

### How are pre-commit and CodeQL shown?

These run as separate CI workflows. The bot waits for them and folds their results into the same table — `pre-commit` and a single combined `CodeQL` row. The CodeQL row fails if CodeQL reports any error / critical / high severity alert.

### The bot timed out — what do I do?

If `pre-commit` or CodeQL takes longer than 15 minutes, the bot times out.
Push an empty commit to re-trigger the workflow:

```bash
git commit --allow-empty -m "ci: retrigger policy check"
git push
```

### How do I re-run the bot after fixing issues?

Push any commit (including `--allow-empty`) to the PR branch.
The `synchronize` event triggers a fresh policy check automatically.
