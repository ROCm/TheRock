# Contributing to TheRock

We are enthusiastic about contributions to our code and documentation.

The project is still in its early days, so please feel free to file issues where documentation or
functionality is lacking or even better volunteer to help contribute to help close these gaps!

> [!TIP]
> For contribution guidelines to other parts of ROCm outside of TheRock, please see
> [ROCm/CONTRIBUTING.md](https://github.com/ROCm/ROCm/blob/develop/CONTRIBUTING.md).

## Quick start for contributors

```bash
# 1. Fork on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/TheRock.git
cd TheRock

# 2. Add upstream remote
git remote add upstream https://github.com/ROCm/TheRock.git

# 3. Create feature branch
git checkout -b users/YOUR_USERNAME/feature-name

# 4. Setup environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install pre-commit
pre-commit install

# 5. Fetch submodules
python3 ./build_tools/fetch_sources.py

# 6. Make changes, then verify
pre-commit run --all-files

# 7. Commit and push
git add <files>
git commit -m "Description of change"
git push origin users/YOUR_USERNAME/feature-name

# 8. Create PR on GitHub targeting 'main' branch
```

## Developer policies

### Governance

This is currently covered by the
[ROCm Project Governance](https://github.com/ROCm/ROCm/blob/develop/GOVERNANCE.md),
which also defines the code of conduct.

### Communication channels

TheRock is focused on making it easy for developers to contribute. In order to facilitate this,
the source of truth for all issue tracking, project planning, and code contributions are in GitHub and
we leverage an open source stack for all development tools and infrastructure so that it they can be
easily leveraged in any fork.

We are also active on the [AMD Developer Community Discord Server](https://discord.com/invite/amd-dev)
in channels like `#therock-contributors` and `#rocm-build-install-help`.

## Development workflows

### Issue tracking

Before filing a new issue, please search through
[existing issues](https://github.com/ROCm/TheRock/issues) to make sure your issue hasn't
already been reported.

General issue guidelines:

- Use your best judgement for issue creation. If your issue is already listed, upvote the issue and
  comment or post to provide additional details, such as how you reproduced this issue.
- If you're not sure if your issue is the same, err on the side of caution and file your issue.
  You can add a comment to include the issue number (and link) for the similar issue. If we evaluate
  your issue as being the same as the existing issue, we'll close the duplicate.
- When filing an issue, be sure to provide as much information as possible, including script output so
  we can collect information about your configuration. This helps reduce the time required to
  reproduce your issue.
- Check your issue regularly, as we may require additional information to successfully reproduce the
  issue.

### New feature development

Discussion about new features is welcome via

- Filing a [GitHub issue](https://github.com/ROCm/TheRock/issues)
- Reaching out [on Discord](https://discord.com/invite/amd-dev)
- Posting a [GitHub discussion](https://github.com/ROCm/TheRock/discussions) (discussions are not as active)

### Pull requests

When you create a pull request, you should target the *main* branch.

- Identify the issue you want to fix
- Target the main branch
- Ensure your code has all workflows pass
- Submit your PR and work with the reviewer or maintainer to get your PR approved
  - If you don't know who to add to a PR as a maintainer, please review the git history to see recently approved PRs in the same file or folder.

> [!IMPORTANT]
> By creating a PR, you agree to allow your contribution to be licensed under the
> terms of the [LICENSE](LICENSE) file.

### pre-commit checks

We use [pre-commit](https://pre-commit.com/) to run automated "hooks" like lint
checks and formatters on each commit. See the list of hooks we currently
run at [`.pre-commit-config.yaml`](.pre-commit-config.yaml). Contributors are
encouraged to download pre-commit and run it on their commits before sending
pull requests for review.

> [!TIP]
> The pre-commit tool can also be "installed" as a git hook to run automatically
> on every `git commit`.

For example:

```bash
# Download (note: this is already included in requirements.txt).
pip install pre-commit

# Run locally on staged files.
pre-commit run

# Run locally on all files.
pre-commit run --all-files

# Install git hook.
pre-commit install
```

### Branch creation and naming

If creating a branch in the shared repository (and not a fork), prefer to choose
a branch name following one of these patterns:

- `users/[USERNAME]/[feature-or-bug-name]`
- `shared/[feature-or-bug-name]`

These naming schemes allow for long-lived branches to be more easily sorted and
possibly cleaned up by repository maintainers.

List of long-lived branches:

- [`main`](https://github.com/ROCm/TheRock/tree/main)
- [`release/*`](https://github.com/ROCm/TheRock/branches/all?query=release%2F)
- [`compiler/amd-staging`](https://github.com/ROCm/TheRock/tree/compiler/amd-staging)
- [`compiler/amd-mainline`](https://github.com/ROCm/TheRock/tree/compiler/amd-mainline)

> [!TIP]
> Most developer workflows are compatible with pull requests coming from forks.
> Some good reasons to create branches in the shared repository are:
>
> - Collaborating on changes on a shared branch
> - Stacking a series of pull requests by setting the base branches for each PR
> - Triggering custom workflows such as "dev" release builds using
>   [workflow_dispatch](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-run-a-workflow)
>   and running on our self-hosted GitHub Actions runners

### Style guides

We have project-wide style guides with recommendations to follow at
[`docs/development/style_guides/`](/docs/development/style_guides/):

- [Bash Style Guide](/docs/development/style_guides/bash_style_guide.md)
- [CMake Style Guide](/docs/development/style_guides/cmake_style_guide.md)
- [GitHub Actions Style Guide](/docs/development/style_guides/github_actions_style_guide.md)
- [Python Style Guide](/docs/development/style_guides/python_style_guide.md)

Core principles across all languages:

- Optimize for readability and debuggability
- Explicit is better than implicit
- Follow DRY, YAGNI, and KISS principles
- Write portable code (Linux/Windows)

### Testing your changes

Before submitting a PR, verify your changes build and pass tests:

```bash
# Rebuild affected component
ninja -C build <component>+build

# Run tests
ctest --test-dir build

# Or run specific test binary
LD_LIBRARY_PATH=build/dist/rocm/lib build/dist/rocm/bin/<test_binary>
```

See the [Development Guide](docs/development/development_guide.md) for more details on
the build system and component development.

## CI/CD pipeline

Pull requests trigger the following workflows:

| Workflow         | Purpose                          |
| ---------------- | -------------------------------- |
| `ci.yml`         | Main CI (Linux + Windows builds) |
| `pre-commit.yml` | Code style validation            |

### Default CI builds

- Linux: gfx94X, gfx110X
- Windows: gfx110X

### Triggering additional builds

Add labels to your PR to trigger builds for additional GPU families:

- `gfx120X-linux` - Build for RDNA4 on Linux
- `gfx94X-linux` - Build for CDNA3 on Linux

See `.github/workflows/ci.yml` for the full list of supported labels.

## Contribution opportunities

### Documentation

- Fill gaps in setup guides (Fedora instructions needed at `docs/environment_setup_guide.md`)
- Improve the FAQ at `docs/faq.md`
- Add troubleshooting content

### GPU architecture support

Check [ROADMAP.md](ROADMAP.md) for architectures that need testing:

- RDNA2 (gfx1030-gfx1036) - not yet tested
- Older GCN architectures

### Build system

- CMake improvements
- Python tooling enhancements
- CI/CD workflow improvements

### Testing

- Add test coverage
- Improve test infrastructure
- Hardware-specific testing

## Key files reference

| File                               | Purpose                       |
| ---------------------------------- | ----------------------------- |
| `CONTRIBUTING.md`                  | This file                     |
| `ROADMAP.md`                       | GPU support roadmap           |
| `CLAUDE.md`                        | Development quick reference   |
| `.pre-commit-config.yaml`          | Pre-commit hook configuration |
| `docs/development/style_guides/`   | Style guides                  |
| `docs/development/build_system.md` | Build architecture details    |
