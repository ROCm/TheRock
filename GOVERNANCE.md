# Governance model

ROCm is a software stack made up of a collection of drivers, development tools, and APIs that enable
GPU programming from the low-level kernel to end-user applications.

Components of ROCm that are inherited from external projects (such as
[LLVM](https://github.com/ROCm/llvm-project) and
[Kernel driver](https://github.com/ROCm/ROCK-Kernel-Driver)) follow their own
governance model and code of conduct. All other components of ROCm are governed by this
document.

## Governance

ROCm is led and managed by AMD.

We welcome contributions from the community. Our maintainers review all proposed changes to
ROCm.

## Roles

* **Maintainers** are responsible for their designated component and repositories.
* **Contributors** provide input and suggest changes to existing components.

### Maintainers

Maintainers are appointed by AMD. They are able to approve changes and can commit to our
repositories. They must use pull requests (PRs) for all changes.

You can find the list of maintainers in the CODEOWNERS file of each repository. Code owners differ
between repositories.

### Contributors

If you're not a maintainer, you're a contributor. We encourage the ROCm community to contribute in
several ways:

* Help other community members by posting questions or solutions in the GitHub discussion forum of
  the relevant repository.
* Notify us of a bug by filing an issue report in the GitHub issue tracker of the affected
  repository.
* Improve our documentation by submitting a PR to the repository that owns the affected content.
* Improve the code base (for smaller or contained changes) by submitting a PR to the component.
* Suggest larger features by adding to the *Ideas* category in the GitHub discussion forum of the
  relevant repository.

ROCm spans many repositories, and each component keeps its issues, discussions, and pull requests in
its own repository. For the build and release system that ties ROCm together, use
[TheRock](https://github.com/ROCm/TheRock) — its
[issues](https://github.com/ROCm/TheRock/issues) and
[discussions](https://github.com/ROCm/TheRock/discussions) are the starting point when you are unsure
which repository owns a topic.

## Common contribution guidelines

These expectations apply to all ROCm repositories governed by this document. Each component may add
project-specific steps in its own `CONTRIBUTING.md`; those instructions extend, but do not replace,
the guidance here.

* **Use pull requests.** All changes land through pull requests reviewed by the maintainers of the
  affected repository.
* **Target the default branch.** Open pull requests against each repository's default integration
  branch unless its `CONTRIBUTING.md` says otherwise.
* **Link an issue.** Associate each pull request with a GitHub issue so reviewers have context and
  changes are traceable.
* **Follow the code of conduct.** All participation is subject to the [Code of conduct](#code-of-conduct)
  below.
* **License your contribution.** By opening a pull request, you agree to license your contribution
  under the terms of the `LICENSE` file in the corresponding repository. Different repositories may
  use different licenses; see the [ROCm licensing](https://rocm.docs.amd.com/en/latest/about/license.html)
  page.

### Pull requests

When you create a pull request, target the repository's default integration branch. The default
branch differs between repositories, so check each repository's `CONTRIBUTING.md` if you are unsure.

When creating a PR, use the following process. Note that each repository may include additional,
project-specific steps. Refer to each repository's PR process for any additional steps.

* Identify the issue you want to fix.
* Target the default branch for integration.
* Ensure your change builds successfully.
* Run the relevant test suites and include evidence of a successful run in your PR.
* Do not break existing test cases.
* Merge new functionality only with accompanying tests. If your PR adds a feature, provide an
  application or test so we can confirm the feature works and continues to be valid.
* Aim for good test coverage.
* Submit your PR and work with the reviewer or maintainer to get it approved.
* Once approved, the PR is integrated through the repository's CI and merge process, as coordinated
  by the maintainer.
* We'll inform you once your change is committed.

For the contribution process specific to a repository, refer to that repository's `CONTRIBUTING.md`.
For TheRock, see its [contribution guidelines](CONTRIBUTING.md).

## Code of conduct

To engage with any AMD ROCm component that is hosted on GitHub, you must abide by the
[GitHub community guidelines](https://docs.github.com/en/site-policy/github-terms/github-community-guidelines)
and the
[GitHub community code of conduct](https://docs.github.com/en/site-policy/github-terms/github-community-code-of-conduct).

## Reporting security vulnerabilities

> [!IMPORTANT]
> Do **not** report security vulnerabilities publicly through GitHub issues. Instead, report them
> through the [AMD Product Security website](https://www.amd.com/en/resources/product-security.html).

Each repository's `SECURITY.md` describes its security reporting policy. For TheRock, see
[SECURITY.md](SECURITY.md).
