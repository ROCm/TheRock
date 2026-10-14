# Security Policy

## Reporting a vulnerability

If you have discovered a security vulnerability in TheRock,
do NOT report it publicly using a GitHub issue. Instead, report the issue
through the [AMD Product Security website](https://www.amd.com/en/resources/product-security.html).

## Automated security scanning

Alongside the reporting path above, this repository is scanned automatically.
The scanners are not implemented here: TheRock calls the shared
[`ROCm/rocm-security-gh`](https://github.com/ROCm/rocm-security-gh)
`security-baseline.yml` reusable workflow, so the scanner versions and
behavior are maintained centrally for ROCm, and this repository supplies only
its own configuration under
[`build_tools/scan_tools/`](/build_tools/scan_tools/).

| Scanner                                          | Looks for                                                   |
| ------------------------------------------------ | ----------------------------------------------------------- |
| [gitleaks](https://github.com/gitleaks/gitleaks) | Secrets and credentials in tracked files and git history    |
| [bandit](https://bandit.readthedocs.io/)         | Unsafe patterns in our Python scripts                       |
| [zizmor](https://docs.zizmor.sh/)                | GitHub Actions workflow vulnerabilities                     |
| [trivy](https://trivy.dev/)                      | Dockerfile misconfigurations and dependency vulnerabilities |
| [CodeQL](https://codeql.github.com/)             | Semantic code analysis of our C/C++ and Python              |

Two workflows run them, and where a finding shows up depends on which one
produced it:

- [`security_scan_pr.yml`](/.github/workflows/security_scan_pr.yml) runs on
  every pull request, scoped to what that pull request changed. It reports in
  the job summary and a build artifact, and deliberately does not upload to the
  Security tab, so pull requests from forks behave the same as those from
  branches in this repository.
- [`security_scan_weekly.yml`](/.github/workflows/security_scan_weekly.yml)
  runs on a schedule across the whole repository and uploads SARIF to this
  repository's Security tab, which is the authoritative view of the current
  state.

Only TheRock's own tree is in scope. The ROCm subprojects it builds are git
submodules and are scanned in their own repositories.

> [!IMPORTANT]
> A finding from these scanners is not a vulnerability report. If a scanner
> finding turns out to be an exploitable vulnerability in shipped ROCm
> software, report it through the AMD Product Security website above rather
> than in a public issue or pull request.

Contributors can run every scanner locally against the same configuration CI
uses; see
[the security scanners section in `CONTRIBUTING.md`](/CONTRIBUTING.md#security-scanners).
For how this fits with the project's other testing, see
[the security scanning section in `TESTING.md`](/TESTING.md#therock-feature-area-security-scanning).
