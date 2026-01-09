# Dockerfiles for TheRock

This directory is home to the Dockerfiles we use as part of ROCm development, as
well as supporting scripts.

## Available Dockerfiles

### `build_manylinux_*.Dockerfile`

| Source .Dockerfile                                                                 | Published package                                                                  |
| ---------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| [`build_manylinux_x86_x64.Dockerfile`](build_manylinux_x86_64.Dockerfile)          | https://github.com/ROCm/TheRock/pkgs/container/therock_build_manylinux_x86_64      |
| [`build_manylinux_rccl_x86_64.Dockerfile`](build_manylinux_rccl_x86_64.Dockerfile) | https://github.com/ROCm/TheRock/pkgs/container/therock_build_manylinux_rccl_x86_64 |

These Dockerfiles are used to build ROCm, PyTorch, and other packages for
release across multiple Linux distributions. They are derived from
[manylinux](https://github.com/pypa/manylinux) Docker images which are based on
older RHEL derivatives, provide a minimal set of development libraries, and
support building binaries compatible with most Linux distributions that use
glibc 2.39 or greater.

Requirements for these files:

- Tools like gcc, CMake, ninja, etc. should generally be set to the minimum
  versions of those tools that we support.
- These Dockerfiles deliberately exclude development _packages_ for most
  libraries because their presence would create hidden dependencies on specific
  library versions that may not be binary-compatible across distributions.
  Instead, TheRock vendors all necessary dependencies and builds them from
  source with specific modifications to ensure they cannot conflict with system
  libraries.
  - Certain development _tools_ like texinfo and DVC are acceptable to install
    in the base build images, so long as they don't leak non-portable code into
    binaries.
  - See also
    [gist - "TheRock Dependency Auditing and Vendoring Guide"](https://gist.github.com/stellaraccident/866acb0cae35e9f59c0c67d2fe3773c1).
    <!-- TODO: Merge that gist into this repository somewhere -->

### `no_rocm_image_*.Dockerfile`

| Source .Dockerfile                                                             | Published package                                                        |
| ------------------------------------------------------------------------------ | ------------------------------------------------------------------------ |
| [`no_rocm_image_ubuntu24_04.Dockerfile`](no_rocm_image_ubuntu24_04.Dockerfile) | https://github.com/ROCm/TheRock/pkgs/container/no_rocm_image_ubuntu24_04 |

These Dockerfiles are used to test ROCm, PyTorch, and other packages. They
are derived from standard OS images (e.g. `FROM ubuntu:24.04`) and contain a
minimal extra set of development tools and environment settings necessary to
run project tests on GPU test machines. These images do NOT include a ROCm
install, as they are used from test workflows that begin by installing ROCm
(or other) packages produced by CI/CD workflows.

Requirements for these files:

- No system install of ROCm - workflows using these images are expected to
  install ROCm after starting a container.
- Minimal other system dependencies, especially development packages, for many
  of the same reasons as for the "build" Dockerfiles above. Test environments
  should be similar to user/production environments, so any extra dependencies
  that are added to these test Dockerfiles may hide real issues.
  - Some extra tools like `lit` for LLVM testing are acceptable and should be
    evaluated on a case-by-case basis.

<!-- ### Multi-distro rocm-runtime base Dockerfile -->

<!-- TODO: add docs together with https://github.com/ROCm/TheRock/pull/2572

base image for images that include ROCm and a base set of runtime dependencies

See https://github.com/ROCm/TheRock/pull/2572#discussion_r2631985736
-->

## Using published images

Images for these Dockerfiles are published as
[GitHub Packages](https://docs.github.com/en/packages) to the
[GitHub Container Registry](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry)
(`ghcr.io`). The full list of published packages can be viewed at
https://github.com/orgs/ROCm/packages?repo_name=TheRock.

Stable tags for these packages are typically `latest` and `main`, while other
tags may be used for test packages.

## Working on the Dockerfiles themselves

### Testing and debugging

<!-- TODO: document how to build a docker image locally -->

<!-- TODO: document how to run a docker image locally (mount options, entrypoints, etc.) -->

### Automated image publishing

Images are automatically built and published to GitHub Packages (`ghcr.io`) by
GitHub Actions after pushes to the associated `.Dockerfile` on the `main` or
`stage/docker/**` branches. These actions can also be triggered manually from
other branches
[using workflow_dispatch](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-run-a-workflow).

The common
[`.github/workflows/publish_dockerfile.yml`](/.github/workflows/publish_dockerfile.yml)
workflow is used by other `publish_*.yml` workflows:

| Workflow file                                                                                                             | Workflow run history                                                                                                                                   |
| ------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [`.github/workflows/publish_build_manylinux_x86_64.yml`](/.github/workflows/publish_build_manylinux_x86_64.yml)           | [actions/workflows/publish_build_manylinux_x86_64.yml](https://github.com/ROCm/TheRock/actions/workflows/publish_build_manylinux_x86_64.yml)           |
| [`.github/workflows/publish_build_manylinux_rccl_x86_64.yml`](/.github/workflows/publish_build_manylinux_rccl_x86_64.yml) | [actions/workflows/publish_build_manylinux_rccl_x86_64.yml](https://github.com/ROCm/TheRock/actions/workflows/publish_build_manylinux_rccl_x86_64.yml) |
| [`.github/workflows/publish_no_rocm_image_ubuntu24_04.yml`](/.github/workflows/publish_no_rocm_image_ubuntu24_04.yml)     | [actions/workflows/publish_no_rocm_image_ubuntu24_04.yml](https://github.com/ROCm/TheRock/actions/workflows/publish_no_rocm_image_ubuntu24_04.yml)     |

Tags for built docker images are set based on the branch name pattern:

| Branch name pattern   | Tag pattern    |
| --------------------- | -------------- |
| `main`                | `latest`       |
| `stage/docker/SUFFIX` | `stage-SUFFIX` |
| `OTHER_NAME`          | `OTHER_NAME`   |

### Updating images used by GitHub Actions workflows

The general sequence for updating an image is this:

1. Make any changes to the relevant Dockerfiles and/or scripts
1. Build and test images locally
   (see [Testing and debugging](#testing-and-debugging))
1. Build and test images on CI
   1. Push the Dockerfile changes to a shared branch and build test packages
      (see [Automated image publishing](#automated-image-publishing))

   1. Edit sha256 pins across the project to use the staging packages, like so:

      ```diff
          container:
      -     image: ghcr.io/rocm/therock_build_manylinux_x86_64@sha256:4af52d56d91ef6ef8b7d0a13c6115af1ab2c9bf4a8a85d9267b489ecb737ed25
      +     image: ghcr.io/rocm/therock_build_manylinux_x86_64@sha256:6e8242d347af7e0c43c82d5031a3ac67b669f24898ea8dc2f1d5b7e4798b66bd
      ```

   1. Trigger test CI/CD jobs (e.g. by creating a draft PR with those changes)
1. Create a pull request containing the Dockerfile changes and go through code
   review
   - For example: https://github.com/ROCm/TheRock/pull/2365
1. Wait for new packages to be published
1. Create a second pull request editting the sha256 pins across the project to
   use the new packages (from the `latest` tag)
   - For example: https://github.com/ROCm/TheRock/pull/2390
