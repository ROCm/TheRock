# This dockerfile uses a multi-stage build, initially using GitHub's action runner image
# then using the manylinux package.
# In order for the GitHub Action runner to activate, we need to copy the /home/runner directory

FROM ghcr.io/actions/actions-runner:latest AS runner

FROM ghcr.io/rocm/therock_build_manylinux_x86_64@sha256:543ba2609de3571d2c64f3872e5f1af42fdfa90d074a7baccb1db120c9514be2

COPY --from=runner /home/runner/ /home/runner/
