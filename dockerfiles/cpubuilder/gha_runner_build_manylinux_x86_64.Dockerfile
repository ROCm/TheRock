# This dockerfile uses a multi-stage build, where it is layered as:
#       GitHub's actions-runner (base layer)
#       therock_build_manylinux_x86_64 (next layer)
#
# This Dockerfile is used on `azure-linux-scale-rocm` runners in the Kubernetes ARC setup.

FROM ghcr.io/actions/actions-runner:latest AS runner

FROM ghcr.io/rocm/therock_build_manylinux_x86_64@sha256:543ba2609de3571d2c64f3872e5f1af42fdfa90d074a7baccb1db120c9514be2

RUN groupadd -g 1001 runner && useradd -m -u 1001 -g runner runner

COPY --from=runner /home/runner /home/runner

RUN chown -R runner:runner /home/runner && chmod +x /home/runner/run.sh

RUN chown -R runner:runner /home/runner && chmod +x /home/runner/run.sh

# Set up runner environment
USER runner
WORKDIR /home/runner

ENV RUNNER_HOME=/home/runner
ENV PATH=$PATH:/home/runner/bin
ENV RUNNER_WORKDIR=/home/runner/_work
