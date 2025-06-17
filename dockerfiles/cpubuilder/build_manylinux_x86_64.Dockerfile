# This dockerfile builds automatically upon push to the main branch. It can be built
# interactively for testing via:
#   docker buildx build --file dockerfiles/cpubilder/build_manylinux_x86_64.Dockerfile dockerfiles/cpubuilder
# This will print a SHA image id, which you can run with (or equiv):
#   sudo docker run --rm -it --entrypoint /bin/bash <<IMAGE>>
#
# To build and push to a test branch, create a pull request on a branch named:
#   docker*
# We build our portable linux releases on the manylinux (RHEL-based)
# images, with custom additional packages installed. We switch to
# new upstream versions as needed.
FROM quay.io/pypa/manylinux_2_28_x86_64@sha256:634656edbdeb07f955667e645762ad218eefe25f0d185fef913221855d610456

# Prepend therock-tools to PATH for all subsequent steps
ENV PATH="/usr/local/therock-tools/bin:$PATH"

######## Python setup #######
# These images come with multiple python versions. We pin one for
# default use.
ENV PATH="/opt/python/cp312-cp312/bin:${PATH}"

######## Pip Packages ########
RUN pip install CppHeaderParser==2.7.4 meson==1.7.0 tomli==2.2.1 PyYAML==6.0.2

######## Repo ########
RUN curl https://storage.googleapis.com/git-repo-downloads/repo > /usr/local/bin/repo && chmod a+x /usr/local/bin/repo

######## CCache ########
WORKDIR /install-ccache
COPY install_ccache.sh ./
RUN ./install_ccache.sh "4.11.2" && rm -rf /install-ccache

######## CMake ########
WORKDIR /install-cmake
ENV CMAKE_VERSION="3.25.2"
COPY install_cmake.sh ./
RUN ./install_cmake.sh "${CMAKE_VERSION}" && rm -rf /install-cmake

######## Ninja ########
WORKDIR /install-ninja
ENV CMAKE_VERSION="1.12.1"
COPY install_ninja.sh ./
RUN ./install_ninja.sh "${CMAKE_VERSION}" && rm -rf /install-ninja

######## AWS CLI ######
WORKDIR /install-awscli
COPY install_awscli.sh ./
RUN ./install_awscli.sh && rm -rf /install-awscli

######## Yum Packages #######
# TODO: Figure out why gcc-toolset-12-libatomic-devel doesn't install with the
# rest of the dev toolset.
RUN yum install -y epel-release && \
    yum install -y \
        gcc \
        gcc-c++ \
        make \
        gcc-toolset-12-libatomic-devel \
        patchelf \
        vim-common \
        git-lfs && \
    yum clean all && \
    rm -rf /var/cache/yum

######## Installing Google test #######
WORKDIR /install-googletest
ENV GOOGLE_TEST_VERSION="1.16.0"
COPY install_googletest.sh ./
RUN ./install_googletest.sh "${GOOGLE_TEST_VERSION}" && rm -rf /install-googletest

######## GIT CONFIGURATION ########
# Git started enforcing strict user checking, which thwarts version
# configuration scripts in a docker image where the tree was checked
# out by the host and mapped in. Disable the check.
# See: https://github.com/openxla/iree/issues/12046
# We use the wildcard option to disable the checks. This was added
# in git 2.35.3
RUN git config --global --add safe.directory '*'
