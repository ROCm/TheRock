# This dockerfile builds automatically upon push to the main branch. It can be built
# interactively for testing via:
#   docker buildx build --file dockerfiles/build_manylinux_x86_64.Dockerfile dockerfiles/
# This will print a SHA image id, which you can run with (or equiv):
#   sudo docker run --rm -it --entrypoint /bin/bash <<IMAGE>>
#
# To build and push to a test branch, create a pull request on a branch named:
#   docker*
# We build our portable linux releases on the manylinux (RHEL-based)
# images, with custom additional packages installed. We switch to
# new upstream versions as needed.
FROM quay.io/pypa/manylinux_2_28_x86_64@sha256:d632b5e68ab39e59e128dcf0e59e438b26f122d7f2d45f3eea69ffd2877ab017

######## Python and CMake setup #######
# These images come with multiple python versions. We pin one for
# default use.
# Prepend therock-tools to PATH
ENV PATH="/usr/local/therock-tools/bin:/opt/python/cp312-cp312/bin:${PATH}"

######## Pip Packages ########
RUN pip install --upgrade pip setuptools==69.1.1 wheel==0.42.0 && \
pip install CppHeaderParser==2.7.4 meson==1.7.0 tomli==2.2.1 PyYAML==6.0.2

######## Repo ########
RUN curl https://storage.googleapis.com/git-repo-downloads/repo > /usr/local/bin/repo && chmod a+x /usr/local/bin/repo

######## CCache ########
WORKDIR /install-ccache
COPY install_ccache.sh ./
RUN ./install_ccache.sh "4.11.2" && rm -rf /install-ccache

######## CMake ########
WORKDIR /install-cmake
ENV CMAKE_VERSION="3.27.9"
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

######## Installing Google test #######
WORKDIR /install-googletest
ENV GOOGLE_TEST_VERSION="1.16.0"
COPY install_googletest.sh ./
RUN ./install_googletest.sh "${GOOGLE_TEST_VERSION}" && rm -rf /install-googletest

######## Yum Packages #######
# We are pinning to gcc-toolset-12 until it is safe to upgrade. The latest
# manylinux containers use gcc-toolset-14 or later, which is not yet compatible
# with the LLVM that ROCm builds. This can be upgraded when clang-21 is used.
RUN yum install -y epel-release && \
    yum remove -y gcc-toolset* && \
    yum install -y \
      gcc-toolset-12-binutils \
      gcc-toolset-12-gcc \
      gcc-toolset-12-gcc-c++ \
      gcc-toolset-12-gcc-gfortran \
      gcc-toolset-12-libatomic-devel \
      gcc-toolset-12-libstdc++-devel \
      patchelf \
      vim-common \
      git-lfs \
      m4 \
      flex \
      libevent-devel \
      hwloc-devel \
      numactl-devel \
      pkgconfig \
      zip \
      unzip \
    && yum clean all && \
    rm -rf /var/cache/yum

######## DVC via pip ######
# dvc's rpm package includes .so dependencies built against glib 2.29
# settling for pip install for now, but it installs modules not needed for dvc pull
# more dvc features may be used in upcoming sequenced builds
RUN pip install dvc[s3]==3.62.0 && \
    which dvc && dvc --version || true

######## Enable GCC Toolset and verify ########
# This is a subset of what is typically sourced in the gcc-toolset enable
# script.
# -- Predefine variables to avoid Dockerfile linting warnings --
# Docker requires environment variables to be defined before reuse.
ENV LIBRARY_PATH=""
ENV PATH="/opt/rh/gcc-toolset-12/root/usr/bin:${PATH}"
ENV LIBRARY_PATH="/opt/rh/gcc-toolset-12/root/usr/lib64:${LIBRARY_PATH}"
ENV LD_LIBRARY_PATH="/opt/rh/gcc-toolset-12/root/usr/lib64:${LD_LIBRARY_PATH}"

######## Enable GCC Toolset and verify ########
RUN which gcc && gcc --version && \
    which g++ && g++ --version && \
    which clang++ || true

######## GIT CONFIGURATION ########
# Git started enforcing strict user checking, which thwarts version
# configuration scripts in a docker image where the tree was checked
# out by the host and mapped in. Disable the check.
# See: https://github.com/openxla/iree/issues/12046
# We use the wildcard option to disable the checks. This was added
# in git 2.35.3
RUN git config --global --add safe.directory '*'

######## Open MPI 5.0.8 (from source) ########
# OpenMPI is currently not vendored into TheRock and temorarily installed
ENV OMPI_VER=5.0.8
ENV OMPI_PREFIX=/opt/openmpi-${OMPI_VER}

# Copy and execute the build/install portion
COPY install_openmpi.sh /tmp/install_openmpi.sh
RUN /tmp/install_openmpi.sh && rm -f /tmp/install_openmpi.sh

# Expose Open MPI by default
ENV PATH="${OMPI_PREFIX}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${OMPI_PREFIX}/lib:${LD_LIBRARY_PATH}"
ENV PKG_CONFIG_PATH="${OMPI_PREFIX}/lib/pkgconfig:${PKG_CONFIG_PATH}"

RUN which mpicc && mpirun && \
    mpirun --version || true
