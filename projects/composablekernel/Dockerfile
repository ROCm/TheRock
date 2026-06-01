
FROM ubuntu:24.04
ARG DEBIAN_FRONTEND=noninteractive
ARG ROCMVERSION=7.1.1
ARG compiler_version=""
ARG compiler_commit=""
ARG CK_SCCACHE=""
ARG DEB_ROCM_REPO=http://repo.radeon.com/rocm/apt/.apt_$ROCMVERSION/
ENV APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=DontWarn
ENV DEBIAN_FRONTEND=noninteractive

# Add rocm repository
RUN set -xe && \
    apt-get update && apt-get install -y --allow-unauthenticated apt-utils wget gnupg2 curl

RUN wget https://repo.radeon.com/amdgpu-install/7.1.1/ubuntu/noble/amdgpu-install_7.1.1.70101-1_all.deb && \
    apt install ./amdgpu-install_7.1.1.70101-1_all.deb -y && \
    apt update && \
    apt install python3-setuptools python3-wheel -y && \
    apt install rocm-dev -y

## Sccache binary built from source for ROCm, only install if CK_SCCACHE is defined
ARG SCCACHE_REPO_URL=http://compute-artifactory.amd.com/artifactory/rocm-generic-experimental/rocm-sccache
ENV SCCACHE_INSTALL_LOCATION=/usr/local/.cargo/bin
ENV PATH=$PATH:${SCCACHE_INSTALL_LOCATION}
ENV CK_SCCACHE=$CK_SCCACHE
RUN if [ "$CK_SCCACHE" != "" ]; then \
        mkdir -p ${SCCACHE_INSTALL_LOCATION} && \
        curl ${SCCACHE_REPO_URL}/portable/0.2.16/sccache-0.2.16-alpha.1-rocm --output ${SCCACHE_INSTALL_LOCATION}/sccache && \
        chmod +x ${SCCACHE_INSTALL_LOCATION}/sccache; \
    fi

# Install dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated \
    build-essential \
    cmake \
    git \
    hip-rocclr \
    iputils-ping \
    jq \
    libelf-dev \
    libnuma-dev \
    libpthread-stubs0-dev \
    mpich \
    net-tools \
    pkg-config \
    python3-full \
    redis \
    rocm-llvm-dev \
    sshpass \
    stunnel \
    software-properties-common \
    vim \
    nano \
    zlib1g-dev \
    zip \
    libzstd-dev \
    openssh-server \
    clang-format-18 \
    kmod && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf amdgpu-install* && \
#Install latest ccache
    git clone https://github.com/ccache/ccache.git && \
    cd ccache && mkdir build && cd build && cmake .. && make install && \
#Install ninja build tracing tools
    cd / && \
    wget -qO /usr/local/bin/ninja.gz https://github.com/ninja-build/ninja/releases/latest/download/ninja-linux.zip && \
    gunzip /usr/local/bin/ninja.gz && \
    chmod a+x /usr/local/bin/ninja && \
#Install ClangBuildAnalyzer
    git clone https://github.com/aras-p/ClangBuildAnalyzer.git && \
    cd ClangBuildAnalyzer/ && \
    make -f projects/make/Makefile && \
    cd / && \
#Install latest cppcheck
    git clone https://github.com/danmar/cppcheck.git && \
    cd cppcheck && mkdir build && cd build && cmake .. && cmake --build . && \
    cd / && \
# Install an init system
    wget https://github.com/Yelp/dumb-init/releases/download/v1.2.0/dumb-init_1.2.0_amd64.deb && \
    dpkg -i dumb-init_*.deb && rm dumb-init_*.deb && \
# Install packages for processing the performance results
    pip3 install --break-system-packages --upgrade pytest pymysql pandas==2.2.3 sqlalchemy==2.0.3 setuptools-rust setuptools sshtunnel==0.4.0 && \
# Add render group
    groupadd -f render && \
# Install the new rocm-cmake version
    git clone -b master https://github.com/ROCm/rocm-cmake.git  && \
    cd rocm-cmake && mkdir build && cd build && \
    cmake  .. && cmake --build . && cmake --build . --target install

WORKDIR /
# Add alternative compilers, if necessary
ENV compiler_version=$compiler_version
ENV compiler_commit=$compiler_commit
RUN sh -c "echo compiler version = '$compiler_version'" && \
    sh -c "echo compiler commit = '$compiler_commit'"

RUN if ( [ "$compiler_version" = "amd-staging" ] || [ "$compiler_version" = "amd-mainline" ] ) && [ "$compiler_commit" = "" ]; then \
        git clone -b "$compiler_version" https://github.com/ROCm/llvm-project.git && \
        cd llvm-project && mkdir build && cd build && \
        cmake -DCMAKE_INSTALL_PREFIX=/opt/rocm/llvm -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=1 -DLLVM_TARGETS_TO_BUILD="AMDGPU;X86" -DLLVM_ENABLE_PROJECTS="clang;lld" -DLLVM_ENABLE_RUNTIMES="compiler-rt" ../llvm && \
        make -j 8 ; \
    else echo "using the release compiler"; \
    fi

RUN if ( [ "$compiler_version" = "amd-staging" ] || [ "$compiler_version" = "amd-mainline" ] ) && [ "$compiler_commit" != "" ]; then \
        git clone -b "$compiler_version" https://github.com/ROCm/llvm-project.git && \
        cd llvm-project && git checkout "$compiler_commit" && echo "checking out commit $compiler_commit" && mkdir build && cd build && \
        cmake -DCMAKE_INSTALL_PREFIX=/opt/rocm/llvm -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=1 -DLLVM_TARGETS_TO_BUILD="AMDGPU;X86" -DLLVM_ENABLE_PROJECTS="clang;lld" -DLLVM_ENABLE_RUNTIMES="compiler-rt" ../llvm && \
        make -j 8 ; \
    else echo "using the release compiler"; \
    fi
