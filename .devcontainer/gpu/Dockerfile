FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=1000
ARG TVM_VERSION=v0.19.0
ARG TVM_HOME=/opt/tvm
ARG TVM_BUILD=/opt/tvm/build

ENV DEBIAN_FRONTEND=noninteractive \
    TVM_VERSION=${TVM_VERSION} \
    TVM_HOME=${TVM_HOME} \
    TVM_BUILD=${TVM_BUILD} \
    TVM_USE_CUDA=ON \
    TVM_USE_VULKAN=ON \
    LLVM_CONFIG=/usr/bin/llvm-config-14

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    sudo \
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    libopenblas-dev \
    libblas-dev \
    liblapack-dev \
    libz-dev \
    llvm-14 \
    llvm-14-dev \
    clang-14 \
    lld-14 \
    gcc-aarch64-linux-gnu \
    g++-aarch64-linux-gnu \
    gcc-arm-linux-gnueabihf \
    g++-arm-linux-gnueabihf \
    emscripten \
    wabt \
    binaryen \
    libvulkan-dev \
    vulkan-tools \
    glslang-tools \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd -s /bin/bash --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} \
    && echo "${USERNAME} ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME} \
    && ln -sf /usr/bin/python3 /usr/local/bin/python \
    && mkdir -p "${TVM_HOME}" "${TVM_BUILD}" \
    && chown -R "${USERNAME}:${USERNAME}" "${TVM_HOME}"

USER ${USERNAME}
WORKDIR /workspaces
