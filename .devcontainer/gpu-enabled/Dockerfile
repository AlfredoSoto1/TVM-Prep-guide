# Match CUDA to your driver/runtime. 12.2 is a good default today.
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

# Create a VS Code-compatible non-root user
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=1000

# Base deps (build tools + Python). Keep lean but ready for source builds too.
RUN apt-get update && \
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  ca-certificates curl git \
  build-essential cmake ninja-build \
  python3 python3-pip python3-venv python3-dev \
  llvm-14 clang-14 \
  && rm -rf /var/lib/apt/lists/*

# Make "python" available at the expected path
RUN ln -s /usr/bin/python3 /usr/local/bin/python

# Create the user
RUN groupadd --gid ${USER_GID} ${USERNAME} \
  && useradd -s /bin/bash --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME}

USER ${USERNAME}
WORKDIR /workspaces
