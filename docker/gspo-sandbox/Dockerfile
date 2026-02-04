# GSPO C++ Code Execution Sandbox
# Secure, isolated environment for compiling and running untrusted C++ code
#
# Build: docker build -t gspo-sandbox:latest .
# Run:   docker run --rm --network=none --memory=512m --cpus=1 \
#          -v /path/to/code.cpp:/sandbox/code.cpp:ro \
#          gspo-sandbox:latest

FROM ubuntu:24.04

LABEL maintainer="nanochat"
LABEL description="Secure C++ code execution sandbox for GSPO training"
LABEL version="1.0"

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install C++ toolchains and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    # C++ compilers
    g++-14 \
    gcc-14 \
    clang-18 \
    clang++-18 \
    # Build tools
    cmake \
    ninja-build \
    make \
    autoconf \
    automake \
    libtool \
    pkg-config \
    # Testing frameworks
    libgtest-dev \
    catch2 \
    # Common libraries
    libboost-all-dev \
    libfmt-dev \
    nlohmann-json3-dev \
    librange-v3-dev \
    # Utilities
    jq \
    coreutils \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set up alternatives for default compilers
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-14 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-14 100 \
    && update-alternatives --install /usr/bin/clang clang /usr/bin/clang-18 100 \
    && update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-18 100

# Create non-root user for running code
RUN groupadd --gid 1000 sandbox \
    && useradd --uid 1000 --gid sandbox --shell /bin/bash --create-home sandbox

# Create sandbox directory with proper permissions
RUN mkdir -p /sandbox \
    && chown sandbox:sandbox /sandbox \
    && chmod 755 /sandbox

# Copy entrypoint script
COPY --chmod=755 entrypoint.sh /usr/local/bin/entrypoint.sh

# Set resource limits via ulimit (additional limits via docker run flags)
# These are soft limits; hard limits should be set via cgroups/docker
RUN echo "sandbox soft nproc 64" >> /etc/security/limits.conf \
    && echo "sandbox hard nproc 128" >> /etc/security/limits.conf \
    && echo "sandbox soft nofile 256" >> /etc/security/limits.conf \
    && echo "sandbox hard nofile 512" >> /etc/security/limits.conf \
    && echo "sandbox soft fsize 52428800" >> /etc/security/limits.conf \
    && echo "sandbox hard fsize 104857600" >> /etc/security/limits.conf

# Switch to non-root user
USER sandbox
WORKDIR /sandbox

# Default environment variables
ENV COMPILER="g++"
ENV CPP_STANDARD="20"
ENV TIMEOUT_SECONDS="10"
ENV OPTIMIZATION_LEVEL="-O2"
ENV EXTRA_FLAGS=""

# Health check (container is ready when compilers are available)
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD g++ --version && clang++ --version

# Entry point
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
