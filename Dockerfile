# 1. Use a build argument to define the base image.
# This allows us to pass a different image for each platform (CPU vs. GPU) during the build command.
ARG BASE_IMAGE=python:3.10-slim

FROM ${BASE_IMAGE}

# 2. Capture the target platform, which is automatically passed by `docker buildx`.
# This will be "linux/arm64" on your Mac and "linux/amd64" for the remote server.
ARG TARGETPLATFORM

# 3. Set up the working directory and install common dependencies.
WORKDIR /workspace
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy and install Python package requirements.
# This file now contains dependencies *except* for PyTorch.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Conditionally install PyTorch.
# If building for arm64, we install the CPU version of PyTorch.
# If building for amd64, we do nothing here because the pytorch/pytorch base image already includes it.
RUN if [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
        echo "Building for arm64. Installing CPU version of PyTorch." && \
        pip install --no-cache-dir torch torchvision; \
    else \
        echo "Building for amd64. PyTorch is included in the base image."; \
    fi

# 6. Set a default command to start a shell.
# This keeps the container running so you can attach to it.
CMD ["bash"]