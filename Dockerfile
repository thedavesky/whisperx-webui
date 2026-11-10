# 1. Use official NVIDIA CUDA 12.8 image
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

# 2. Install system dependencies + Build Tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev git ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 3. Set working directory
WORKDIR /app

# 4. Setup virtual environment
RUN python3 -m venv /opt/venv

# 5. Set python/pip in PATH
ENV PATH="/opt/venv/bin:$PATH"

# 6. Copy the repository files
COPY . /app

# 7. Upgrade pip, setuptools, and wheel FIRST so git repos compile correctly
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# 8. Install dependencies
RUN python -m pip install --no-cache-dir -r requirements.txt

# 9. Set the default startup command
ENTRYPOINT ["/bin/bash", "-c", "python app.py ${COMMANDLINE_ARGS}"]
