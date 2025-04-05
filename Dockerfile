FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH="/root/miniconda3/bin:${PATH}"

# Install essential packages
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    ca-certificates \
    build-essential \
    pkg-config \
    libglfw3-dev \
    libgl1-mesa-dev \
    libglib2.0-0 \
    zip \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && chmod +x /tmp/miniconda.sh \
    && /tmp/miniconda.sh -b \
    && rm /tmp/miniconda.sh

# Create a basic Python environment instead of using the complex environment.yaml
RUN conda create -n ldm python=3.8.5 pytorch=1.11.0 torchvision=0.12.0 cudatoolkit=11.3 -c pytorch -y

# Make RUN commands use the conda environment
SHELL ["conda", "run", "-n", "ldm", "/bin/bash", "-c"]

# Set working directory
WORKDIR /app

# Copy repository files
COPY . /app/

# Install additional requirements through pip
RUN pip install -r requirements.txt

# Install taming-transformers and CLIP
RUN git clone https://github.com/CompVis/taming-transformers.git && \
    pip install -e ./taming-transformers
    
RUN git clone https://github.com/openai/CLIP.git && \
    pip install -e ./CLIP

# Install the stable-diffusion package itself
RUN pip install -e .

# Create directories for model weights and outputs
RUN mkdir -p /app/models/ldm/stable-diffusion-v1 /app/outputs

# Make model directory mount point
VOLUME /app/models/ldm/stable-diffusion-v1

# Expose port for Gradio web UI
EXPOSE 7860

# Set entrypoint
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "ldm", "python", "app.py"]