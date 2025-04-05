# Stable Diffusion Deployment Guide

This guide provides instructions for deploying Stable Diffusion in different environments.

## Prerequisites

- CUDA-capable NVIDIA GPU with at least 10GB VRAM
- CUDA and cuDNN installed
- Python 3.8+
- Git

## Option 1: Local Deployment with Conda

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/stable-diffusion.git
   cd stable-diffusion
   ```

2. **Create and activate the Conda environment**:
   ```bash
   conda env create -f environment.yaml
   conda activate ldm
   ```

3. **Install additional requirements**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download model weights**:
   ```bash
   bash scripts/download_models.sh
   ```
   
   Alternatively, download the model weights manually from [Hugging Face](https://huggingface.co/CompVis/stable-diffusion) and place the `model.ckpt` file in `models/ldm/stable-diffusion-v1/`.

5. **Run the web UI**:
   ```bash
   python app.py
   ```

   The UI will be available at `http://localhost:7860`.

## Option 2: Docker Deployment

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/stable-diffusion.git
   cd stable-diffusion
   ```

2. **Download model weights**:
   ```bash
   bash scripts/download_models.sh
   ```
   
   Make sure the model weights are in `models/ldm/stable-diffusion-v1/model.ckpt`.

3. **Build and start the Docker container**:
   ```bash
   docker-compose up -d
   ```

   The UI will be available at `http://localhost:7860`.

## Option 3: Cloud Deployment

### Google Cloud Run

1. **Build the Docker image**:
   ```bash
   docker build -t gcr.io/[PROJECT_ID]/stable-diffusion .
   ```

2. **Push to Google Container Registry**:
   ```bash
   docker push gcr.io/[PROJECT_ID]/stable-diffusion
   ```

3. **Deploy to Cloud Run**:
   ```bash
   gcloud run deploy stable-diffusion --image gcr.io/[PROJECT_ID]/stable-diffusion --platform managed
   ```

### AWS Elastic Container Service (ECS)

1. **Create an Amazon ECR repository**:
   ```bash
   aws ecr create-repository --repository-name stable-diffusion
   ```

2. **Authenticate Docker to your ECR registry**:
   ```bash
   aws ecr get-login-password | docker login --username AWS --password-stdin [AWS_ACCOUNT_ID].dkr.ecr.[REGION].amazonaws.com
   ```

3. **Build and tag the Docker image**:
   ```bash
   docker build -t [AWS_ACCOUNT_ID].dkr.ecr.[REGION].amazonaws.com/stable-diffusion:latest .
   ```

4. **Push the image to ECR**:
   ```bash
   docker push [AWS_ACCOUNT_ID].dkr.ecr.[REGION].amazonaws.com/stable-diffusion:latest
   ```

5. **Create an ECS task definition and service** using the AWS Management Console or CLI.

## Troubleshooting

### Common Issues

1. **CUDA out of memory error**:
   - Reduce batch size or image dimensions
   - Try using a different sampler (PLMS instead of DDIM)
   - Close other GPU-intensive applications

2. **Model not found error**:
   - Ensure model weights are downloaded and placed in the correct directory
   - Check that the model path in `app.py` matches your file location

3. **Docker issues**:
   - Ensure NVIDIA Container Toolkit is installed for GPU support
   - Check that the volumes are mounted correctly

### Getting Help

If you encounter any issues not covered in this guide, please [open an issue](https://github.com/yourusername/stable-diffusion/issues) on GitHub.