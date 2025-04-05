#!/bin/bash

# Build the Docker image
echo "Building Docker image..."
docker build -t stable-diffusion-test .

# Run the container with mock model
echo "Running container with mock model..."
docker run --rm -p 7860:7860 stable-diffusion-test bash -c "python download_weights.py --mock && python app.py"

# The container will stay running until you press Ctrl+C 