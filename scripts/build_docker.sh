#!/bin/bash

# Build script for Docker images

set -e

echo "🐳 Building Agentic Segmentation Pipeline Docker Image..."

# Build the Docker image
docker build -t agentic-segmentation:latest .

echo "✅ Docker image built successfully!"

# Optional: Run a quick test
echo "🧪 Running quick test..."
docker run --rm agentic-segmentation:latest python -c "
import sys
sys.path.append('/app/src')
from src.pipeline import AgenticSegmentationPipeline
print('✅ Pipeline import successful!')
"

echo "🎉 Build completed successfully!"
echo ""
echo "To run the container:"
echo "  docker run -p 8000:8000 agentic-segmentation:latest"
echo ""
echo "To run with docker-compose:"
echo "  docker-compose up"
