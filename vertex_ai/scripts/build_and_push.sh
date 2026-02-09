#!/bin/bash
# Build and push the TPU training container to Artifact Registry
#
# Prerequisites:
#   - gcloud CLI installed and authenticated
#   - Docker installed and running
#   - Artifact Registry repository created

set -e

PROJECT_ID="alpine-aspect-459819-m4"
REGION="us-central1"
REPO_NAME="nanochat"
IMAGE_NAME="tpu-trainer"
TAG="${1:-latest}"

FULL_IMAGE_NAME="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${TAG}"

echo "=============================================="
echo "Building TPU Training Container"
echo "=============================================="
echo "Project: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "Image: ${FULL_IMAGE_NAME}"
echo "=============================================="

# Navigate to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

echo ""
echo "Step 1: Configure Docker authentication..."
gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet

echo ""
echo "Step 2: Create Artifact Registry repository (if not exists)..."
gcloud artifacts repositories create ${REPO_NAME} \
    --repository-format=docker \
    --location=${REGION} \
    --description="nanochat training containers" \
    2>/dev/null || echo "Repository already exists"

echo ""
echo "Step 3: Building Docker image..."
docker build \
    -t ${FULL_IMAGE_NAME} \
    -f vertex_ai/docker/Dockerfile.tpu \
    .

echo ""
echo "Step 4: Pushing to Artifact Registry..."
docker push ${FULL_IMAGE_NAME}

echo ""
echo "=============================================="
echo "SUCCESS: Image pushed to ${FULL_IMAGE_NAME}"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Submit training job:"
echo "     python vertex_ai/submit_job.py --tpu-type=v5e --topology=2x2"
echo ""
echo "  2. Or use gcloud directly:"
echo "     gcloud ai custom-jobs create \\"
echo "       --region=${REGION} \\"
echo "       --display-name=nanochat-tpu-training \\"
echo "       --config=vertex_ai/config/tpu_v5e_config.yaml"
