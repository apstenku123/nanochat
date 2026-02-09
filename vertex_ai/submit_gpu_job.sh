#!/bin/bash
# Submit the nanochat 400M training job to Vertex AI
# CRITICAL: num_iterations=50000 (FIFTY THOUSAND - NOT 5000!)

set -e

# Configuration
PROJECT_ID="alpine-aspect-459819-m4"
REGION="us-central1"
JOB_NAME="nanochat-d16-400M-vertex-50k-$(date +%Y%m%d-%H%M%S)"

echo "=============================================="
echo "Submitting nanochat 400M training job"
echo "=============================================="
echo "Job Name: $JOB_NAME"
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo ""
echo "CRITICAL PARAMETERS:"
echo "  depth=16"
echo "  num_iterations=50000 (FIFTY THOUSAND)"
echo "  fim_rate=0.4"
echo "  kernel=cce"
echo "  run=d16_400M_vertex_50k"
echo "=============================================="

# Submit the job
gcloud ai custom-jobs create \
    --region="$REGION" \
    --display-name="$JOB_NAME" \
    --config="vertex_ai/config/gpu_a100_400M_50k_prebuilt.yaml"

echo ""
echo "Job submitted! Monitor with:"
echo "  gcloud ai custom-jobs list --region=$REGION"
echo "  gcloud ai custom-jobs describe JOB_ID --region=$REGION"
