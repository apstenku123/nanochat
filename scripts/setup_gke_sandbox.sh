#!/bin/bash
# Setup GKE sandbox for GSPO training
#
# Prerequisites:
#   - gcloud CLI installed and authenticated
#   - kubectl configured
#   - Docker installed
#
# Usage:
#   ./scripts/setup_gke_sandbox.sh [--create-cluster] [--deploy-only]

set -euo pipefail

# Configuration
PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-}"
CLUSTER_NAME="${GKE_CLUSTER:-gspo-cluster}"
ZONE="${GKE_ZONE:-us-central1-a}"
NAMESPACE="gspo-sandbox"
IMAGE_NAME="gspo-sandbox"
IMAGE_TAG="${IMAGE_TAG:-latest}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    if [[ -z "$PROJECT_ID" ]]; then
        log_error "GOOGLE_CLOUD_PROJECT not set"
        echo "  Set it with: export GOOGLE_CLOUD_PROJECT=your-project-id"
        exit 1
    fi

    if ! command -v gcloud &> /dev/null; then
        log_error "gcloud CLI not found. Install from: https://cloud.google.com/sdk/docs/install"
        exit 1
    fi

    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl not found. Install with: gcloud components install kubectl"
        exit 1
    fi

    if ! command -v docker &> /dev/null; then
        log_error "docker not found. Install Docker first."
        exit 1
    fi

    log_info "Prerequisites OK"
}

# Create GKE cluster
create_cluster() {
    log_info "Creating GKE cluster: $CLUSTER_NAME"

    # Check if cluster exists
    if gcloud container clusters describe "$CLUSTER_NAME" --zone="$ZONE" --project="$PROJECT_ID" &> /dev/null; then
        log_warn "Cluster $CLUSTER_NAME already exists"
        return 0
    fi

    # Create cluster with Spot VMs for cost savings
    gcloud container clusters create "$CLUSTER_NAME" \
        --zone="$ZONE" \
        --project="$PROJECT_ID" \
        --machine-type="e2-standard-2" \
        --num-nodes=1 \
        --enable-autoscaling \
        --min-nodes=0 \
        --max-nodes=10 \
        --spot \
        --enable-network-policy \
        --workload-pool="${PROJECT_ID}.svc.id.goog" \
        --release-channel=regular \
        --no-enable-basic-auth \
        --metadata=disable-legacy-endpoints=true \
        --enable-shielded-nodes

    log_info "Cluster created successfully"
}

# Get cluster credentials
get_credentials() {
    log_info "Getting cluster credentials..."

    gcloud container clusters get-credentials "$CLUSTER_NAME" \
        --zone="$ZONE" \
        --project="$PROJECT_ID"

    log_info "kubectl configured for cluster: $CLUSTER_NAME"
}

# Build and push Docker image
build_and_push_image() {
    log_info "Building Docker image..."

    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
    DOCKER_DIR="$PROJECT_ROOT/docker/gspo-sandbox"

    if [[ ! -f "$DOCKER_DIR/Dockerfile" ]]; then
        log_error "Dockerfile not found at $DOCKER_DIR/Dockerfile"
        exit 1
    fi

    # Build image
    docker build -t "$IMAGE_NAME:$IMAGE_TAG" "$DOCKER_DIR"

    # Tag for GCR
    GCR_IMAGE="gcr.io/$PROJECT_ID/$IMAGE_NAME:$IMAGE_TAG"
    docker tag "$IMAGE_NAME:$IMAGE_TAG" "$GCR_IMAGE"

    # Push to GCR
    log_info "Pushing image to GCR..."
    gcloud auth configure-docker --quiet
    docker push "$GCR_IMAGE"

    log_info "Image pushed: $GCR_IMAGE"
}

# Deploy Kubernetes resources
deploy_k8s_resources() {
    log_info "Deploying Kubernetes resources..."

    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
    K8S_DIR="$PROJECT_ROOT/k8s/gspo-sandbox"

    # Create namespace
    kubectl apply -f "$K8S_DIR/namespace.yaml"

    # Create service account
    kubectl apply -f "$K8S_DIR/service-account.yaml"

    # Apply network policy
    kubectl apply -f "$K8S_DIR/network-policy.yaml"

    # Update job template with correct image
    log_info "Updating job template..."
    sed "s|gcr.io/PROJECT_ID/gspo-sandbox:latest|gcr.io/$PROJECT_ID/$IMAGE_NAME:$IMAGE_TAG|g" \
        "$K8S_DIR/job.yaml" | kubectl apply -f -

    log_info "Kubernetes resources deployed"
}

# Test the sandbox
test_sandbox() {
    log_info "Testing sandbox..."

    # Create a test ConfigMap
    kubectl create configmap gspo-code-test \
        --namespace="$NAMESPACE" \
        --from-literal='code.cpp=
#include <iostream>
int main() {
    std::cout << "Hello from GSPO sandbox!" << std::endl;
    return 0;
}
' --dry-run=client -o yaml | kubectl apply -f -

    # Create a test job
    cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: gspo-test-job
  namespace: $NAMESPACE
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: 60
  template:
    spec:
      serviceAccountName: gspo-sandbox-runner
      restartPolicy: Never
      securityContext:
        runAsUser: 1000
        runAsGroup: 1000
      containers:
        - name: sandbox
          image: gcr.io/$PROJECT_ID/$IMAGE_NAME:$IMAGE_TAG
          resources:
            limits:
              memory: "256Mi"
              cpu: "500m"
          env:
            - name: CODE_CONTENT
              valueFrom:
                configMapKeyRef:
                  name: gspo-code-test
                  key: code.cpp
EOF

    # Wait for job to complete
    log_info "Waiting for test job..."
    kubectl wait --for=condition=complete job/gspo-test-job \
        --namespace="$NAMESPACE" \
        --timeout=60s || true

    # Get logs
    log_info "Test job logs:"
    kubectl logs -l job-name=gspo-test-job --namespace="$NAMESPACE" || true

    # Cleanup
    kubectl delete job gspo-test-job --namespace="$NAMESPACE" || true
    kubectl delete configmap gspo-code-test --namespace="$NAMESPACE" || true

    log_info "Test completed"
}

# Show status
show_status() {
    log_info "Sandbox status:"
    echo ""

    echo "Namespace:"
    kubectl get namespace "$NAMESPACE" 2>/dev/null || echo "  Not found"
    echo ""

    echo "Service Account:"
    kubectl get serviceaccount -n "$NAMESPACE" 2>/dev/null || echo "  Not found"
    echo ""

    echo "Network Policies:"
    kubectl get networkpolicy -n "$NAMESPACE" 2>/dev/null || echo "  Not found"
    echo ""

    echo "Running Jobs:"
    kubectl get jobs -n "$NAMESPACE" 2>/dev/null || echo "  None"
    echo ""

    echo "Pods:"
    kubectl get pods -n "$NAMESPACE" 2>/dev/null || echo "  None"
}

# Main
main() {
    local create_cluster=false
    local deploy_only=false
    local test_only=false
    local status_only=false

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --create-cluster)
                create_cluster=true
                shift
                ;;
            --deploy-only)
                deploy_only=true
                shift
                ;;
            --test)
                test_only=true
                shift
                ;;
            --status)
                status_only=true
                shift
                ;;
            --help|-h)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --create-cluster  Create a new GKE cluster"
                echo "  --deploy-only     Only deploy K8s resources (skip image build)"
                echo "  --test            Run a test job"
                echo "  --status          Show sandbox status"
                echo "  --help            Show this help"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done

    check_prerequisites

    if $status_only; then
        get_credentials
        show_status
        exit 0
    fi

    if $test_only; then
        get_credentials
        test_sandbox
        exit 0
    fi

    if $create_cluster; then
        create_cluster
    fi

    get_credentials

    if ! $deploy_only; then
        build_and_push_image
    fi

    deploy_k8s_resources

    log_info "Setup complete!"
    echo ""
    echo "Next steps:"
    echo "  1. Run a test: $0 --test"
    echo "  2. Check status: $0 --status"
    echo "  3. Use GKEExecutor in Python:"
    echo "     from nanochat.gke_executor import GKEExecutor"
    echo "     executor = GKEExecutor()"
    echo "     result = await executor.execute(code='...')"
}

main "$@"
