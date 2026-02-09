# Vertex AI TPU Training Pipeline for nanochat

This directory contains configuration and scripts for training nanochat models on Google Cloud Vertex AI with TPU accelerators.

## Overview

- **Project**: `alpine-aspect-459819-m4`
- **Region**: `us-central1` (TPU v5e/v6e availability)
- **GCS Bucket**: `gs://nanochat-training-data-2026/`
- **Supported TPUs**: TPU v5e, TPU v6e

## Directory Structure

```
vertex_ai/
├── README.md                    # This file
├── config/
│   ├── tpu_v5e_config.yaml     # TPU v5e job configuration
│   └── tpu_v6e_config.yaml     # TPU v6e job configuration
├── docker/
│   └── Dockerfile.tpu          # PyTorch XLA training container
├── trainer/
│   └── train_tpu.py            # TPU-adapted training script
└── submit_job.py               # Python SDK job submission script
```

## Quick Start

### 1. Build and Push Docker Container

```bash
# Build the training container
cd vertex_ai/docker
docker build -t us-central1-docker.pkg.dev/alpine-aspect-459819-m4/nanochat/tpu-trainer:latest -f Dockerfile.tpu .

# Push to Artifact Registry
gcloud auth configure-docker us-central1-docker.pkg.dev
docker push us-central1-docker.pkg.dev/alpine-aspect-459819-m4/nanochat/tpu-trainer:latest
```

### 2. Submit Training Job

**Using gcloud CLI:**

```bash
# TPU v5e (4 chips, 2x2 topology)
gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name=nanochat-tpu-v5e-training \
  --config=vertex_ai/config/tpu_v5e_config.yaml

# TPU v6e (4 chips, 2x2 topology)
gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name=nanochat-tpu-v6e-training \
  --config=vertex_ai/config/tpu_v6e_config.yaml
```

**Using Python SDK:**

```bash
python vertex_ai/submit_job.py --tpu-type=v5e --topology=2x2
```

### 3. Monitor Training

```bash
# List running jobs
gcloud ai custom-jobs list --region=us-central1

# Stream logs
gcloud ai custom-jobs stream-logs JOB_ID --region=us-central1

# Describe job status
gcloud ai custom-jobs describe JOB_ID --region=us-central1
```

## TPU Configuration Options

### TPU v5e (Recommended for Cost-Efficiency)

| Machine Type | TPU Chips | Topology | Use Case |
|--------------|-----------|----------|----------|
| `ct5lp-hightpu-1t` | 1 | 1x1 | Small experiments |
| `ct5lp-hightpu-4t` | 4 | 2x2 | Medium models (~100M params) |
| `ct5lp-hightpu-8t` | 8 | 2x4 | Large models (~400M+ params) |

### TPU v6e (Higher Performance)

| Machine Type | TPU Chips | Topology | Use Case |
|--------------|-----------|----------|----------|
| `ct6e-standard-1t` | 1 | 1x1 | Small experiments |
| `ct6e-standard-4t` | 4 | 2x2 | Medium models |
| `ct6e-standard-8t` | 8 | 2x4 | Large models |

## Training Data

Data is stored in GCS:

```
gs://nanochat-training-data-2026/
├── data/
│   ├── combined_sft.jsonl      # SFT training data
│   └── diff_sft.jsonl          # Diff-based SFT data
├── tokenizer/
│   ├── tokenizer.json          # HuggingFace tokenizer config
│   ├── tokenizer.pkl           # Pickled tokenizer
│   ├── token_bytes.pt          # Token bytes tensor
│   └── fixed_vocab.json        # Vocabulary
└── eval/                       # Evaluation datasets
```

## Environment Variables

The training container uses these environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `PJRT_DEVICE` | PyTorch XLA runtime device | `TPU` |
| `XLA_USE_SPMD` | Enable SPMD for distributed training | `1` |
| `GCS_BUCKET` | GCS bucket for data/checkpoints | `gs://nanochat-training-data-2026` |
| `MODEL_DEPTH` | Transformer depth | `20` |
| `BATCH_SIZE` | Per-device batch size | `32` |
| `MAX_SEQ_LEN` | Maximum sequence length | `2048` |

## Cost Estimation

| TPU Type | Chips | On-Demand $/hr | Spot $/hr | 24hr Cost |
|----------|-------|----------------|-----------|-----------|
| v5e-4 | 4 | ~$3.20 | ~$0.96 | $23-$77 |
| v5e-8 | 8 | ~$6.40 | ~$1.92 | $46-$154 |
| v6e-4 | 4 | ~$4.80 | ~$1.44 | $35-$115 |
| v6e-8 | 8 | ~$9.60 | ~$2.88 | $69-$230 |

*Prices are approximate and may vary by region.*

## Troubleshooting

### Common Issues

1. **"TPU quota exceeded"**: Request quota increase in Cloud Console
2. **"Container failed to start"**: Check container logs, ensure PyTorch XLA is installed correctly
3. **"OOM on TPU"**: Reduce batch size or model size
4. **"Slow data loading"**: Ensure data is in the same region as TPU

### Useful Commands

```bash
# Check TPU quota
gcloud compute regions describe us-central1 --format="yaml(quotas)" | grep -i tpu

# Cancel a running job
gcloud ai custom-jobs cancel JOB_ID --region=us-central1

# Delete old jobs
gcloud ai custom-jobs delete JOB_ID --region=us-central1
```

## References

- [Vertex AI TPU Training Documentation](https://cloud.google.com/vertex-ai/docs/training/training-with-tpu-vm)
- [PyTorch/XLA Documentation](https://docs.pytorch.org/xla/master/accelerators/tpu.html)
- [TPU v6e Training Guide](https://cloud.google.com/tpu/docs/v6e-training)
