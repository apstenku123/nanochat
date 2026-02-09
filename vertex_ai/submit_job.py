#!/usr/bin/env python3
"""
Vertex AI Custom Job Submission Script for TPU Training

This script provides a Python SDK interface for submitting nanochat
training jobs to Vertex AI with TPU accelerators.

Usage:
    # TPU v5e with 4 chips
    python submit_job.py --tpu-type=v5e --topology=2x2

    # TPU v6e with 8 chips
    python submit_job.py --tpu-type=v6e --topology=2x4

    # Custom configuration
    python submit_job.py --tpu-type=v5e --topology=2x2 --model-depth=30 --batch-size=64
"""

import argparse
import os
from datetime import datetime

from google.cloud import aiplatform
from google.cloud.aiplatform import gapic as aip_gapic


# Project configuration
PROJECT_ID = "alpine-aspect-459819-m4"
REGION = "us-central1"
GCS_BUCKET = "gs://nanochat-training-data-2026"
CONTAINER_IMAGE = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/nanochat/tpu-trainer:latest"


def get_job_name():
    """Generate a unique job name with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"nanochat-tpu-{timestamp}"


def get_machine_spec(tpu_type: str, topology: str) -> dict:
    """Get machine specification for the given TPU type and topology."""
    # TPU v5e machine types
    v5e_machines = {
        "1x1": "ct5lp-hightpu-1t",
        "2x2": "ct5lp-hightpu-4t",
        "2x4": "ct5lp-hightpu-8t",
    }

    # TPU v6e machine types
    v6e_machines = {
        "1x1": "ct6e-standard-1t",
        "2x2": "ct6e-standard-4t",
        "2x4": "ct6e-standard-8t",
    }

    if tpu_type == "v5e":
        machine_type = v5e_machines.get(topology)
    elif tpu_type == "v6e":
        machine_type = v6e_machines.get(topology)
    else:
        raise ValueError(f"Unknown TPU type: {tpu_type}. Use 'v5e' or 'v6e'.")

    if machine_type is None:
        available = list(v5e_machines.keys() if tpu_type == "v5e" else v6e_machines.keys())
        raise ValueError(f"Unknown topology: {topology}. Available: {available}")

    # Calculate number of chips from topology
    dims = topology.split("x")
    num_chips = int(dims[0]) * int(dims[1])

    return {
        "machine_type": machine_type,
        "tpu_topology": topology,
        "num_chips": num_chips,
    }


def create_custom_job(
    tpu_type: str,
    topology: str,
    model_depth: int = 20,
    batch_size: int = 32,
    max_seq_len: int = 2048,
    num_iterations: int = 10000,
    eval_every: int = 500,
    save_every: int = 1000,
    use_spot: bool = False,
) -> dict:
    """Create a Vertex AI CustomJob specification."""

    machine_spec = get_machine_spec(tpu_type, topology)
    job_name = get_job_name()
    checkpoint_dir = f"{GCS_BUCKET}/checkpoints/{job_name}"

    # Build training arguments
    training_args = [
        f"--gcs_bucket={GCS_BUCKET}",
        f"--model_depth={model_depth}",
        f"--batch_size={batch_size}",
        f"--max_seq_len={max_seq_len}",
        f"--num_iterations={num_iterations}",
        f"--checkpoint_dir={checkpoint_dir}",
        f"--eval_every={eval_every}",
        f"--save_every={save_every}",
    ]

    # Environment variables
    env_vars = [
        {"name": "PJRT_DEVICE", "value": "TPU"},
        {"name": "XLA_USE_SPMD", "value": "1"},
        {"name": "TPU_CHIPS", "value": str(machine_spec["num_chips"])},
        {"name": "TPU_TOPOLOGY", "value": topology},
        {"name": "GOOGLE_CLOUD_PROJECT", "value": PROJECT_ID},
    ]

    # XLA optimization flags
    xla_flags = [
        "--xla_tpu_enable_data_parallel_all_reduce_opt=true",
        "--xla_tpu_data_parallel_opt_different_sized_ops=true",
    ]
    if tpu_type == "v6e":
        xla_flags.append("--xla_tpu_enable_async_collective_fusion=true")

    env_vars.append({"name": "XLA_FLAGS", "value": " ".join(xla_flags)})

    # Build job specification
    custom_job = {
        "display_name": job_name,
        "job_spec": {
            "worker_pool_specs": [
                {
                    "machine_spec": {
                        "machine_type": machine_spec["machine_type"],
                        "tpu_topology": machine_spec["tpu_topology"],
                    },
                    "replica_count": 1,
                    "container_spec": {
                        "image_uri": CONTAINER_IMAGE,
                        "args": training_args,
                        "env": env_vars,
                    },
                }
            ],
        },
    }

    # Add spot/preemptible scheduling if requested
    if use_spot:
        custom_job["job_spec"]["scheduling"] = {"strategy": "SPOT"}

    return custom_job


def submit_job(custom_job: dict, dry_run: bool = False) -> str:
    """Submit a custom job to Vertex AI."""

    print("\n" + "=" * 60)
    print("Vertex AI Custom Job Submission")
    print("=" * 60)
    print(f"Project: {PROJECT_ID}")
    print(f"Region: {REGION}")
    print(f"Job Name: {custom_job['display_name']}")
    print(f"Container: {CONTAINER_IMAGE}")
    print("=" * 60)

    # Print job configuration
    worker_spec = custom_job["job_spec"]["worker_pool_specs"][0]
    print(f"\nMachine Type: {worker_spec['machine_spec']['machine_type']}")
    print(f"TPU Topology: {worker_spec['machine_spec']['tpu_topology']}")
    print(f"\nTraining Arguments:")
    for arg in worker_spec["container_spec"]["args"]:
        print(f"  {arg}")

    if dry_run:
        print("\n[DRY RUN] Job not submitted.")
        return None

    # Initialize Vertex AI
    aiplatform.init(project=PROJECT_ID, location=REGION)

    # Create and submit job using gapic client
    client_options = {"api_endpoint": f"{REGION}-aiplatform.googleapis.com"}
    job_client = aip_gapic.JobServiceClient(client_options=client_options)

    parent = f"projects/{PROJECT_ID}/locations/{REGION}"
    response = job_client.create_custom_job(parent=parent, custom_job=custom_job)

    job_name = response.name
    print(f"\n[SUCCESS] Job submitted!")
    print(f"Job Name: {job_name}")
    print(f"\nMonitor with:")
    print(f"  gcloud ai custom-jobs describe {job_name.split('/')[-1]} --region={REGION}")
    print(f"\nStream logs with:")
    print(f"  gcloud ai custom-jobs stream-logs {job_name.split('/')[-1]} --region={REGION}")

    return job_name


def main():
    parser = argparse.ArgumentParser(
        description="Submit nanochat TPU training job to Vertex AI"
    )

    # TPU configuration
    parser.add_argument(
        "--tpu-type",
        type=str,
        choices=["v5e", "v6e"],
        default="v5e",
        help="TPU generation (v5e or v6e)",
    )
    parser.add_argument(
        "--topology",
        type=str,
        choices=["1x1", "2x2", "2x4"],
        default="2x2",
        help="TPU topology (chip arrangement)",
    )

    # Model configuration
    parser.add_argument(
        "--model-depth", type=int, default=20, help="Transformer model depth"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Per-device batch size"
    )
    parser.add_argument(
        "--max-seq-len", type=int, default=2048, help="Maximum sequence length"
    )

    # Training configuration
    parser.add_argument(
        "--num-iterations", type=int, default=10000, help="Number of training iterations"
    )
    parser.add_argument(
        "--eval-every", type=int, default=500, help="Evaluate every N steps"
    )
    parser.add_argument(
        "--save-every", type=int, default=1000, help="Save checkpoint every N steps"
    )

    # Job options
    parser.add_argument(
        "--use-spot",
        action="store_true",
        help="Use spot/preemptible instances for cost savings",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print job configuration without submitting",
    )

    args = parser.parse_args()

    # Create and submit job
    custom_job = create_custom_job(
        tpu_type=args.tpu_type,
        topology=args.topology,
        model_depth=args.model_depth,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        num_iterations=args.num_iterations,
        eval_every=args.eval_every,
        save_every=args.save_every,
        use_spot=args.use_spot,
    )

    submit_job(custom_job, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
