"""
Meta-device model initialization for memory-efficient model creation.

Uses PyTorch's meta device to avoid double memory allocation when creating
large models. Instead of allocating on CPU and then moving to GPU (2x memory),
this allocates tensor metadata only (no backing storage) and then materializes
directly on the target device.

Usage:
    from nanochat.meta_init import create_model_on_device
    from nanochat.gpt import GPT, GPTConfig

    config = GPTConfig(...)
    model = create_model_on_device(GPT, config, device)

This is the modern PyTorch idiom (torch.device("meta") + to_empty + init_weights)
and is especially useful for large models where double allocation would exceed
available memory.
"""

import torch


def create_model_on_device(model_cls, config, device):
    """Create a model using the meta device pattern to avoid double memory allocation.

    1. Instantiate on the meta device (no actual tensor storage allocated)
    2. Materialize empty tensors directly on the target device
    3. Initialize weights in-place

    Args:
        model_cls: The model class (e.g. GPT) - must have an init_weights() method.
        config: The model config to pass to the constructor.
        device: Target device (e.g. torch.device("cuda:0")).

    Returns:
        Initialized model on the target device.
    """
    with torch.device("meta"):
        model = model_cls(config)
    # Materialize on target device without intermediate CPU allocation
    model = model.to_empty(device=device)
    model.init_weights()
    return model
