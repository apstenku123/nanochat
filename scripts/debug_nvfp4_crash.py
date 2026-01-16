"""
Debug NVFP4 crash with minimal repro.
"""
import os
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

import torch
from nanochat.gpt import GPT, GPTConfig, select_precision, make_autocast_ctx

device = torch.device('cuda')

depth = 20
model_dim = 1280
num_heads = 10
vocab_size = 65536
max_seq_len = 2048
B = 16  # B=32 OOMs during backward
T = max_seq_len

print(f'NVFP4 crash debug: B={B}, T={T}')

precision_plan = select_precision(target='nvfp4')
autocast_ctx = make_autocast_ctx(precision_plan, 'cuda')

print(f'Precision: {precision_plan.name}')
print(f'Recipe: {type(precision_plan.recipe).__name__ if precision_plan.recipe else None}')

with torch.device('meta'):
    config = GPTConfig(sequence_len=max_seq_len, vocab_size=vocab_size, n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim)
    model = GPT(config)
model.to_empty(device=device)
model.init_weights()
model.to(dtype=torch.bfloat16)

print(f'Model created, params: {sum(p.numel() for p in model.parameters()):,}')
print(f'GPU memory: {torch.cuda.memory_allocated()/1e9:.1f} GB')

# NO torch.compile - test raw
print('Testing forward pass...')
x = torch.randint(0, vocab_size, (B, T), device=device)
y = torch.randint(0, vocab_size, (B, T), device=device)

print(f'Input x: {x.shape}, {x.dtype}')

with autocast_ctx():
    print('Forward...')
    loss = model(x, y)
    print(f'Loss: {loss.item()}')

print('Backward...')
loss.backward()

print('Success!')
