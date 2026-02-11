import torch
import torch.nn as nn
import torch.nn.functional as F


def _parse_ngram_orders(ngram_orders):
    if isinstance(ngram_orders, str):
        parts = [x.strip() for x in ngram_orders.split(",")]
        orders = [int(x) for x in parts if x]
    else:
        orders = [int(x) for x in ngram_orders]
    # Keep first occurrence order while removing duplicates.
    deduped = []
    seen = set()
    for order in orders:
        if order <= 0 or order in seen:
            continue
        deduped.append(order)
        seen.add(order)
    return tuple(deduped or [2, 3, 4])


class EngramBranch(nn.Module):
    """
    Lightweight causal n-gram branch.

    Input:  (B, T, C)
    Output: (B, T, C)
    """

    def __init__(self, n_embd, ngram_orders="2,3,4", bottleneck_dim=0, dropout=0.0):
        super().__init__()
        self.n_embd = n_embd
        self.ngram_orders = _parse_ngram_orders(ngram_orders)
        bottleneck_dim = int(bottleneck_dim) if bottleneck_dim else max(1, n_embd // 4)
        self.bottleneck_dim = bottleneck_dim

        self.in_proj = nn.Linear(n_embd, bottleneck_dim, bias=False)
        self.order_mix = nn.ModuleList([
            nn.Linear(bottleneck_dim, bottleneck_dim, bias=False)
            for _ in self.ngram_orders
        ])
        self.out_proj = nn.Linear(bottleneck_dim, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Conservative init: enabling Engram starts close to baseline.
        torch.nn.init.zeros_(self.out_proj.weight)

    def _causal_local_average(self, x, order):
        # x shape: (B, T, Cb) -> (B, T, Cb), only attends to current/past tokens
        x_t = x.transpose(1, 2)
        x_t = F.pad(x_t, (order - 1, 0))
        x_t = F.avg_pool1d(x_t, kernel_size=order, stride=1)
        return x_t.transpose(1, 2)

    def forward(self, x):
        z = self.in_proj(x)
        y = torch.zeros_like(z)
        for order, mix in zip(self.ngram_orders, self.order_mix):
            local = self._causal_local_average(z, order)
            y = y + mix(local)
        y = y / len(self.ngram_orders)
        y = self.out_proj(y)
        y = self.dropout(y)
        return y
