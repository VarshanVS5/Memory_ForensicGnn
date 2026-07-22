"""
GNN architectures for graph-level fileless-malware classification.

Two options provided:
  - SAGEClassifier: GraphSAGE convs -> good default, handles inductive
    settings well (new artifact-node patterns at test time).
  - GATClassifier: Graph Attention convs -> slightly heavier, but attention
    weights are directly interpretable ("which artifact type drove this
    prediction"), which is valuable for a forensics report / your mentor demo.

Both do graph classification: node-level message passing, then
global mean+max pooling -> MLP head -> binary logit.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATv2Conv, global_mean_pool, global_max_pool


class SAGEClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, num_layers=3, dropout=0.3, num_classes=2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.dropout = dropout
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        pooled = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        return self.head(pooled)


class GATClassifier(nn.Module):
    """Attention-based variant. attn_weights exposed for interpretability."""

    def __init__(self, in_dim, hidden_dim=64, heads=4, num_layers=2, dropout=0.3, num_classes=2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATv2Conv(in_dim, hidden_dim, heads=heads, dropout=dropout))
        for _ in range(num_layers - 1):
            self.convs.append(GATv2Conv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout))
        self.dropout = dropout
        pooled_dim = hidden_dim * heads * 2
        self.head = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x, edge_index, batch, return_attention=False):
        attn_all = []
        for conv in self.convs:
            if return_attention:
                x, (ei, alpha) = conv(x, edge_index, return_attention_weights=True)
                attn_all.append((ei, alpha))
            else:
                x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        pooled = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        out = self.head(pooled)
        if return_attention:
            return out, attn_all
        return out


def build_model(name, in_dim, **kwargs):
    if name == "sage":
        return SAGEClassifier(in_dim, **kwargs)
    elif name == "gat":
        return GATClassifier(in_dim, **kwargs)
    raise ValueError(f"Unknown model: {name}")
