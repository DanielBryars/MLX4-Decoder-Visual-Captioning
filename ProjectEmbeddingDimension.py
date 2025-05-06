import torch.nn as nn

class ProjectEmbeddingDimension(nn.Module):
    def __init__(self, d_from = 768, d_to = 512):
        super().__init__()
        self.proj = nn.Linear(d_from, d_to)

    def forward(self, x):
        return self.proj(x)