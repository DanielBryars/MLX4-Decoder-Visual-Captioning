from DecoderBlock import DecoderBlock
from PositionalEncoding import PositionalEncoding, causal_mask
import torch
import torch.nn as nn

class DansDecoder(nn.Module):
    def __init__(self, d_model, n_heads, num_layers, vocab_size, max_len=128):
        super().__init__()
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, n_heads) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x_embeds):  # x_embeds: (B, T, D)
        x = self.pos_enc(x_embeds)
        B, T, _ = x.shape
        mask = causal_mask(T).to(x.device)  # shape (1, 1, T, T)

        for layer in self.layers:
            x = layer(x, attn_mask=mask)

        x = self.ln_f(x)
        return self.head(x)  # (B, T, vocab_size)