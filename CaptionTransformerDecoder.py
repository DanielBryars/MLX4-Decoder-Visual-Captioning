import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import CLIPModel, CLIPTokenizer
from torchvision import transforms
from torch.utils.data import DataLoader
from FlickrDataset import FlickrDataset
from ProjectEmbeddingDimension import ProjectEmbeddingDimension

class CaptionTransformerDecoder(nn.Module):
    def __init__(self, 
                 embed_dim,
                 vocab_size, 
                 num_layers=6, 
                 num_heads=8, 
                 dropout=0.1):
        super().__init__()

        self.image_proj = ProjectEmbeddingDimension(d_from = 768, d_to = embed_dim)

        self.embed_dim = embed_dim
        #self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, embed_dim))  # learnable positions

        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(embed_dim, vocab_size)

    def make_combined_mask(image_len, caption_len, device):
        total_len = image_len + caption_len
        mask = torch.full((total_len, total_len), float('-inf')).to(device)
        mask[:image_len, :] = 0  # image tokens attend to all
        mask[image_len:, :image_len] = 0  # caption tokens attend to image tokens
        mask[image_len:, image_len:] = torch.triu(torch.full((caption_len, caption_len), float('-inf')), 1)
        return mask

    def forward(self, image_embeds, caption_embeds):
        
        #
        """
        image_embeds: [B, 50, 768]
        bos_embed: [B, 1, D]
        caption_embeds: [B, T, D] (already embedded tokens, excluding final token)
        Returns:
            logits: [B, T, vocab_size]
        """
        #print(f"image_embeds.shape:{image_embeds.shape}")
        image_embeds = self.image_proj(image_embeds)
        #print(f"image_embeds.shape:{image_embeds.shape}")

        #print(f"bos_embed.shape:{bos_embed.shape}")
        #print(f"caption_embeds.shape:{caption_embeds.shape}")

        B, T, D = caption_embeds.shape
        device = caption_embeds.device

        # Construct full input sequence to decoder: [image | BOS | caption[:-1]]
        #decoder_input = torch.cat([image_embeds, bos_embed, caption_embeds], dim=1)  # [B, S, D]
        decoder_input = torch.cat([image_embeds, caption_embeds], dim=1)  # [B, S, D]
        
        seq_len = decoder_input.size(1)

        # Add positional encoding
        #decoder_input = decoder_input + self.pos_embed[:, :seq_len]

        # Transformer expects [seq_len, batch, dim]
        decoder_input = decoder_input.transpose(0, 1)

        # Create causal mask
        tgt_mask = self.make_combined_mask(image_len=image_embeds.size(1), caption_len=caption_embeds.size(1), device=device)

        # Dummy memory (not used here — purely decoder-only)
        dummy_memory = torch.zeros(1, B, D, device=device)

        # Decoder forward
        output = self.transformer_decoder(decoder_input, dummy_memory, tgt_mask=tgt_mask)  # [S, B, D]
        output = output.transpose(0, 1)  # [B, S, D]

        # Predict token logits
        logits = self.output_proj(output)  # [B, S, V]

        # Discard logits for image and BOS tokens; return only caption token logits
        #return logits[:, image_embeds.size(1) + 1:, :]  # [B, T, V]
        return logits[:, image_embeds.size(1):, :]
    #    