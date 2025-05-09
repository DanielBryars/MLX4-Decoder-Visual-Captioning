import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import CLIPModel, CLIPTokenizer
from torchvision import transforms
from torch.utils.data import DataLoader
from FlickrDataset import FlickrDataset
from ProjectEmbeddingDimension import ProjectEmbeddingDimension


import matplotlib.pyplot as plt

def save_attention_mask(mask, filename="mask.png", title="Attention Mask"):
    import matplotlib.pyplot as plt

    if mask.dtype != torch.bool:
        raise ValueError("Expected boolean mask")

    data = mask.cpu().numpy().astype(int)  # True → 1, False → 0

    plt.figure(figsize=(6, 6))
    plt.imshow(data, cmap='gray', aspect='auto', interpolation='nearest')
    plt.title(title)
    plt.xlabel("Key Position")
    plt.ylabel("Query Position")
    plt.colorbar(label='Mask: 1 = Blocked, 0 = Allowed')
    plt.tight_layout()
    plt.savefig(filename)
    #plot.show()
    plt.close()
    

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

    def make_combined_mask(self, image_len, caption_len, device):
        """
        Returns a boolean mask of shape [total_len, total_len]
        True = mask (disallow attention), False = allow
        """
        total_len = image_len + caption_len

        # Start with all allowed
        mask = torch.zeros((total_len, total_len), dtype=torch.bool, device=device)

        # Causal mask for caption → caption (upper triangle, excluding diagonal)
        mask[image_len:, image_len:] = torch.triu(
            torch.ones((caption_len, caption_len), dtype=torch.bool, device=device),
            diagonal=1
        )

        # Make top right blocked
        #mask[:image_len, :] = 1
        mask[:image_len, :image_len] = 0  # allow image→image attention
        mask[:image_len, image_len:] = 1  # block image→caption
        # All other attention (image tokens attend to all; captions to image) is allowed (False)
        return mask


    def forward(self, image_embeds, caption_embeds, caption_attention_mask):
        


        #
        """
        image_embeds: [B, 50, 768]
        bos_embed: [B, 1, D]
        caption_embeds: [B, T, D] (already embedded tokens, excluding final token)
        Returns:
            logits: [B, T, vocab_size]
        """


        B, T = caption_attention_mask.shape
        device = caption_attention_mask.device
        padding_mask = torch.cat([
            torch.zeros((B, image_embeds.size(1)), dtype=torch.bool, device=device),  # no mask on image tokens
            ~caption_attention_mask.bool()  # invert to match key_padding_mask expectations
        ], dim=1)  # [B, S]


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

        save_attention_mask(tgt_mask, filename="tgt_mask.png", title="Combined Decoder Mask")

        default_causal = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).to(dtype=torch.bool)
        save_attention_mask(default_causal, filename="default_causal_mask.png", title="Default Decoder Mask")

        # Dummy memory (not used here — purely decoder-only)
        dummy_memory = torch.zeros(1, B, D, device=device)

        save_attention_mask(~caption_attention_mask.bool() , filename="caption_attention_mask_only.png", title="Padding Mask Text ONLY")

        save_attention_mask(padding_mask, filename="padding_mask.png", title="Padding Mask")

        # Decoder forward        
        output = self.transformer_decoder(
           decoder_input,
            dummy_memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=padding_mask
        )

        output = output.transpose(0, 1)  # [B, S, D]

        # Predict token logits
        logits = self.output_proj(output)  # [B, S, V]

        # Discard logits for image and BOS tokens; return only caption token logits
        #return logits[:, image_embeds.size(1) + 1:, :]  # [B, T, V]
        return logits[:, image_embeds.size(1):, :]
    #    