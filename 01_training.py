import torch
import torch.nn as nn
from tqdm import tqdm
from DecoderBlock import DecoderBlock
#from PositionalEncoding import PositionalEncoding, causal_mask
from transformers import CLIPModel, CLIPTokenizer
from torchvision import transforms
from torch.utils.data import DataLoader
from FlickrDataset import FlickrDataset
import math


def causal_mask(seq_len):
    return torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)


class CaptionTransformerDecoder(nn.Module):
    def __init__(self, 
                 embed_dim,
                 vocab_size, 
                 num_layers=6, 
                 num_heads=8, 
                 dropout=0.1, 
                 max_seq_len=128):
        super().__init__()
        self.embed_dim = embed_dim
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, embed_dim))  # learnable positions

        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(embed_dim, vocab_size)

    def forward(self, image_embeds, bos_embed, caption_embeds):
        """
        image_embeds: [B, 50, D]
        bos_embed: [B, 1, D]
        caption_embeds: [B, T, D] (already embedded tokens, excluding final token)
        Returns:
            logits: [B, T, vocab_size]
        """
        B, T, D = caption_embeds.shape
        device = caption_embeds.device

        # Construct full input sequence to decoder: [image | BOS | caption[:-1]]
        decoder_input = torch.cat([image_embeds, bos_embed, caption_embeds], dim=1)  # [B, S, D]
        seq_len = decoder_input.size(1)

        # Add positional encoding
        decoder_input = decoder_input + self.pos_embed[:, :seq_len]

        # Transformer expects [seq_len, batch, dim]
        decoder_input = decoder_input.transpose(0, 1)

        # Create causal mask
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)

        # Dummy memory (not used here — purely decoder-only)
        dummy_memory = torch.zeros(1, B, D, device=device)

        # Decoder forward
        output = self.transformer_decoder(decoder_input, dummy_memory, tgt_mask=tgt_mask)  # [S, B, D]
        output = output.transpose(0, 1)  # [B, S, D]

        # Predict token logits
        logits = self.output_proj(output)  # [B, S, V]

        # Discard logits for image and BOS tokens; return only caption token logits
        return logits[:, image_embeds.size(1) + 1:, :]  # [B, T, V]

# --- Image projection module ---
class ImageProjector(nn.Module):
    def __init__(self, d_img, d_txt):
        super().__init__()
        self.linear = nn.Linear(d_img, d_txt)

    def forward(self, x):
        return self.linear(x)

# --- Setup model and training ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D_img = 768
D_txt = 512

#https://huggingface.co/openai/clip-vit-base-patch32
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()
clip_model.to(device)

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")


decoder = CaptionTransformerDecoder(
    embed_dim=D_txt,
    vocab_size=tokenizer.vocab_size
).to(device)

image_proj = ImageProjector(d_img=D_img, d_txt=D_txt).to(device)

optimizer = torch.optim.Adam(
    list(decoder.parameters()) + list(image_proj.parameters()),
    lr=1e-4
)

print("Loading Flickr Dataset")
builder = FlickrDataset()
builder.download_and_prepare()
print("Flickr Dataset Loaded")

clip_preprocess = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    ),
])

def transform_fn(batch):
    batch["image"] = [clip_preprocess(item) for item in batch["image"]]
    return batch

dataset = builder.as_dataset(split="test")

dataset.set_transform(transform_fn)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


# --- Training loop ---
decoder.train()
for batch in tqdm(dataloader):
    images = batch["image"].to(device)
    captions = batch["caption"]

    # Use first caption for now
    caption_texts = captions[0] # [cap[0] for cap in captions]
    tokenised = tokenizer(caption_texts, return_tensors="pt", padding="max_length", truncation=True).to(device)
    caption_token_ids = tokenised["input_ids"]  # (B, 77)

    clip_bos_embed_single = clip_model.text_model.embeddings.token_embedding.weight[0]  # usually ID 0
    assert clip_bos_embed_single.shape[0] == 512

    clip_bos_embed = clip_bos_embed_single.unsqueeze(0).unsqueeze(0)
    clip_bos_embed = clip_bos_embed.repeat(32, 1, 1)
    assert clip_bos_embed.shape == (32, 1, 512)
    #(B, 77, 512)

    with torch.no_grad():
        caption_embeddings = clip_model.text_model.embeddings(input_ids=caption_token_ids)
        assert caption_embeddings.shape == (32, 77, 512)

        caption_embeddings = caption_embeddings[:, 1:, :] #strip off the BOS
        assert caption_embeddings.shape == (32, 76, 512)

    with torch.no_grad():
        image_embeddings = clip_model.vision_model.embeddings(images)  # (B, 50, 768)
        image_embeddings = image_proj(image_embeddings)  # ADJUST TO (B, 50, 512)
        assert image_embeddings.shape == (32, 50, 512)


    logits = decoder(image_embeddings, clip_bos_embed, caption_embeddings) #:, :-1 ??

    loss = nn.CrossEntropyLoss()(
        logits.reshape(-1, logits.size(-1)),  # [B*T, V]
        caption_token_ids[:, 1:].reshape(-1)  # [B*T]
    )

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    print(f"Loss: {loss.item():.4f}")
