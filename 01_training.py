import torch
import torch.nn as nn
from tqdm import tqdm
from DecoderBlock import DecoderBlock
from PositionalEncoding import PositionalEncoding, causal_mask

import torch
from transformers import CLIPModel, CLIPTokenizer
from torchvision import transforms
from torch.utils.data import DataLoader

from FlickrDataset import FlickrDataset

#https://huggingface.co/openai/clip-vit-base-patch32
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")



# --- Utility function to prepare decoder inputs and labels ---
def prepare_inputs_and_labels_with_sos(image_embeds, target_ids, ignore_index=-100):
    """
    - image_embeds: (B, 50, D)
    - target_ids: (B, 77)
    Returns:
    - decoder_inputs: (B, 51, D)
    - labels: (B, 51)
    """
    B = image_embeds.size(0)
    labels = torch.full((B, 51), ignore_index, dtype=torch.long).to(target_ids.device)

    for i in range(B):
        t = target_ids[i][:51]  # clip if longer than decoder output
        labels[i, :len(t)] = t

    return image_embeds, labels

# --- Decoder model with learnable SOS ---
class SimpleDecoderWithSOS(nn.Module):
    def __init__(self, d_model, n_heads, num_layers, vocab_size, max_len=128):
        super().__init__()
        self.sos_embed = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, n_heads) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):  # x: (B, 50, D)
        B = x.size(0)
        sos = self.sos_embed.expand(B, 1, -1)  # (B, 1, D)
        x = torch.cat([x, sos], dim=1)        # (B, 51, D)
        x = self.pos_enc(x)

        T = x.size(1)
        mask = causal_mask(T).to(x.device)

        for layer in self.layers:
            x = layer(x, attn_mask=mask)

        x = self.ln_f(x)
        return self.head(x)  # (B, 51, vocab_size)

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

decoder = SimpleDecoderWithSOS(
    d_model=D_txt,
    n_heads=8,
    num_layers=6,
    vocab_size=tokenizer.vocab_size,
    max_len=128
).to(device)

image_proj = ImageProjector(d_img=D_img, d_txt=D_txt).to(device)

optimizer = torch.optim.Adam(
    list(decoder.parameters()) + list(image_proj.parameters()),
    lr=1e-4
)
loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

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

    with torch.no_grad():
        image_emb = model.vision_model.embeddings(images)  # (B, 50, 768)

    image_proj_out = image_proj(image_emb)  # (B, 50, 512)

    # Use first caption for now
    caption_texts = [cap[0] for cap in captions]
    tokenised = tokenizer(caption_texts, return_tensors="pt", padding="max_length", truncation=True).to(device)
    text_input_ids = tokenised["input_ids"]  # (B, 77)

    decoder_inputs, labels = prepare_inputs_and_labels_with_sos(image_proj_out, text_input_ids)

    logits = decoder(decoder_inputs)  # (B, 51, vocab_size)
    loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    print(f"Loss: {loss.item():.4f}")
