import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from CaptionTransformerDecoder import CaptionTransformerDecoder
from ProjectEmbeddingDimension import ProjectEmbeddingDimension
from ModelFactory import ModelFactory 


snapshot_path = "checkpoints/ts.2025_05_07__15_59_59.epoch.21.CaptionTransformerDecoder.pth"
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

modelFactory = ModelFactory()

model = modelFactory.CreateFromSnapshot(snapshot_path, tokenizer.vocab_size)
model.to(device)
model.eval()

image_path = "test_images/1271.original.png"
image = Image.open(image_path).convert("RGB")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
inputs = processor(images=image, return_tensors="pt").to(device)

with torch.no_grad():
    image_features = clip_model.get_image_features(**inputs)  # [1, 512]
    image_embeds = clip_model.vision_model.embeddings(inputs["pixel_values"])  # [1, 50, 768]
    image_proj = ProjectEmbeddingDimension(d_from=768, d_to=model.embed_dim).to(device)
    image_proj_out = image_proj(image_embeds)  # [1, 50, model.embed_dim]

# --- Decode ---
max_len = 50
bos_token_id = tokenizer.bos_token_id
eos_token_id = tokenizer.eos_token_id
generated_ids = [bos_token_id]

for _ in range(max_len):
    input_ids = torch.tensor(generated_ids, device=device).unsqueeze(0)  # [1, T]
    with torch.no_grad():
        caption_embeds = clip_model.text_model.embeddings(input_ids=input_ids)  # [1, T, D]
        logits = model(image_embeds, caption_embeds)  # [1, T, vocab]
        next_token_logits = logits[0, -1, :]  # last token's logits
        next_token = torch.argmax(next_token_logits).item()
    generated_ids.append(next_token)
    if next_token == eos_token_id:
        break

# --- Decode tokens to text ---
caption = tokenizer.decode(generated_ids[1:], skip_special_tokens=True)  # skip BOS
print("Generated caption:", caption)

import matplotlib.pyplot as plt

plt.imshow(image)
plt.axis('off')
plt.title(caption, fontsize=12)
plt.show()