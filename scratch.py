import torch
from transformers import CLIPModel, CLIPTokenizer
from torchvision import transforms
from torch.utils.data import DataLoader

from FlickrDataset import FlickrDataset

#https://huggingface.co/openai/clip-vit-base-patch32
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")


unk_token="<|endoftext|>",
bos_token="<|startoftext|>",
eos_token="<|endoftext|>",
pad_token="<|endoftext|>"


#v = transformers.ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

count_params = lambda m: sum(p.numel() for p in m.parameters())

#print(f"CLIP: num parameters:{count_params(c)} ")
#print (c)

print("Loading Flickr Dataset")
builder = FlickrDataset()

builder.download_and_prepare()


print("Flickr Dataset Loaded")


'''
Input size: 224×224

Mean: [0.48145466, 0.4578275, 0.40821073]

Std: [0.26862954, 0.26130258, 0.27577711]

'''

#resize = transforms.Compose([
#    transforms.Resize((224, 224)),
#    transforms.ToTensor(),
#])

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

# Example iteration
for batch in dataloader:
    
    #print(captions[0])
    images = batch["image"]
    captions = batch["caption"]

    print(images.shape)                         # Should be torch.Size([32, 3, 224, 224])
    print(len(captions))                        # Should be 32
    print(len(captions[0]))                     # Should be 5
    #print(captions[0])   

    #.to(device)

    with torch.no_grad():

        image_features = model.get_image_features(pixel_values=batch["image"],output_hidden_states=True)
        print(f"image_features shape:{image_features.shape}")

        image_embeddings = model.vision_model.embeddings(batch["image"])
        print(f"image_embedding shape:{image_embeddings.shape}")

        text_embeddings_set = []
        for captionset in captions:

            tokenised = tokenizer(captionset,return_tensors="pt", padding="max_length", truncation=True)
            #print(f"tokenised shape:{tokenised.shape}")

            text_embeddings = model.text_model.embeddings(input_ids=tokenised["input_ids"])
            print(f"text_embeddings shape:{text_embeddings.shape}")
            text_embeddings_set.append(text_embeddings)
            
            

    break


