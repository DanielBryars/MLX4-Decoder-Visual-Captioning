import transformers
c = transformers.CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
#v = transformers.ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

count_params = lambda m: sum(p.numel() for p in m.parameters())

print(f"CLIP: num parameters:{count_params(c)} ")
print (c)
'''


from torchvision import transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = Flickr30kDataset(
    root_dir='data/flickr30k/flickr30k-images',
    json_file='data/flickr30k/dataset_flickr30k.json',
    split='train',
    transform=transform
)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Example iteration
for images, captions in dataloader:
    print(captions[0])
    break


'''