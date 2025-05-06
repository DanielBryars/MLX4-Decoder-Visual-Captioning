from FlickrDataset import FlickrDataset
from torchvision import transforms

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

train_data = builder.as_dataset(split="train")
train_data.set_transform(transform_fn)

test_data = builder.as_dataset(split="test")
test_data.set_transform(transform_fn)
