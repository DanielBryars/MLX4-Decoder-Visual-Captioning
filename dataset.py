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


from datasets import load_dataset

dataset = builder.as_dataset(split="test") #this is ALL the data even though it says test
dataset.set_transform(transform_fn)

split_dataset = dataset.train_test_split(test_size=0.2)

train_data = split_dataset['train']
test_data = split_dataset['test']

if __name__ == "__main__":
    print(f"len(train_dataset):{len(train_data)}")
    print(f"len(test_data):{len(test_data)}")
    
