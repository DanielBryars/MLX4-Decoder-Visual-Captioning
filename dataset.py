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
    import random
    import os
    from PIL import Image
    import torchvision.transforms.functional as TF
    import json

    num_images = 10
    output_dir = 'test_images'
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset inside __main__
    sampl_dataset = builder.as_dataset(split="test")

    sample_clip_preprocess = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        ),
    ])

#    sampl_dataset = dataset.with_transform(
#        lambda example: {
#            "image": sample_clip_preprocess(example["image"]),
#            "caption": example["caption"]
#        })
    
    sampl_dataset = dataset.with_transform(
    lambda example: {
        "image": clip_preprocess(example["image"][0]) if isinstance(example["image"], list) else clip_preprocess(example["image"]),
        "caption": example["caption"]
    }
)

    sample_indexes = random.sample(range(len(sampl_dataset)), num_images)
    captions_dict = {}

    n_captions = 0
    n_images = 0
    for idx, sample_idx in enumerate(sample_indexes):
        row = sampl_dataset[sample_idx]
        image_tensor = row["image"]
        captions = row["caption"]

        n_images +=1
        n_captions += len(captions)

        img = TF.to_pil_image(image_tensor)
        filename = f"{idx}.png"
        img.save(os.path.join(output_dir, filename))

        captions_dict[filename] = captions

    with open(os.path.join(output_dir, "captions.json"), "w") as f:
        json.dump(captions, f, indent=2)

    print(f"{n_images} Images and {n_captions} captions saved to '{output_dir}'")
