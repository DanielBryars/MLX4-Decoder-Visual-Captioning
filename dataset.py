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

    
    sampl_dataset = test_data
    sample_indexes = random.sample(range(len(sampl_dataset)), num_images) #not really needed since the data is randomised anyway


    alternative_clip_preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    def alternative_clip_preprocess_fn(batch):
        batch["image"] = [alternative_clip_preprocess(item) for item in batch["image"]]
        return batch

    sampl_dataset.set_transform(alternative_clip_preprocess_fn)

    for idx, sample_idx in enumerate(sample_indexes):
        row = sampl_dataset[sample_idx]
        img_id = row["img_id"]
        image_tensor = row["image"]
        captions = row["caption"]

        img = TF.to_pil_image(image_tensor)
        img.save(os.path.join(output_dir, f"{img_id}.alternative.png"))



    identity_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    def identity_transform_fn(batch):
        batch["image"] = [identity_transform(item) for item in batch["image"]]
        return batch


    sampl_dataset.set_transform(identity_transform_fn)

    for idx, sample_idx in enumerate(sample_indexes):
        row = sampl_dataset[sample_idx]
        img_id = row["img_id"]
        image_tensor = row["image"]
        captions = row["caption"]

        img = TF.to_pil_image(image_tensor)
        img.save(os.path.join(output_dir, f"{img_id}.original.png"))


    resize_transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    
    def resize_transform_fn(batch):
        batch["image"] = [resize_transform(item) for item in batch["image"]]
        return batch


    sampl_dataset.set_transform(resize_transform_fn)

    n_captions = 0
    n_images = 0
    for idx, sample_idx in enumerate(sample_indexes):
        row = sampl_dataset[sample_idx]
        img_id = row["img_id"]
        image_tensor = row["image"]
        captions = row["caption"]

        n_images +=1
        n_captions += len(captions)

        img = TF.to_pil_image(image_tensor)
        img.save(os.path.join(output_dir, f"{img_id}.resized.png"))
        
        with open(os.path.join(output_dir, f"{img_id}.json"), "w") as f:
            json.dump(captions, f, indent=2)

    print(f"{n_images} Images and {n_captions} captions saved to '{output_dir}'")
