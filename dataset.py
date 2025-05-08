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

    output_dir = 'test_images'

    print(f"len(train_dataset):{len(train_data)}")
    print(f"len(test_data):{len(test_data)}")
    
    os.makedirs(output_dir, exist_ok=True)

    def save_random_images_with_captions(dataset, split_name):
        sampled = random.sample(range(len(dataset)), 10)
        captions_dict = {}

        n_captions = 0
        n_images = 0
        for idx, sample_idx in enumerate(sampled):
            sample = dataset[sample_idx]
            image_tensor = sample["image"]
            captions = sample["caption"]  

            n_images +=1
            n_captions += len(captions)

            img = TF.to_pil_image(image_tensor)
            filename = f"{split_name}_{idx}.png"
            img.save(os.path.join(output_dir, filename))

            captions_dict[filename] = captions

        print(f"{n_images} Images and {n_captions} captions saved to {output_dir}")

        return captions_dict

    train_captions = save_random_images_with_captions(train_data, "train")
    test_captions = save_random_images_with_captions(test_data, "test")

    with open(os.path.join(output_dir, "captions.json"), "w") as f:
        json.dump({"train": train_captions, "test": test_captions}, f, indent=2)
