#!/bin/bash

set -e  # Exit on error

# Create target directory
mkdir -p data/flickr30k
cd data/flickr30k

# Download the annotation file
ANNOTATION_FILE="dataset_flickr30k.json"
if [ ! -f "$ANNOTATION_FILE" ]; then
  echo "Downloading Flickr30k annotations JSON from Hugging Face..."
  wget https://huggingface.co/datasets/openflamingo/eval_benchmark/resolve/main/flickr30k/dataset_flickr30k.json
else
  echo "Annotation JSON already exists, skipping download."
fi
