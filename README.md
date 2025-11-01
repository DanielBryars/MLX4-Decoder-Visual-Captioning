# MLX4-Decoder-Visual-Captioning

Transformer decoder-only model for image captioning using CLIP vision encoder and custom decoder architecture on the Flickr8k dataset.

## Overview

This project implements an image captioning system that generates textual descriptions from images using:

- **CLIP Vision Encoder** (ViT-base-patch32) for image feature extraction
- **Custom Transformer Decoder** for caption generation
- **Decoder-only architecture** with combined attention masking
- **Flickr8k dataset** for training and evaluation

## Architecture

```
Image → CLIP ViT Encoder → Projection Layer → Transformer Decoder → Caption
         (768-dim)           (768→512)         (4 layers, 2 heads)
```

### Key Components

1. **CLIP Vision Encoder** - Pre-trained ViT-base-patch32 (frozen)
   - Outputs 50 image patch embeddings
   - Dimension: 768

2. **Projection Layer** - Projects CLIP embeddings to decoder dimension
   - Input: 768 (CLIP)
   - Output: 512 (decoder dimension)

3. **Custom Transformer Decoder** - Generates captions autoregressively
   - 4 decoder layers
   - 2 attention heads
   - 512 embedding dimension
   - Dropout: 0.1

4. **Combined Attention Masking**
   - Image tokens can attend to other image tokens
   - Image tokens cannot attend to caption tokens (future information)
   - Caption tokens use causal masking (autoregressive)
   - Caption tokens can attend to all image tokens

### Masking Strategy

The custom masking implementation ensures:
- **Causal decoding**: Each caption token only attends to previous tokens
- **Image conditioning**: Caption tokens can attend to all image patches
- **No future leakage**: Image embeddings don't see future caption tokens

Visualizations of attention masks are saved during training (`tgt_mask.png`, `padding_mask.png`).

## Dataset

**Flickr8k** - 8,000 images with 5 captions each
- Training: 6,000 images
- Validation: 1,000 images
- Test: 1,000 images

### Download Dataset

```bash
./getflickr.sh
```

This downloads and extracts the Flickr8k dataset.

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Key dependencies:
- PyTorch
- Transformers (HuggingFace)
- CLIP (openai/clip-vit-base-patch32)
- Weights & Biases
- Pillow
- tqdm

## Training

### Configuration

Hyperparameters in `01_training.py`:
```python
hyperparameters = {
    'learning_rate': 1e-4,
    'weight_decay': 0.01,
    'batch_size': 64,
    'patience': 3,         # Early stopping patience
    'num_layers': 4,       # Decoder layers
    'num_heads': 2,        # Attention heads
    'dropout': 0.1,
    'num_epochs': 10
}
```

### Run Training

```bash
python 01_training.py
```

Training features:
- Adam optimizer with weight decay
- Early stopping (patience=3)
- Weights & Biases logging
- Checkpoint saving (best validation loss)
- Attention mask visualization
- Per-epoch validation

### Training Process

The model is trained to:
1. Encode image into patch embeddings via CLIP
2. Project embeddings to decoder dimension
3. Generate captions autoregressively
4. Minimize cross-entropy loss against ground truth

## Inference

### Generate Captions

```bash
python 02_inference.py
```

or

```bash
python inference.py
```

Load a trained checkpoint and generate captions for test images.

### Example Usage

```python
from ModelFactory import ModelFactory
from transformers import CLIPModel, CLIPTokenizer

# Load model
model = ModelFactory().CreateModelFromHyperparameters(hyperparameters, vocab_size)
model.load_state_dict(torch.load("checkpoint.pth"))

# Load CLIP encoder
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

# Encode image
image_embeds = clip_model.get_image_features(pixel_values)

# Generate caption (autoregressive decoding)
caption = generate_caption(model, image_embeds, tokenizer)
```

## Project Structure

```
MLX4-Decoder-Visual-Captioning/
├── 01_training.py                    # Main training script
├── 02_inference.py                   # Inference script
├── CaptionTransformerDecoder.py      # Custom decoder implementation
├── DansDecoder.py                    # Alternative decoder
├── DecoderBlock.py                   # Decoder block components
├── FlickrDataset.py                  # Flickr8k dataset loader
├── ModelFactory.py                   # Model instantiation
├── PositionalEncoding.py             # Positional encoding module
├── ProjectEmbeddingDimension.py      # Projection layer
├── dataset.py                        # Dataset utilities
├── training.py                       # Training loop utilities
├── test_images/                      # Sample test images
├── docs/                             # Documentation
├── requirements.txt                  # Python dependencies
└── *.png                             # Attention mask visualizations
```

## Key Features

### Custom Masking

The decoder implements a sophisticated masking strategy:
- **Combined mask** for image + caption tokens
- **Causal masking** within caption sequence
- **Bidirectional attention** within image patches
- **Cross-attention** from captions to images

### Attention Visualization

During training, attention masks are saved as PNG images:
- `tgt_mask.png` - Combined decoder mask
- `default_causal_mask.png` - Standard causal mask comparison
- `padding_mask.png` - Padding mask for variable-length sequences
- `caption_attention_mask_only.png` - Text-only padding mask

### Model Checkpoints

Best model checkpoints are saved with timestamp:
```
checkpoint_YYYY_MM_DD__HH_MM_SS.pth
```

## Experiment Tracking

The project uses Weights & Biases for:
- Training loss curves
- Validation loss and accuracy
- Hyperparameter tracking
- Model artifact versioning

Project name: `MLX7-W4-VIT-CAPTIONS-106`

## Model Details

### Parameters
- Total parameters: ~7-8M (depending on vocab size)
- CLIP encoder: Frozen (not trained)
- Trainable: Projection layer + Decoder

### Tokenization
- CLIP tokenizer vocabulary
- BOS (Beginning of Sequence) token
- EOS (End of Sequence) token
- Maximum sequence length: Configurable

## Evaluation Metrics

- **Perplexity** - Measures prediction quality
- **Token Accuracy** - Per-token prediction accuracy
- **Validation Loss** - Cross-entropy loss on validation set

## Use Cases

- Learning decoder-only architectures for vision-language tasks
- Understanding attention masking in multi-modal transformers
- Image captioning with pre-trained vision encoders
- Experimenting with custom decoder implementations
- Studying autoregressive generation for sequences

## Technical Highlights

### Decoder-Only Design
Unlike encoder-decoder models, this uses a decoder-only architecture where:
- Image embeddings are prepended to caption sequence
- Single attention mechanism handles both cross-attention and self-attention
- More parameter-efficient than full encoder-decoder

### Masking Implementation
Custom masking allows fine-grained control:
```python
mask[:image_len, :image_len] = 0    # image→image allowed
mask[:image_len, image_len:] = 1    # image→caption blocked
mask[image_len:, image_len:] = causal_mask  # caption→caption (causal)
```

## References

Papers and documentation in `docs/` directory.

### Related Work
- CLIP (Contrastive Language-Image Pre-training)
- Vision Transformers (ViT)
- Decoder-only language models (GPT-style)
- Image captioning with attention

## Notes

- Requires CUDA-capable GPU for efficient training
- CLIP encoder remains frozen during training
- Attention masks are visualized for debugging purposes
- Early stopping prevents overfitting
- Batch size of 64 requires ~8-12GB GPU memory
