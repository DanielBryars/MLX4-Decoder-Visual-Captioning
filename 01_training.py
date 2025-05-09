import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import CLIPModel, CLIPTokenizer

from torch.utils.data import DataLoader
from FlickrDataset import FlickrDataset
from CaptionTransformerDecoder import CaptionTransformerDecoder
from ModelFactory import ModelFactory
from ProjectEmbeddingDimension import ProjectEmbeddingDimension
import wandb
import torch
import datetime
import torch
import wandb
import torch
import wandb
import torch
import os 
from tqdm import tqdm 
import torch.nn.functional as F
import random
import numpy as np
import torch.nn as nn
from dataset import *
from training import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ts = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
print(f"Using device:{device}")
set_seed()

hyperparameters = {
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'batch_size': 12,
        'patience': 3,
        'num_layers': 4,
        'num_heads':2,
        'dropout':0.1,
        'num_epochs':10
}

wandb.init(project='MLX7-W4-VIT-CAPTIONS-103', config=hyperparameters)
config = wandb.config

D_img = 768
D_txt = 512

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()
clip_model.to(device)

model = ModelFactory().CreateModelFromHyperparameters(hyperparameters, tokenizer.vocab_size).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")

train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=hyperparameters['batch_size'])
val_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=hyperparameters['batch_size'])

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=hyperparameters['learning_rate'], 
    weight_decay=hyperparameters['weight_decay']
)

step = 0
best_val_loss = float('inf')
epochs_no_improve = 0
patience= hyperparameters['patience']
epoch_pbar = tqdm(range(1, hyperparameters['num_epochs'] + 1))

for epoch in epoch_pbar:
    step = train_one_epoch          (model, train_loader, tokenizer, clip_model, optimizer, device, epoch, step_offset=step)
    val_loss, accuracy  = evaluate  (model, val_loader, tokenizer, clip_model, device, epoch=epoch, step=step)

    print(f"Epoch {epoch} complete | Val Loss: {val_loss:.4f} | Accuracy: {accuracy:.4f}")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        save_checkpoint(model, hyperparameters, epoch, ts)
    else:
        epochs_no_improve += 1
        print(f"No improvement. Early stop patience: {epochs_no_improve}/{patience}")
    if epochs_no_improve >= patience:
        print("Early stopping triggered.")
        break