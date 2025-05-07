import wandb
import torch
import os 
from tqdm import tqdm 
import torch.nn.functional as F
import random
import numpy as np
import torch.nn as nn

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def evaluate(
        model,
        dataloader, 
        tokeniser, 
        clip_model, 
        device, 
        epoch=None, 
        step=None):
    
    model.eval()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_batches = 0

    loop = tqdm(dataloader, desc=f"Epoch {epoch} [Val]", leave=False)
    with torch.no_grad():
        for batch in loop:
            loss = ForwardThroughModel(model, tokeniser, clip_model, device, batch)

            total_loss += loss
            total_batches += 1
            
            wandb.log({'val/loss': loss.item()}, step=step)
            loop.set_postfix(loss=loss.item())


    avg_loss = total_loss / total_batches if total_batches > 0 else float('nan')
    accuracy = total_correct / total_samples if total_samples > 0 else float('nan')

    if step is not None:
        wandb.log({'val/avg_loss': avg_loss, 'val/avg_accuracy': accuracy}, step=step)
    
    return avg_loss,accuracy

def train_one_epoch(
        model,
        dataloader, 
        tokeniser, 
        clip_model, 
        optimizer, 
        device, 
        epoch, 
        step_offset=0):
    
    model.train()
    step = step_offset

    loop = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", leave=False)
    for batch in loop:        
        loss = ForwardThroughModel(model, tokeniser, clip_model, device, batch)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        wandb.log({'train/loss': loss.item()}, step=step)
        loop.set_postfix(loss=f"{loss.item():.4f}")

    return step

def ForwardThroughModel(model, tokeniser, clip_model, device, batch):
    images = batch["image"].to(device)
    captions = batch["caption"]

    batch_size = images.shape[0]

        # Use first caption for now
        # TODO use all captions
    caption_texts = captions[0] # [cap[0] for cap in captions]
        
    tokenised = tokeniser(caption_texts, return_tensors="pt", padding="max_length", truncation=True).to(device)
    caption_token_ids = tokenised["input_ids"]

    clip_bos_embed_single = clip_model.text_model.embeddings.token_embedding.weight[0]  # TODO Fix this, assuming 0 looks very brittle to me
    clip_bos_embed = clip_bos_embed_single.unsqueeze(0).unsqueeze(0)
    clip_bos_embed = clip_bos_embed.repeat(batch_size, 1, 1)
        
    with torch.no_grad():
        caption_embeddings = clip_model.text_model.embeddings(input_ids=caption_token_ids[:, :-1]) #Note [:, :-1] I don't to pass in last token, we're going to predict that
        caption_embeddings = caption_embeddings[:, 1:, :] #strip off the BOS
            
        image_embeddings = clip_model.vision_model.embeddings(images)  # (B, 50, 768) This includes the CLS at the start   

    logits = model(image_embeddings, clip_bos_embed, caption_embeddings) #:, :-1 ??
    assert tokeniser.vocab_size == logits.shape[2]

        #labels = caption_token_ids[:, 1:]
    labels = caption_token_ids[:, 1:1 + caption_embeddings.shape[1]]


    logits_reshaped = logits.reshape(-1, logits.size(-1))
    labels_reshaped = labels.reshape(-1)

        #print(f"logits.shape: {logits.shape}")
        #print(f"logits_reshaped.shape: {logits_reshaped.shape}")

        #print(f"labels.shape: {labels.shape}")
        #print(f"labels_reshaped.shape: {labels_reshaped.shape}")

    loss = nn.CrossEntropyLoss()(
            logits_reshaped,
            labels_reshaped)
    
    return loss

def save_checkpoint(model, hyperparameters, epoch, ts):
    checkpoint_dir = './checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    model_type = type(model).__name__
    descriptive_name = f'ts.{ts}.epoch.{epoch + 1}.{model_type}'
    checkpoint_name = f'{descriptive_name}.pth'
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

    print(f"Saving '{checkpoint_path}'")
    torch.save({
        'model': model.state_dict(),
        'epoch': epoch,
        'hyperparameters': hyperparameters
    }, checkpoint_path)

    # Create wandb artifact and log it
    artifact = wandb.Artifact(
        name=descriptive_name,
        type='model',
        description=f'{model_type} model weights from epoch {epoch + 1}, timestamp {ts}')
    
    #actually upload the artifact!!!!
    artifact.add_file(checkpoint_path)
    wandb.log_artifact(artifact)