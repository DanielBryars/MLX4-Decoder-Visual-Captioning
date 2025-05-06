import wandb
import torch
import os 
from tqdm import tqdm 
import torch.nn.functional as F
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def evaluate(model, dataloader, device, epoch=None, step=None):
    model.eval()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_batches = 0

    loop = tqdm(dataloader, desc=f"Epoch {epoch} [Val]", leave=False)
    with torch.no_grad():
        for batch in loop:
            images, labels = [x.to(device) for x in batch]

            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            total_loss += loss.item()
            total_batches += 1
            loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / total_batches if total_batches > 0 else float('nan')
    accuracy = total_correct / total_samples if total_samples > 0 else float('nan')

    if step is not None:
        wandb.log({'val/loss': avg_loss, 'val/accuracy': accuracy}, step=step)

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

    pbar = tqdm(dataloader)
    for batch in pbar:
        images = batch["image"].to(device)
        captions = batch["caption"]

        # Use first caption for now
        caption_texts = captions[0] # [cap[0] for cap in captions]
        tokenised = tokeniser(caption_texts, return_tensors="pt", padding="max_length", truncation=True).to(device)
        caption_token_ids = tokenised["input_ids"]  # (B, 77)

        clip_bos_embed_single = clip_model.text_model.embeddings.token_embedding.weight[0]  # usually ID 0
        
        clip_bos_embed = clip_bos_embed_single.unsqueeze(0).unsqueeze(0)
        clip_bos_embed = clip_bos_embed.repeat(32, 1, 1)
        
        with torch.no_grad():
            caption_embeddings = clip_model.text_model.embeddings(input_ids=caption_token_ids)
            caption_embeddings = caption_embeddings[:, 1:, :] #strip off the BOS
            
            image_embeddings = clip_model.vision_model.embeddings(images)  # (B, 50, 768)    

        logits = model(image_embeddings, clip_bos_embed, caption_embeddings) #:, :-1 ??
        
        assert tokeniser.vocab_size == logits.shape[2]

        loss = nn.CrossEntropyLoss()(
            logits.reshape(-1, logits.size(-1)),  # [B*T, V]
            caption_token_ids[:, 1:].reshape(-1)  # [B*T]
        )

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return step

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