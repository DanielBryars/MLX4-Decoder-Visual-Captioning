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

            images = batch["image"].to(device)
            captions = batch["caption"]

            batch_size = images.shape[0]

            flat_captions = []
            num_captions_per_image = len(captions)
            
            for image_idx in range(batch_size):
                for caption_idx in range(len(captions)):
                    flat_captions.append(captions[caption_idx][image_idx])
                
            images_repeated = images.repeat_interleave(num_captions_per_image, dim=0)

            loss = ForwardThroughModel(model, tokeniser, clip_model, device, images_repeated, flat_captions)

            total_loss += loss
            total_batches += 1
            
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

        images = batch["image"].to(device)
        captions = batch["caption"]

        batch_size = images.shape[0]

        flat_captions = []
        num_captions_per_image = len(captions)
        
        for image_idx in range(batch_size):
            for caption_idx in range(len(captions)):
                flat_captions.append(captions[caption_idx][image_idx])
            
        images_repeated = images.repeat_interleave(num_captions_per_image, dim=0)

        loss = ForwardThroughModel(model, tokeniser, clip_model, device, images_repeated, flat_captions)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        wandb.log({'train/loss': loss.item()}, step=step)
        loop.set_postfix(loss=f"{loss.item():.4f}")
        step += 1

    return step


def ForwardThroughModel(model, tokeniser, clip_model, device, images, caption_texts):
    # Tokenise with BOS and EOS (CLIP adds BOS automatically)
    tokenised = tokeniser(
        caption_texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        add_special_tokens=True
    ).to(device)

    input_ids = tokenised["input_ids"]             # [B, T+1]
    attention_mask = tokenised["attention_mask"]   # [B, T+1]

    # 🔍 Debug: print a few examples
#    for i in range(min(10, len(caption_texts))):
#        print(f"\n--- Caption {i+1} ---")
#        print(f"Original         : {caption_texts[i]}")
#        print(f"Input IDs        : {input_ids[i].tolist()}")
#        print(f"Decoded (all)    : {tokeniser.decode(input_ids[i], skip_special_tokens=False)}")
#        print(f"Decoded (no spec): {tokeniser.decode(input_ids[i], skip_special_tokens=True)}")

    # Prepare inputs and labels for next-token prediction
    inputs_for_embedding = input_ids[:, :-1]       # [B, T]
    labels = input_ids[:, 1:]                      # [B, T]
    attention_mask = attention_mask[:, :-1]        # [B, T] (match caption_embeds)

    # Get text embeddings (no gradient needed)
    with torch.no_grad():
        caption_embeddings = clip_model.text_model.embeddings(input_ids=inputs_for_embedding)  # [B, T, D]
        image_embeddings = clip_model.vision_model.embeddings(images)  # [B, 50, 768]

    # Forward through decoder
    logits = model(
        image_embeddings,              # [B, 50, 768]
        caption_embeddings,            # [B, T, 512]
        caption_attention_mask=attention_mask  # [B, T]
    )  # returns [B, T, V]

    # Sanity check
    assert logits.shape[:2] == labels.shape, f"Mismatch: logits {logits.shape}, labels {labels.shape}"

    
    with torch.no_grad():
        probs = torch.softmax(logits, dim=-1)  # [B, T, V]
        B, T, V = probs.shape

        table = wandb.Table(columns=["step", "true_token", "true_id", "top1", "top1_prob", "top5", "top5_probs"])

        for b in range(min(B, 1)):  # Limit to first example for clarity
            true_ids = labels[b].tolist()
            for t in range(T):
                true_id = true_ids[t]
                if true_id == tokeniser.pad_token_id:
                    continue  # Skip padding positions

                true_tok = tokeniser.decode([true_id])
                topk_probs, topk_ids = probs[b, t].topk(5)
                topk_tokens = [tokeniser.decode([i.item()]) for i in topk_ids]
                topk_probs = [round(p.item(), 4) for p in topk_probs]
                
                table.add_data(
                    t,
                    true_tok,
                    true_id,
                    topk_tokens[0],
                    topk_probs[0],
                    topk_tokens,
                    topk_probs
                )

        wandb.log({"token_predictions": table})

    # Compute loss
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokeniser.pad_token_id)
    loss = loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

    return loss


def ForwardThroughModelOLD(model, tokeniser, clip_model, device, images, caption_texts):
    tokenised = tokeniser(caption_texts, return_tensors="pt", padding="max_length", truncation=True).to(device)
    caption_token_ids = tokenised["input_ids"]
    attention_mask = tokenised['attention_mask'] #how do I use this?

    #clip_bos_embed_single = clip_model.text_model.embeddings.token_embedding.weight[0]  # TODO Fix this, assuming 0 looks very brittle to me
    #clip_bos_embed = clip_bos_embed_single.unsqueeze(0).unsqueeze(0)
    #clip_bos_embed = clip_bos_embed.repeat(batch_size, 1, 1)
        
    with torch.no_grad():
        caption_embeddings = clip_model.text_model.embeddings(input_ids=caption_token_ids[:, :-1]) #Note [:, :-1] I don't to pass in last token, we're going to predict that
        #caption_embeddings = caption_embeddings[:, 1:, :] #strip off the BOS
            
        image_embeddings = clip_model.vision_model.embeddings(images)  # (B, 50, 768) This includes the CLS at the start   


    logits = model(image_embeddings, caption_embeddings, caption_attention_mask=attention_mask[:, :-1]) #:, :-1 ??
    assert tokeniser.vocab_size == logits.shape[2]

    #labels = caption_token_ids[:, 1:caption_embeddings.shape[1]]
    labels = caption_token_ids[:, 1:]
    
    #labels = caption_token_ids[:, 1:1 + caption_embeddings.shape[1]]

    logits_reshaped = logits.reshape(-1, logits.size(-1))
    labels_reshaped = labels.reshape(-1)

    #print(f"logits.shape: {logits.shape}")
    #print(f"logits_reshaped.shape: {logits_reshaped.shape}")

    #print(f"labels.shape: {labels.shape}")
    #print(f"labels_reshaped.shape: {labels_reshaped.shape}")

    pad_id = tokeniser.pad_token_id
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)

    loss = loss_fn(logits_reshaped, labels_reshaped)
    
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