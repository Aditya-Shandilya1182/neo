import torch
import pickle
import tiktoken
from datasets import load_dataset
from tiktoken import get_encoding
from tqdm import tqdm
from neo import ModelConfig, Neo

config = ModelConfig()
device = config.device
model = Neo(config)
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

dataset = load_dataset("roneneldan/TinyStories")

tokenizer = tiktoken.get_encoding("gpt2")

encoded_train_chunks = []
for sample in dataset['train']:
    encoded_train_chunks.append(tokenizer.encode(sample['text']))
train_encoded = torch.tensor([token for chunk in encoded_train_chunks for token in chunk], dtype=torch.long)

encoded_val_chunks = []
for sample in dataset['validation']:
    encoded_val_chunks.append(tokenizer.encode(sample['text']))
val_encoded = torch.tensor([token for chunk in encoded_val_chunks for token in chunk], dtype=torch.long)

train_encoded.to(device)
val_encoded.to(device)

eval_iters = config.eval_iters
block_size = config.block_size
batch_size = config.batch_size

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval() 
    for split in ['train', 'val']:
        data = train_encoded if split == 'train' else val_encoded
        
        if data.size(0) <= block_size:
            raise ValueError(f"{split.capitalize()} dataset size is too small for the requested block size.")
        
        losses = torch.zeros(eval_iters)
        
        for k in range(eval_iters):
            ix = torch.randint(0, data.size(0) - block_size, (batch_size,))
            x = torch.stack([data[i:i+block_size] for i in ix])
            y = torch.stack([data[i+1:i+block_size+1] for i in ix])

            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        
        out[split] = losses.mean().item()
    
    model.train()  
    return out

max_iters = config.max_iters
gradient_accumulation_steps = 8  
for iter in range(max_iters):
    print(iter)
    if iter % eval_iters == 0:
        losses = estimate_loss()
        print(f"step: {iter}, train loss: {losses['train']:.3f}, val loss: {losses['val']:.3f}")

    ix = torch.randint(len(train_encoded) - block_size, (batch_size,))
    x = torch.stack([train_encoded[i:i+block_size] for i in ix])
    y = torch.stack([train_encoded[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)

    logits, loss = model.forward(x, y)
    
    loss = loss / gradient_accumulation_steps  
    loss.backward()

    if (iter + 1) % gradient_accumulation_steps == 0:  
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)  

    if (iter + 1) % gradient_accumulation_steps == 0:
        print(f"Loss at step {iter + 1}: {loss.item() * gradient_accumulation_steps:.3f}")


with open('model-03.pkl', 'wb') as f:
    pickle.dump(model, f)
print('model saved')

prompt = 'Hello! Can you see me?'
context = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device)
generated_chars = tokenizer.decode(model.generate(context.unsqueeze(0), max_new_tokens=100)[0].tolist())
print(generated_chars)

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

total_params, trainable_params = count_parameters(model)
print(total_params)
print(trainable_params)
