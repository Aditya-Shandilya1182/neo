import torch
import pickle
import tiktoken
from datasets import load_dataset
from tiktoken import get_encoding
from tqdm import tqdm
from neo import ModelConfig, Neo
from torch.amp import GradScaler, autocast

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
            losses[k] = loss.mean().item()

        out[split] = losses.mean().item()

    model.train()
    return out

max_iters = config.max_iters
gradient_accumulation_steps = 8
eval_iters = config.eval_iters
block_size = config.block_size
batch_size = config.batch_size
max_grad_norm = 1.0

model = Neo(config)
model = nn.DataParallel(model, device_ids=[0, 1])
model = model.to(device)
model = torch.compile(model)


print(f"Using devices: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
print(next(model.parameters()).device)

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
scaler = GradScaler()

train_losses = []
val_losses = []

for iter in range(max_iters):
    print(iter)
    
    if iter % eval_iters == 0:
        losses = estimate_loss()
        print(f"step: {iter}, train loss: {losses['train']:.3f}, val loss: {losses['val']:.3f}")

    ix = torch.randint(train_len - block_size, (batch_size,))
    x = torch.stack([train_encoded[i:i+block_size] for i in ix])
    y = torch.stack([train_encoded[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)

    with autocast(device_type='cuda', dtype=torch.float16):
        logits, loss = model(x, y)
        loss = loss / gradient_accumulation_steps

    loss_mean = loss.mean()
    scaler.scale(loss_mean).backward()

    if (iter + 1) % gradient_accumulation_steps == 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        print(f"Loss at step {iter + 1}: {loss_mean.item() * gradient_accumulation_steps:.3f}")
        train_losses.append(loss_mean.item() * gradient_accumulation_steps)


with open('model-03.pkl', 'wb') as f:
    pickle.dump(model, f)
print('model saved')

prompt = 'Hello! Can you see me?'
context = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)  # (1, T)

# Use .module if model is wrapped in DataParallel
if isinstance(model, torch.nn.DataParallel):
    generated = model.module.generate(context, max_new_tokens=100)
else:
    generated = model.generate(context, max_new_tokens=100)

# Decode and print output
generated_chars = tokenizer.decode(generated[0].tolist())
print(generated_chars)
